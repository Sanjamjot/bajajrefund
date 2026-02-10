"""
Professional Refund Automation System
Processes refund.json file with Python + Gemini AI fallback

Run: python main.py
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    print("   Please create .env file with: GEMINI_API_KEY=your_api_key_here")
    exit(1)

GEMINI_MODEL = "gemini-2.5-flash"
CONFIDENCE_THRESHOLD = 70

genai.configure(api_key=GEMINI_API_KEY)


class ReportType(str, Enum):
    FFR = "FFR"
    FPR = "FPR"
    CPR = "CPR"
    BPR = "BPR"
    BFR = "BFR"


REPORT_EVENT_MAP = {
    ReportType.FFR: {"MAGNUS VAS SUPREME LITE -FFR"},
    ReportType.FPR: {"MAGNUS VAS SUPREME LITE -FPR"},
    ReportType.CPR: {"MAGNUS VAS SUPREME LITE -CPR"},
    ReportType.BPR: {"MAGNUS VAS SUPREME LITE -BPR"},
    ReportType.BFR: {"MAGNUS VAS SUPREME LITE -BFR"},
}

REFUND_EVENTS = {
    "FFR REFUND",
    "FPR REFUND",
    "CPR REFUND",
    "BPR REFUND",
    "BFR REFUND",
    "VAS REFUND",
}


def normalize_transaction_reports(soa: Dict[str, Any]) -> Dict[str, Any]:
    normalized = []
    unknown_report_events = []
    
    for tr in soa.get("transactionReports", []):
        tr = tr.copy()
        event = tr.get("event", "").strip()
        
        found_match = False
        for report_type, aliases in REPORT_EVENT_MAP.items():
            if event in aliases:
                tr["normalized_event"] = "VAS_REPORT"
                tr["report_type"] = report_type.value
                tr["debitAmount"] = tr.get("debitAmount", 0)
                found_match = True
                break
        
        if not found_match and "MAGNUS VAS" in event:
            unknown_report_events.append({
                "event": event,
                "date": tr.get("transactionDate", "N/A"),
                "amount": tr.get("debitAmount", 0)
            })
        
        normalized.append(tr)
    
    soa["transactionReports"] = normalized
    
    if unknown_report_events:
        raise ValueError(f"Unknown VAS report events detected. AI analysis required: {unknown_report_events}")
    
    return soa


class RefundRuleEngine:
    def __init__(self, soa: Dict[str, Any]):
        self.soa = soa
        self.transactions = soa.get("transactionReports", [])
        self.summary = soa.get("soaSummaryReports", [])

    def loan_is_active(self) -> bool:
        return self.soa.get("status", "").upper() == "ACTIVE"

    def get_latest_report_charge(self) -> Optional[Dict[str, Any]]:
        charges = [
            t for t in self.transactions
            if t.get("normalized_event") == "VAS_REPORT"
        ]
        if not charges:
            return None
        return max(
            charges,
            key=lambda x: datetime.strptime(x["transactionDate"], "%Y-%m-%d")
        )

    def refund_already_processed(self, charge_date: str) -> bool:
        for t in self.transactions:
            if t.get("event", "").upper() in REFUND_EVENTS:
                if t.get("transactionDate") > charge_date:
                    return True
        return False

    def calculate_deductions(self) -> float:
        penal = bounce = 0.0
        for comp in self.summary:
            name = comp.get("component", "").lower()
            overdue = float(comp.get("overDue", 0))
            if "penal" in name:
                penal += overdue
            elif "bounce" in name:
                bounce += overdue
        return penal + bounce

    def decide(self) -> Dict[str, Any]:
        if not self.loan_is_active():
            return self._no_refund("Loan is not active")

        charge = self.get_latest_report_charge()
        if not charge:
            return self._no_refund("No report charge found")

        charge_date = charge["transactionDate"]
        if self.refund_already_processed(charge_date):
            return self._no_refund("Refund already processed")

        deductions = self.calculate_deductions()
        charge_amount = float(charge.get("debitAmount", 0))
        refund_amount = charge_amount - deductions

        if refund_amount <= 0:
            return self._no_refund("Charge adjusted against penalties")

        return {
            "decision": "AUTO_REFUND",
            "refund_eligible": True,
            "refund_amount": refund_amount,
            "reason": "Valid report charge found"
        }

    def _no_refund(self, reason: str) -> Dict[str, Any]:
        return {
            "decision": "NO_REFUND",
            "refund_eligible": False,
            "refund_amount": 0,
            "reason": reason
        }


def process_refund_case(soa_response: Dict[str, Any]) -> Dict[str, Any]:
    soa = soa_response.get("statementOfAccount", soa_response)
    soa = normalize_transaction_reports(soa)
    engine = RefundRuleEngine(soa)
    result = engine.decide()
    return {
        "finReference": soa.get("finReference"),
        "decision": result,
        "metadata": {
            "engine": "deterministic_refund_engine_v1",
            "timestamp": datetime.utcnow().isoformat()
        }
    }


class GeminiAI:
    
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "max_output_tokens": 1024,
            },
            system_instruction="""You are a financial analyst AI. 
You MUST respond with ONLY valid JSON. 
Never include any text before or after the JSON object.
Never use markdown formatting.
Just pure JSON."""
        )
    
    def analyze(self, soa_data: Dict[str, Any], error: str, error_type: str) -> Dict[str, Any]:
        
        soa = soa_data.get("statementOfAccount", {})
        
        prompt = f"""Analyze this loan refund case that failed processing.

ERROR: {error_type} - {error}

LOAN DATA:
- Reference: {soa.get('finReference', 'N/A')}
- Status: {soa.get('status', 'N/A')}
- Customer: {soa.get('custShrtName', 'N/A')}
- Has Transactions: {len(soa.get('transactionReports', []))} records
- Has Summary: {len(soa.get('soaSummaryReports', []))} records

Respond with ONLY this JSON structure (no other text):
{{
    "decision": "AUTO_REFUND",
    "refund_amount": 1250,
    "reason": "Valid FFR charge found despite date error",
    "confidence": 75,
    "ai_reasoning": "Despite invalid date format, charge data is valid",
    "data_issues": ["Invalid date format in transaction"]
}}

Rules:
- decision: "AUTO_REFUND" or "NO_REFUND" only
- refund_amount: number (0 if denying)
- confidence: 0-100 integer
- If data is too corrupted, use low confidence (below 70)

Respond now with ONLY the JSON:"""
        
        try:
            print("   🤖 Calling Gemini API...")
            
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
            
            raw_text = response.text.strip()
            print(f"   ✓ Received response ({len(raw_text)} characters)")
            
            result = self._parse_gemini_response(raw_text)
            
            result["handled_by"] = "GEMINI_AI_FALLBACK"
            result["refund_eligible"] = result["decision"] == "AUTO_REFUND"
            result["requires_manual_review"] = result["confidence"] < CONFIDENCE_THRESHOLD
            result["original_error"] = error
            
            result.setdefault("refund_amount", 0)
            result.setdefault("data_issues", [])
            result.setdefault("ai_reasoning", "AI analysis completed")
            
            if result["requires_manual_review"]:
                original_reason = result["reason"]
                result["decision"] = "MANUAL_REVIEW"
                result["reason"] = f"Low confidence ({result['confidence']}%) - requires human review. Original: {original_reason}"
            
            return result
            
        except Exception as e:
            print(f"   ❌ Gemini AI failed: {str(e)}")
            
            return self._create_safe_fallback(soa_data, error, error_type, str(e))
    
    def _parse_gemini_response(self, text: str) -> Dict[str, Any]:
        import re
        
        try:
            return json.loads(text)
        except:
            pass
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(text)
        except:
            pass
        
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
        
        raise ValueError(f"Could not parse JSON from response. Text: {text[:200]}")
    
    def _create_safe_fallback(
        self, 
        soa_data: Dict[str, Any], 
        python_error: str, 
        error_type: str,
        ai_error: str
    ) -> Dict[str, Any]:
        soa = soa_data.get("statementOfAccount", {})
        
        has_transactions = bool(soa.get("transactionReports"))
        has_summary = bool(soa.get("soaSummaryReports"))
        is_active = soa.get("status", "").upper() == "ACTIVE"
        
        report_charges = 0
        total_charge_amount = 0
        for tr in soa.get("transactionReports", []):
            event = tr.get("event", "")
            if any(rtype in event for rtype in ["FFR", "FPR", "CPR", "BPR", "BFR"]):
                report_charges += 1
                total_charge_amount += tr.get("debitAmount", 0)
        
        total_penalties = 0
        for comp in soa.get("soaSummaryReports", []):
            total_penalties += float(comp.get("overDue", 0))
        
        if not is_active:
            decision = "NO_REFUND"
            reason = "Loan is not active"
            confidence = 90
        elif report_charges == 0:
            decision = "NO_REFUND"
            reason = "No report charges found"
            confidence = 90
        elif "date" in python_error.lower() and report_charges > 0:
            decision = "NO_REFUND"
            reason = f"Data quality issue (invalid date) prevents automated refund. Found {report_charges} charge(s) worth Rs. {total_charge_amount}"
            confidence = 40
        else:
            decision = "NO_REFUND"
            reason = f"Data processing error: {error_type}"
            confidence = 20
        
        return {
            "decision": decision,
            "refund_amount": 0,
            "refund_eligible": False,
            "reason": reason,
            "confidence": confidence,
            "handled_by": "GEMINI_AI_FALLBACK",
            "requires_manual_review": confidence < CONFIDENCE_THRESHOLD,
            "original_error": python_error,
            "ai_error": ai_error,
            "ai_reasoning": f"AI parsing failed. Fallback decision based on: Active={is_active}, Charges={report_charges}, Penalties={total_penalties}",
            "data_issues": [
                f"AI processing failed: {ai_error}",
                f"Python error: {python_error}",
                f"Data analysis: {report_charges} charges found, Rs. {total_charge_amount} total"
            ]
        }




def process_refund_file(filename: str = "refund.json") -> Dict[str, Any]:
    
    print("\n" + "🎯"*40)
    print("PROFESSIONAL REFUND AUTOMATION SYSTEM")
    print("🎯"*40)
    print(f"\n📂 Loading file: {filename}")
    
    try:
        with open(filename, 'r') as f:
            soa_data = json.load(f)
        print(f"✅ File loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found")
        print(f"   Please create {filename} in the same directory")
        return {"error": "File not found"}
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {str(e)}")
        return {"error": "Invalid JSON"}
    
    soa = soa_data.get("statementOfAccount", {})
    fin_ref = soa.get("finReference", "N/A")
    customer = soa.get("custShrtName", "N/A")
    status = soa.get("status", "N/A")
    
    print(f"\n📋 Loan Information:")
    print(f"   Reference: {fin_ref}")
    print(f"   Customer: {customer}")
    print(f"   Status: {status}")
    
    print(f"\n" + "="*80)
    print("TIER 1: Python Rule Engine (Deterministic)")
    print("="*80)
    print("🐍 Processing with Python engine...")
    
    try:
        result = process_refund_case(soa_data)
        
        decision = result["decision"]
        decision["handled_by"] = "PYTHON_ENGINE"
        decision["confidence"] = 100
        decision["requires_manual_review"] = False
        
        print("✅ Python engine SUCCESS - No issues found!")
        print(f"\n📊 Decision Details:")
        print(f"   Decision: {decision['decision']}")
        print(f"   Refund Amount: Rs. {decision.get('refund_amount', 0):,.2f}")
        print(f"   Reason: {decision['reason']}")
        print(f"   Confidence: 100%")
        print(f"   Handled By: Python Engine")
        
        print("\n" + "="*80)
        print("✅ PROCESSING COMPLETE - Python Engine")
        print("="*80)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"⚠️  Python engine FAILED - Issues detected!")
        print(f"   Error Type: {error_type}")
        print(f"   Error Details: {error_msg}")
        
        print(f"\n" + "="*80)
        print("TIER 2: Gemini AI Fallback Engine")
        print("="*80)
        print("🤖 Activating Gemini AI to handle the issue...")
        
        ai = GeminiAI()
        ai_result = ai.analyze(soa_data, error_msg, error_type)
        
        print(f"✅ AI Analysis complete")
        print(f"\n📊 Decision Details:")
        print(f"   Decision: {ai_result['decision']}")
        print(f"   Refund Amount: Rs. {ai_result.get('refund_amount', 0):,.2f}")
        print(f"   Confidence: {ai_result['confidence']}%")
        print(f"   Handled By: Gemini AI")
        print(f"   Issues Found: {', '.join(ai_result.get('data_issues', []))}")
        
        if ai_result.get("requires_manual_review"):
            print(f"\n⚠️  ⚠️  ⚠️  FLAGGED FOR MANUAL REVIEW ⚠️  ⚠️  ⚠️")
            print(f"   Reason: {ai_result['reason']}")
        
        print("\n" + "="*80)
        print("✅ PROCESSING COMPLETE - Gemini AI Fallback")
        print("="*80)
        
        return {
            "finReference": fin_ref,
            "decision": ai_result,
            "metadata": {
                "engine": "gemini_ai_fallback_v1",
                "timestamp": datetime.utcnow().isoformat(),
                "python_error": error_msg,
                "python_error_type": error_type
            }
        }


def main():
    result = process_refund_file("refund1.json")
    
    output_file = "result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Result saved to: {output_file}")
    print("\n" + "="*80)
    print("🎉 Processing Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()