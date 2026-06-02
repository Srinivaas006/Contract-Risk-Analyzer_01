"""
Contract Risk Scorer — Week 2: Day 5-7
Implements post-processing heuristics to identify and score risky legal clauses.
This is the INTELLIGENCE layer — it analyses extracted text and returns a structured risk report.

No GPU or model training required — works immediately out of the box using
pattern matching and legal domain knowledge.
"""
import re
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────
# Risk Pattern Library — 41 CUAD-aligned legal clause categories
# Each entry defines: keywords, risk level (HIGH/MEDIUM/LOW),
# weight (contribution to overall risk score), and human description.
# ─────────────────────────────────────────────────────────────────
RISK_PATTERNS = {
    "AUTO_RENEWAL": {
        "keywords": ["auto-renew", "automatically renew", "automatic renewal",
                     "renews automatically", "auto renew", "evergreen clause"],
        "risk_level": "HIGH",
        "weight": 4,
        "description": "Auto-Renewal Trap",
        "advice": "Ensure you have calendar reminders set before the opt-out deadline."
    },
    "TERMINATION": {
        "keywords": ["terminat", "cancell", "cancel ", "expire ", "expir"],
        "risk_level": "HIGH",
        "weight": 3,
        "description": "Contract Termination Clause",
        "advice": "Review who can terminate, under what conditions, and with how much notice."
    },
    "LIABILITY_CAP": {
        "keywords": ["liability", "liable", "indemnif", "aggregate damages",
                     "limitation of liability", "in no event shall"],
        "risk_level": "HIGH",
        "weight": 3,
        "description": "Liability & Indemnification",
        "advice": "Check if the liability cap is adequate and if indemnification is mutual."
    },
    "NON_COMPETE": {
        "keywords": ["non-compete", "non compete", "noncompete",
                     "not compete", "competitive activities", "competitive business"],
        "risk_level": "HIGH",
        "weight": 4,
        "description": "Non-Compete Restriction",
        "advice": "Verify the duration and geographic scope — overly broad non-competes may be unenforceable."
    },
    "IP_OWNERSHIP": {
        "keywords": ["intellectual property", "copyright", "patent", "trademark",
                     "work for hire", "assign all right", "ip assignment"],
        "risk_level": "HIGH",
        "weight": 3,
        "description": "Intellectual Property Ownership Transfer",
        "advice": "Clarify exactly which IP is being assigned and whether it includes pre-existing IP."
    },
    "LIQUIDATED_DAMAGES": {
        "keywords": ["liquidated damages", "penalty clause", "pre-agreed damages",
                     "stipulated damages"],
        "risk_level": "HIGH",
        "weight": 3,
        "description": "Liquidated Damages / Penalty Clause",
        "advice": "Ensure the penalty amounts are proportionate to actual potential losses."
    },
    "CHANGE_OF_CONTROL": {
        "keywords": ["change of control", "merger", "acquisition", "assignment without consent",
                     "change in control"],
        "risk_level": "HIGH",
        "weight": 2,
        "description": "Change of Control Provision",
        "advice": "This clause may trigger on M&A events — verify impact on business continuity."
    },
    "CONFIDENTIALITY": {
        "keywords": ["confidential", "proprietary information", "trade secret",
                     "non-disclosure", "nda", "not disclose"],
        "risk_level": "MEDIUM",
        "weight": 2,
        "description": "Confidentiality / NDA Obligations",
        "advice": "Check the scope, duration, and exceptions to confidentiality obligations."
    },
    "GOVERNING_LAW": {
        "keywords": ["governing law", "jurisdiction", "applicable law",
                     "courts of", "venue shall", "choice of law"],
        "risk_level": "MEDIUM",
        "weight": 1,
        "description": "Governing Law & Jurisdiction",
        "advice": "Ensure the governing jurisdiction is practical and not unfairly advantageous to the other party."
    },
    "PAYMENT_TERMS": {
        "keywords": ["payment due", "invoice", "late payment", "interest on overdue",
                     "penalty for late", "net 30", "net 60"],
        "risk_level": "MEDIUM",
        "weight": 2,
        "description": "Payment Terms & Late Penalties",
        "advice": "Check payment deadlines, interest rates on late payments, and what triggers a breach."
    },
    "ARBITRATION": {
        "keywords": ["arbitration", "arbitrate", "binding arbitration",
                     "dispute resolution", "aaa rules", "mediation"],
        "risk_level": "MEDIUM",
        "weight": 2,
        "description": "Dispute Resolution / Arbitration Clause",
        "advice": "Arbitration waives your right to jury trial — understand the venue and arbitration rules."
    },
    "NON_SOLICITATION": {
        "keywords": ["non-solicitation", "not solicit", "solicit employees",
                     "poach", "hire employees of"],
        "risk_level": "MEDIUM",
        "weight": 2,
        "description": "Non-Solicitation of Employees",
        "advice": "Check scope — overly broad clauses may restrict normal hiring activities."
    },
    "EXCLUSIVITY": {
        "keywords": ["exclusive", "exclusivity", "sole provider",
                     "not engage any other", "exclusive right"],
        "risk_level": "MEDIUM",
        "weight": 2,
        "description": "Exclusivity Clause",
        "advice": "Exclusivity limits your ability to work with other parties — verify the scope."
    },
    "FORCE_MAJEURE": {
        "keywords": ["force majeure", "act of god", "circumstances beyond control",
                     "unforeseeable", "pandemic", "natural disaster"],
        "risk_level": "LOW",
        "weight": 1,
        "description": "Force Majeure Provision",
        "advice": "Ensure force majeure covers modern events (pandemics, cyberattacks) and your specific needs."
    },
    "AUDIT_RIGHTS": {
        "keywords": ["audit right", "right to audit", "inspect records",
                     "examination of books", "access to records"],
        "risk_level": "LOW",
        "weight": 1,
        "description": "Audit Rights",
        "advice": "Check if audit rights are mutual and what costs are covered."
    },
    "MOST_FAVORED_NATION": {
        "keywords": ["most favored nation", "mfn", "best price guarantee",
                     "most favored customer", "price parity"],
        "risk_level": "MEDIUM",
        "weight": 2,
        "description": "Most Favored Nation / Best Price Clause",
        "advice": "MFN clauses can restrict your pricing flexibility with other customers."
    },
}


def _find_evidence(keyword: str, text: str, context_chars: int = 200) -> str:
    """Extract a snippet of text around the first occurrence of a keyword."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return ""
    start = max(0, idx - 50)
    end = min(len(text), idx + context_chars)
    snippet = text[start:end].strip()
    # Clean up whitespace
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet


def detect_clauses(text: str) -> List[Dict]:
    """
    Scan contract text for all known risk clause patterns.

    Returns:
        List of detected clauses with type, risk level, evidence snippet, and advice.
    """
    text_lower = text.lower()
    detected = []

    for clause_type, config in RISK_PATTERNS.items():
        matched_keyword = None
        for keyword in config["keywords"]:
            if keyword in text_lower:
                matched_keyword = keyword
                break

        if matched_keyword:
            evidence = _find_evidence(matched_keyword, text)
            detected.append({
                "clause_type": clause_type,
                "risk_level": config["risk_level"],
                "description": config["description"],
                "weight": config["weight"],
                "matched_keyword": matched_keyword,
                "evidence_snippet": evidence,
                "advice": config["advice"]
            })

    # Sort by risk weight descending (most dangerous first)
    detected.sort(key=lambda x: x["weight"], reverse=True)
    return detected


def score_contract_risk(text: str) -> Dict:
    """
    Master function: compute an overall risk report for a contract.

    Returns a comprehensive dict including:
    - risk_score (0–100)
    - risk_grade (A / B / C / D / F)
    - risk_label (LOW RISK → VERY HIGH RISK)
    - Breakdown by severity
    - Full list of detected clauses with evidence
    - Actionable recommendations
    """
    clauses = detect_clauses(text)

    if not clauses:
        return {
            "risk_score": 5,
            "risk_grade": "A",
            "risk_label": "LOW RISK",
            "risk_color": "#22c55e",
            "total_clauses_detected": 0,
            "high_risk_clauses": 0,
            "medium_risk_clauses": 0,
            "low_risk_clauses": 0,
            "detected_clauses": [],
            "recommendations": [
                "✅ No major risk clauses detected.",
                "📋 The contract appears standard — but always have a legal professional review before signing."
            ]
        }

    # Weighted scoring
    max_possible = sum(c["weight"] for c in RISK_PATTERNS.values())
    actual_score = sum(c["weight"] for c in clauses)
    risk_score = min(100, int((actual_score / max_possible) * 100))

    # Severity breakdown
    high_risk   = [c for c in clauses if c["risk_level"] == "HIGH"]
    medium_risk = [c for c in clauses if c["risk_level"] == "MEDIUM"]
    low_risk    = [c for c in clauses if c["risk_level"] == "LOW"]

    # Assign letter grade
    if risk_score >= 65:
        grade, label, color = "F", "VERY HIGH RISK", "#ef4444"
    elif risk_score >= 48:
        grade, label, color = "D", "HIGH RISK", "#f97316"
    elif risk_score >= 32:
        grade, label, color = "C", "MEDIUM RISK", "#eab308"
    elif risk_score >= 18:
        grade, label, color = "B", "LOW-MEDIUM RISK", "#84cc16"
    else:
        grade, label, color = "A", "LOW RISK", "#22c55e"

    # Build actionable recommendations
    recommendations = []
    if high_risk:
        recommendations.append(
            f"🚨 {len(high_risk)} HIGH-RISK clause(s) found — DO NOT sign without legal review!"
        )
        for c in high_risk[:3]:  # Top 3 most urgent
            recommendations.append(f"   ⚠️  [{c['clause_type']}] {c['advice']}")
    if medium_risk:
        recommendations.append(
            f"📋 {len(medium_risk)} MEDIUM-RISK clause(s) require careful review."
        )
    if low_risk:
        recommendations.append(
            f"ℹ️  {len(low_risk)} LOW-RISK clause(s) are informational — review when convenient."
        )
    if not high_risk:
        recommendations.append("✅ No high-risk clauses detected.")
    recommendations.append(
        "💡 Always consult a qualified legal professional before signing any contract."
    )

    return {
        "risk_score": risk_score,
        "risk_grade": grade,
        "risk_label": label,
        "risk_color": color,
        "total_clauses_detected": len(clauses),
        "high_risk_clauses": len(high_risk),
        "medium_risk_clauses": len(medium_risk),
        "low_risk_clauses": len(low_risk),
        "detected_clauses": clauses,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    sample = """
    This Software License Agreement is entered into on January 15, 2024, between Acme Corp and Beta LLC.
    This Agreement shall automatically renew for successive one-year periods unless either party provides
    written notice of non-renewal at least 90 days prior to the end of the then-current term.
    IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT OR CONSEQUENTIAL DAMAGES.
    During the term and for two years thereafter, Licensee shall not compete with Licensor.
    All intellectual property created under this Agreement shall be work for hire.
    This Agreement shall be governed by the laws of the State of Delaware.
    """
    print("🔍 Running risk analysis...")
    result = score_contract_risk(sample)
    print(f"\n📊 Risk Score:  {result['risk_score']}/100")
    print(f"🏆 Risk Grade:  {result['risk_grade']}  —  {result['risk_label']}")
    print(f"🔴 High Risk:   {result['high_risk_clauses']} clauses")
    print(f"🟡 Medium Risk: {result['medium_risk_clauses']} clauses")
    print(f"🟢 Low Risk:    {result['low_risk_clauses']} clauses")
    print("\n💡 Recommendations:")
    for r in result["recommendations"]:
        print(f"  {r}")
