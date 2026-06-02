"""
Contract Recency & Date Features — Week 1: Day 6-7
Extracts and analyzes time-based risk signals from legal contract text.
Identifies deadlines, notice periods, and calculates how time-sensitive the contract is.
"""
import re
from datetime import datetime
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────
# Date Extraction Patterns
# ─────────────────────────────────────────────────────────────────
MONTH_NAMES = (
    r"(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)

DATE_PATTERNS = [
    # "January 15, 2024" or "Jan 15, 2024"
    rf"\b{MONTH_NAMES}\s+\d{{1,2}},?\s+\d{{4}}\b",
    # "15th day of January, 2024" or "1st day of March 2024"
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+day\s+of\s+{MONTH_NAMES},?\s+\d{{4}}\b",
    # "2024-01-15"
    r"\b\d{4}-\d{2}-\d{2}\b",
    # "01/15/2024" or "15/01/2024"
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    # "15.01.2024"
    r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
]


def extract_dates_from_text(text: str) -> List[str]:
    """
    Extract all date strings found in a contract using regex patterns.

    Returns:
        Deduplicated list of date strings found.
    """
    dates = []
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique_dates = []
    for d in dates:
        if d.strip() not in seen:
            seen.add(d.strip())
            unique_dates.append(d.strip())

    return unique_dates


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try to parse a date string into a datetime object."""
    formats = [
        "%B %d, %Y",     # January 15, 2024
        "%B %d %Y",      # January 15 2024
        "%b %d, %Y",     # Jan 15, 2024
        "%Y-%m-%d",      # 2024-01-15
        "%m/%d/%Y",      # 01/15/2024
        "%d/%m/%Y",      # 15/01/2024
        "%d.%m.%Y",      # 15.01.2024
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def compute_contract_age_features(effective_date_str: Optional[str] = None) -> Dict:
    """
    Compute age-based risk signals for a contract.

    Logic: Contracts that are more than 2 years old may have outdated terms,
    legal references, or pricing — increasing compliance risk.

    Returns:
        Dict with age metrics and risk flags.
    """
    now = datetime.now()

    base_features = {
        "current_date": now.strftime("%Y-%m-%d"),
        "effective_date": None,
        "age_days": None,
        "age_years": None,
        "is_recent": True,
        "age_risk_flag": False,
        "age_risk_level": "LOW",
        "note": "No effective date detected"
    }

    if not effective_date_str:
        return base_features

    parsed = _parse_date(effective_date_str)
    if not parsed:
        base_features["note"] = f"Could not parse date: {effective_date_str}"
        return base_features

    age_days = (now - parsed).days
    age_years = round(age_days / 365.25, 2)
    is_recent = age_years < 2

    age_risk_level = "LOW"
    if age_years > 5:
        age_risk_level = "HIGH"
    elif age_years > 2:
        age_risk_level = "MEDIUM"

    base_features.update({
        "effective_date": effective_date_str,
        "age_days": age_days,
        "age_years": age_years,
        "is_recent": is_recent,
        "age_risk_flag": age_years > 2,
        "age_risk_level": age_risk_level,
        "note": "Active" if is_recent else f"⚠️ Contract is {age_years:.1f} years old — review for outdated terms"
    })

    return base_features


def compute_deadline_urgency(text: str) -> Dict:
    """
    Scan contract for notice periods, renewal windows, and payment deadlines.
    Flags time-sensitive clauses where the window is ≤ 30 days.

    Returns:
        Dict with urgency analysis for each deadline type.
    """
    # Regex patterns: each captures the number of days
    urgency_patterns = {
        "termination_notice": (
            r"\b(\d+)\s+days?[\s\w]*(?:written\s+)?notice[\s\w]*(?:terminat|cancel)",
            "Notice required to terminate contract"
        ),
        "renewal_opt_out_window": (
            r"\b(\d+)\s+days?\s+(?:written\s+)?(?:notice\s+)?prior\s+to\s+(?:the\s+)?(?:end|expir|renew)",
            "Opt-out window before auto-renewal"
        ),
        "payment_deadline": (
            r"\b(?:due|payable|pay)\s+within\s+(\d+)\s+days?\b",
            "Payment due within N days"
        ),
        "cure_period": (
            r"\b(?:cure|remedy|rectify)\s+(?:within|in)\s+(\d+)\s+days?\b",
            "Breach cure / remedy period"
        ),
        "response_deadline": (
            r"\b(?:respond|response|reply)\s+(?:within|in)\s+(\d+)\s+(?:business\s+)?days?\b",
            "Required response window"
        ),
    }

    findings = {}
    text_lower = text.lower()

    for feature_name, (pattern, description) in urgency_patterns.items():
        matches = re.findall(pattern, text_lower)

        if matches:
            day_values = [int(m) for m in matches]
            min_days = min(day_values)
            findings[feature_name] = {
                "detected": True,
                "description": description,
                "day_values": day_values,
                "min_days": min_days,
                "urgency_flag": min_days <= 30,   # Tight window
                "risk_level": "HIGH" if min_days <= 14 else "MEDIUM" if min_days <= 30 else "LOW"
            }
        else:
            findings[feature_name] = {
                "detected": False,
                "description": description,
                "risk_level": "LOW"
            }

    # Overall urgency summary
    high_urgency = [k for k, v in findings.items()
                    if v.get("detected") and v.get("risk_level") == "HIGH"]

    findings["_summary"] = {
        "high_urgency_deadlines": len(high_urgency),
        "high_urgency_items": high_urgency,
        "overall_urgency": "HIGH" if high_urgency else "LOW",
        "action_required": bool(high_urgency)
    }

    return findings


if __name__ == "__main__":
    sample_text = """
    This Agreement commences January 15, 2024 and expires December 31, 2025.
    Either party may terminate with 90 days written notice prior to the end of the term.
    Payment is due within 30 days of invoice. Late payments accrue interest at 1.5% per month.
    In the event of breach, the defaulting party shall have 15 days to cure.
    """

    print("📅 Extracted Dates:", extract_dates_from_text(sample_text))
    print("\n⏳ Contract Age:", compute_contract_age_features("January 15, 2024"))
    print("\n⚠️  Deadline Urgency:")
    urgency = compute_deadline_urgency(sample_text)
    for k, v in urgency.items():
        if v.get("detected"):
            print(f"   [{v['risk_level']}] {k}: {v['day_values']} days")
