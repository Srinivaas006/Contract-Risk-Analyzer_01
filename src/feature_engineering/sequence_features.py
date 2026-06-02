"""
Contract Clause Sequence & Structural Analysis — Week 2
Analyses the document structure of a contract: section headers, clause ordering,
completeness check, and readability complexity metrics.

A well-structured contract has all critical sections in logical order.
Missing critical sections is itself a risk flag.
"""
import re
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────
# Standard Legal Section Headers (CUAD-aligned)
# ─────────────────────────────────────────────────────────────────
SECTION_HEADERS = [
    "RECITALS", "BACKGROUND", "PREAMBLE",
    "DEFINITIONS", "INTERPRETATION",
    "TERM", "DURATION", "EFFECTIVE DATE",
    "SERVICES", "SCOPE OF WORK", "DELIVERABLES",
    "PAYMENT", "FEES", "COMPENSATION", "PRICING",
    "CONFIDENTIALITY", "NON-DISCLOSURE",
    "INTELLECTUAL PROPERTY", "IP OWNERSHIP", "PROPRIETARY RIGHTS",
    "TERMINATION", "CANCELLATION",
    "WARRANTIES", "REPRESENTATIONS",
    "LIMITATION OF LIABILITY", "INDEMNIFICATION", "INDEMNITY",
    "FORCE MAJEURE",
    "GOVERNING LAW", "APPLICABLE LAW", "JURISDICTION",
    "ARBITRATION", "DISPUTE RESOLUTION",
    "GENERAL", "MISCELLANEOUS", "GENERAL PROVISIONS",
    "ENTIRE AGREEMENT", "INTEGRATION",
    "AMENDMENTS", "MODIFICATIONS",
    "NOTICES", "NOTICE",
    "ASSIGNMENT",
    "NON-COMPETE", "NON-SOLICITATION",
    "AUDIT RIGHTS",
    "SEVERABILITY",
    "WAIVER",
]

# Clauses that MUST be present in any enforceable contract
CRITICAL_CLAUSES = [
    "TERM",           # When does it start/end?
    "PAYMENT",        # How much is owed?
    "TERMINATION",    # How can it end?
    "CONFIDENTIALITY",# What's secret?
    "GOVERNING LAW",  # Which jurisdiction's law applies?
]

# Legal jargon that increases reading difficulty
LEGAL_JARGON = [
    "whereas", "heretofore", "hereinafter", "notwithstanding", "pursuant",
    "indemnify", "indemnification", "subrogation", "estoppel", "tortious",
    "liquidated", "arbitration", "jurisdiction", "severability", "ipso facto",
    "inter alia", "mutatis mutandis", "pro rata", "bona fide", "force majeure",
    "ultra vires", "in perpetuity", "assign", "sublicense",
]


def extract_contract_sections(text: str) -> List[Dict]:
    """
    Parse a contract's structure by identifying section headers.

    Heuristic: a line is a section header if it's short (<80 chars),
    in ALL CAPS or Title Case, and matches a known legal section name.

    Returns:
        List of sections: [{title, content, line_number, word_count}]
    """
    sections = []
    lines = text.split('\n')
    current_section = None
    current_content = []
    current_line_no = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        stripped_upper = stripped.upper()

        if not stripped:
            if current_section:
                current_content.append("")
            continue

        # Detect header: short line matching a known section keyword
        is_header = (
            len(stripped) < 80
            and any(header in stripped_upper for header in SECTION_HEADERS)
        )

        if is_header:
            # Save the previous section
            if current_section is not None:
                content_text = ' '.join(current_content).strip()
                content_text = re.sub(r'\s+', ' ', content_text)
                sections.append({
                    "title": current_section,
                    "content": content_text,
                    "line_number": current_line_no,
                    "word_count": len(content_text.split()) if content_text else 0
                })
            current_section = stripped
            current_content = []
            current_line_no = i
        elif current_section is not None:
            current_content.append(stripped)

    # Flush the last section
    if current_section:
        content_text = ' '.join(current_content).strip()
        content_text = re.sub(r'\s+', ' ', content_text)
        sections.append({
            "title": current_section,
            "content": content_text,
            "line_number": current_line_no,
            "word_count": len(content_text.split()) if content_text else 0
        })

    return sections


def analyze_clause_sequence(sections: List[Dict]) -> Dict:
    """
    Check whether the contract contains all critical sections
    and analyze the ordering logic.

    Returns:
        Dict with completeness score, missing clauses, and ordering notes.
    """
    section_titles_upper = [s["title"].upper() for s in sections]

    # Check for critical clause presence
    present = []
    missing = []

    for clause in CRITICAL_CLAUSES:
        found = any(clause in title for title in section_titles_upper)
        if found:
            present.append(clause)
        else:
            missing.append(clause)

    completeness_pct = round(len(present) / len(CRITICAL_CLAUSES) * 100, 1)

    # Grade completeness
    if completeness_pct >= 100:
        completeness_grade = "COMPLETE"
    elif completeness_pct >= 60:
        completeness_grade = "MOSTLY COMPLETE"
    else:
        completeness_grade = "INCOMPLETE — HIGH RISK"

    avg_section_words = round(
        sum(s["word_count"] for s in sections) / max(1, len(sections)), 1
    )

    # Flag if critical sections are missing
    missing_risk = []
    for m in missing:
        missing_risk.append(f"⚠️  Missing '{m}' clause — this is a critical contract element!")

    return {
        "total_sections_found": len(sections),
        "sections_identified": [s["title"] for s in sections],
        "critical_clauses_present": present,
        "critical_clauses_missing": missing,
        "completeness_score": completeness_pct,
        "completeness_grade": completeness_grade,
        "average_section_length_words": avg_section_words,
        "missing_clause_warnings": missing_risk,
    }


def compute_contract_complexity(text: str) -> Dict:
    """
    Compute reading complexity metrics for a contract.

    High complexity = dense legal jargon + very long sentences.
    This correlates with how difficult the contract is to understand and audit.

    Returns:
        Dict with word count, readability metrics, jargon density, and complexity grade.
    """
    # Basic stats
    words = text.split()
    word_count = len(words)

    # Sentence splitting
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    sentence_count = max(1, len(sentences))

    avg_sentence_length = round(word_count / sentence_count, 1)

    # Legal jargon density
    text_lower = text.lower()
    jargon_hits = [term for term in LEGAL_JARGON if term in text_lower]
    jargon_count = len(jargon_hits)
    jargon_density = round(jargon_count / max(1, word_count / 100), 2)  # per 100 words

    # Flesch-Kincaid approximate grade level
    # FK Grade = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    # Approximate syllables as word_length / 3
    avg_syllables = sum(max(1, len(w) // 3) for w in words) / max(1, word_count)
    fk_grade = round(0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59, 1)

    # Complexity classification
    if avg_sentence_length > 40 or jargon_density > 5:
        complexity_level = "VERY HIGH"
        complexity_note = "🔴 Extremely dense legal language — specialist review strongly advised."
    elif avg_sentence_length > 28 or jargon_density > 3:
        complexity_level = "HIGH"
        complexity_note = "🟠 Complex language — legal professional review recommended."
    elif avg_sentence_length > 18:
        complexity_level = "MEDIUM"
        complexity_note = "🟡 Moderately complex — careful reading required."
    else:
        complexity_level = "LOW"
        complexity_note = "🟢 Relatively readable for a legal document."

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length_words": avg_sentence_length,
        "flesch_kincaid_grade_approx": fk_grade,
        "legal_jargon_count": jargon_count,
        "legal_jargon_terms_found": jargon_hits[:10],  # Top 10
        "jargon_density_per_100_words": jargon_density,
        "complexity_level": complexity_level,
        "complexity_note": complexity_note,
        "readability_concern": avg_sentence_length > 35,
    }


if __name__ == "__main__":
    sample_text = """
TERM
This Agreement commences on the Effective Date and continues for one year.

PAYMENT
All invoices are due within 30 days. Late payments accrue interest.

CONFIDENTIALITY
Each party agrees to maintain the confidentiality of Confidential Information.

NON-COMPETE
During the term and for two years thereafter, Licensee shall not compete.

GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

TERMINATION
Either party may terminate this Agreement upon 90 days written notice.
    """

    sections = extract_contract_sections(sample_text)
    print(f"Sections found: {len(sections)}")

    analysis = analyze_clause_sequence(sections)
    print(f"Completeness: {analysis['completeness_score']}% — {analysis['completeness_grade']}")
    print(f"Missing: {analysis['critical_clauses_missing']}")

    complexity = compute_contract_complexity(sample_text)
    print(f"Complexity: {complexity['complexity_level']} (avg sentence: {complexity['avg_sentence_length_words']} words)")
