"""
CUAD Dataset Loader — Week 1: Day 1-2
Loads and pre-processes the Contract Understanding Atticus Dataset (CUAD).

CUAD contains 510 commercial contracts with 41 legal clause categories —
it is the primary training source for the LegalClauseClassifier.

Data Source: https://www.atticusprojectai.org/cuad
HuggingFace:  theatticusproject/cuad
"""
import json
import os
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────
# CUAD question → internal clause type mapping
# ─────────────────────────────────────────────────────────────────
CUAD_CLAUSE_MAP = {
    "Document Name": "PARTIES",
    "Parties": "PARTIES",
    "Agreement Date": "DATE",
    "Effective Date": "DATE",
    "Expiration Date": "TERMINATION",
    "Auto-Renewal": "AUTO_RENEWAL",
    "Termination For Convenience": "TERMINATION",
    "Change Of Control": "CHANGE_OF_CONTROL",
    "Anti-Assignment": "ASSIGNMENT",
    "Revenue/Profit Sharing": "PAYMENT_TERMS",
    "Price Restrictions": "PAYMENT_TERMS",
    "Minimum Commitment": "PAYMENT_TERMS",
    "Volume Restriction": "PAYMENT_TERMS",
    "Ip Ownership Assignment": "IP_OWNERSHIP",
    "Joint Ip Ownership": "IP_OWNERSHIP",
    "License Grant": "IP_OWNERSHIP",
    "Non-Transferable License": "IP_OWNERSHIP",
    "Exclusivity": "EXCLUSIVITY",
    "No-Solicit Of Customers": "NON_SOLICITATION",
    "Competitive Restriction Exception": "NON_COMPETE",
    "Non-Compete": "NON_COMPETE",
    "Limitation Of Liability": "LIABILITY_CAP",
    "Liability Cap": "LIABILITY_CAP",
    "Warranty Duration": "WARRANTIES",
    "Insurance": "INSURANCE",
    "Audit Rights": "AUDIT_RIGHTS",
    "Most Favored Nation": "MOST_FAVORED_NATION",
    "Cap On Liability": "LIABILITY_CAP",
    "Liquidated Damages": "LIQUIDATED_DAMAGES",
    "Governing Law": "GOVERNING_LAW",
    "Dispute Resolution": "ARBITRATION",
    "Source Code Escrow": "ESCROW",
    "Post-Termination Services": "TERMINATION",
    "Covenant Not To Sue": "IP_OWNERSHIP",
    "Third Party Beneficiary": "THIRD_PARTY",
    "Uncapped Liability": "LIABILITY_CAP",
    "Irrevocable Or Perpetual License": "IP_OWNERSHIP",
    "Confidentiality Duration": "CONFIDENTIALITY",
    "Confidentiality General": "CONFIDENTIALITY",
}


def load_cuad_from_huggingface(max_samples: int = 5000) -> Optional[List[Dict]]:
    """
    Load CUAD from HuggingFace datasets library.

    Requirements:
        pip install datasets

    Returns:
        List of training samples, or None if unavailable.
    """
    try:
        from datasets import load_dataset
        print("📥 Downloading CUAD from HuggingFace (this may take a few minutes)...")
        dataset = load_dataset("theatticusproject/cuad")
        print(f"✅ CUAD loaded: {len(dataset['train'])} train + {len(dataset['test'])} test samples")
        return extract_training_samples(dataset["train"], max_samples=max_samples)
    except ImportError:
        print("❌ Install datasets: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Error loading CUAD: {e}")
        return None


def load_cuad_from_json(json_path: str) -> Optional[List[Dict]]:
    """
    Load CUAD from a locally downloaded JSON file.

    Download from: https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1.json

    Args:
        json_path: Path to the CUAD_v1.json file.

    Returns:
        List of training samples, or None if file not found.
    """
    if not os.path.exists(json_path):
        print(f"❌ CUAD JSON not found at: {json_path}")
        print("   Download from: https://www.atticusprojectai.org/cuad")
        return None

    print(f"📂 Loading CUAD from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    contracts = data.get("data", [])
    print(f"✅ Loaded {len(contracts)} contracts from CUAD JSON")

    samples = []
    for contract in contracts:
        title = contract.get("title", "Unknown")
        for para in contract.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                clause_type = _map_question_to_clause(question)

                if answers:
                    for ans in answers[:1]:  # Use first answer only
                        samples.append({
                            "contract_title": title,
                            "text": context[:512],
                            "clause_text": ans.get("text", "")[:256],
                            "clause_type": clause_type,
                            "label": 1  # Positive: clause found
                        })
                else:
                    samples.append({
                        "contract_title": title,
                        "text": context[:512],
                        "clause_text": "",
                        "clause_type": clause_type,
                        "label": 0  # Negative: no such clause
                    })

    print(f"✅ Extracted {len(samples)} training samples from CUAD")
    return samples


def extract_training_samples(hf_dataset, max_samples: int = 5000) -> List[Dict]:
    """Convert HuggingFace CUAD dataset items to training dicts."""
    samples = []
    for i, item in enumerate(hf_dataset):
        if len(samples) >= max_samples:
            break
        context = item.get("context", "")
        question = item.get("question", "")
        answers = item.get("answers", {}).get("text", [])
        clause_type = _map_question_to_clause(question)

        if answers:
            samples.append({
                "text": context[:512],
                "clause_text": answers[0][:256],
                "clause_type": clause_type,
                "label": 1
            })
        else:
            samples.append({
                "text": context[:512],
                "clause_text": "",
                "clause_type": clause_type,
                "label": 0
            })
    return samples


def _map_question_to_clause(question: str) -> str:
    """Map a CUAD question string to an internal clause type."""
    for key, clause_type in CUAD_CLAUSE_MAP.items():
        if key.lower() in question.lower():
            return clause_type
    return "GENERAL"


def get_sample_contracts() -> List[Dict]:
    """
    Returns built-in sample contracts for immediate testing.
    Use this when you don't have internet access or haven't downloaded CUAD yet.
    These are realistic contracts covering all major clause types.
    """
    return [
        {
            "contract_id": "sample_001",
            "title": "Software License Agreement — High Risk",
            "text": (
                "This Software License Agreement ('Agreement') is entered into as of January 15, 2024, "
                "between Acme Technology Corp. ('Licensor') and Beta Solutions LLC ('Licensee').\n\n"
                "TERM\nThis Agreement commences on the Effective Date and continues for one (1) year. "
                "This Agreement shall automatically renew for successive one-year periods unless either "
                "party provides written notice of non-renewal at least 90 days prior to the end of the "
                "then-current term.\n\n"
                "LIMITATION OF LIABILITY\nIN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, "
                "SPECIAL, OR CONSEQUENTIAL DAMAGES. LICENSOR'S TOTAL CUMULATIVE LIABILITY SHALL NOT EXCEED "
                "THE FEES PAID BY LICENSEE IN THE TWELVE MONTHS PRECEDING THE CLAIM.\n\n"
                "CONFIDENTIALITY\nEach party agrees to maintain the confidentiality of the other party's "
                "Confidential Information and not to disclose such information to third parties.\n\n"
                "NON-COMPETE\nDuring the term and for two (2) years thereafter, Licensee agrees not to "
                "develop, market, or distribute any software product that competes with Licensor's products.\n\n"
                "INTELLECTUAL PROPERTY\nAll work product, inventions, and deliverables created under this "
                "Agreement shall be considered work for hire and shall be the exclusive property of Licensor. "
                "Licensee hereby assigns all intellectual property rights in such work to Licensor.\n\n"
                "GOVERNING LAW\nThis Agreement shall be governed by the laws of the State of Delaware. "
                "Any disputes shall be resolved through binding arbitration under AAA rules in Wilmington, Delaware.\n\n"
                "PAYMENT\nLicensee shall pay the License Fee of $50,000 per year within 30 days of invoice. "
                "Late payments shall accrue interest at 1.5% per month."
            ),
            "expected_clauses": ["AUTO_RENEWAL", "LIABILITY_CAP", "CONFIDENTIALITY",
                                  "NON_COMPETE", "IP_OWNERSHIP", "GOVERNING_LAW", "ARBITRATION", "PAYMENT_TERMS"]
        },
        {
            "contract_id": "sample_002",
            "title": "Service Agreement — Medium Risk",
            "text": (
                "SERVICE AGREEMENT entered into this 1st day of March 2024, between Global Services Inc. "
                "('Service Provider') and Client Corp. ('Client').\n\n"
                "SERVICES\nService Provider agrees to provide software development services as detailed in "
                "the attached Statement of Work.\n\n"
                "PAYMENT\nClient shall pay Service Provider $150 per hour. Invoices are due within 30 days. "
                "Client may terminate for convenience upon 30 days written notice.\n\n"
                "TERMINATION\nEither party may terminate this Agreement for convenience upon 30 days "
                "written notice. Client may terminate immediately for material breach.\n\n"
                "INTELLECTUAL PROPERTY\nAll work product created under this Agreement shall be considered "
                "work for hire and shall be the exclusive property of Client.\n\n"
                "DISPUTE RESOLUTION\nAny disputes arising under this Agreement shall be resolved through "
                "binding arbitration in New York under AAA rules.\n\n"
                "GOVERNING LAW\nThis Agreement shall be governed by New York law."
            ),
            "expected_clauses": ["PAYMENT_TERMS", "TERMINATION", "IP_OWNERSHIP", "ARBITRATION", "GOVERNING_LAW"]
        },
        {
            "contract_id": "sample_003",
            "title": "NDA Agreement — Low Risk",
            "text": (
                "NON-DISCLOSURE AGREEMENT entered into on June 1, 2024, between Zaalima Dev Inc. "
                "and Research Partners LLC.\n\n"
                "CONFIDENTIALITY\nEach party agrees to hold the other's Confidential Information in strict "
                "confidence and not disclose it to any third party without prior written consent.\n\n"
                "TERM\nThis Agreement shall remain in effect for three (3) years from the Effective Date.\n\n"
                "GOVERNING LAW\nThis Agreement is governed by the laws of California.\n\n"
                "GENERAL\nThis Agreement constitutes the entire agreement between the parties with respect "
                "to the subject matter hereof. This Agreement may not be assigned without prior written consent."
            ),
            "expected_clauses": ["CONFIDENTIALITY", "GOVERNING_LAW"]
        }
    ]


def save_samples_to_processed(samples: List[Dict], output_dir: str = "data/processed"):
    """Save extracted training samples as JSON files for the training pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    for i, sample in enumerate(samples[:100]):  # Save first 100
        filename = os.path.join(output_dir, f"cuad_sample_{i:04d}.json")
        with open(filename, "w") as f:
            json.dump(sample, f, indent=2)
    print(f"✅ Saved {min(len(samples), 100)} samples to {output_dir}/")


if __name__ == "__main__":
    print("Testing CUAD Loader with built-in samples...")
    samples = get_sample_contracts()
    print(f"\n📂 {len(samples)} sample contracts available:")
    for s in samples:
        print(f"   [{s['contract_id']}] {s['title']}")
        print(f"   Expected clauses: {s['expected_clauses']}")
        print()

    print("\nTo download the real CUAD dataset:")
    print("  Option 1 (HuggingFace):  pip install datasets")
    print("                           from src.data_engineering.cuad_loader import load_cuad_from_huggingface")
    print("                           samples = load_cuad_from_huggingface()")
    print()
    print("  Option 2 (Local JSON):   Download CUAD_v1.json from https://www.atticusprojectai.org/cuad")
    print("                           from src.data_engineering.cuad_loader import load_cuad_from_json")
    print("                           samples = load_cuad_from_json('data/CUAD_v1.json')")
