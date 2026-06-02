import json
import os
import re

def clean_contract_text(text: str) -> str:
    """Removes messy whitespace and formatting anomalies from OCR legal text."""
    # Replace multiple spaces/newlines with single variations
    text = re.sub(re.compile(r'\s+'), ' ', text)
    return text.strip()

def prepare_nlp_training_sample(contract_id: str, text: str, clause_type: str, clause_text: str):
    """
    Formats parsed data into a standardized JSON payload 
    ready for Week 2 Transformer fine-tuning.
    """
    cleaned_text = clean_contract_text(text)
    cleaned_clause = clean_contract_text(clause_text)
    
    # Find exact character indices for Named Entity Recognition (NER) labeling
    start_idx = cleaned_text.find(cleaned_clause)
    end_idx = start_idx + len(cleaned_clause) if start_idx != -1 else -1
    
    payload = {
        "contract_id": contract_id,
        "processed_text": cleaned_text,
        "annotations": {
            "label": clause_type,
            "text_segment": cleaned_clause,
            "char_start": start_idx,
            "char_end": end_idx
        }
    }
    
    # Save target file
    os.makedirs("data/processed", exist_ok=True)
    target_file = f"data/processed/{contract_id}.json"
    with open(target_file, "w") as f:
        json.dump(payload, f, indent=4)
        
    print(f"Saved processed sample to: {target_file}")
    return payload

if __name__ == "__main__":
    # Test our baseline data converter with a simulated legal clause
    sample_raw_text = "This AGREEMENT is entered into this 1st day of June 2026 by and between Zaalima Dev..."
    sample_clause = "between Zaalima Dev"
    
    prepare_nlp_training_sample(
        contract_id="contract_001", 
        text=sample_raw_text, 
        clause_type="PARTIES", 
        clause_text=sample_clause
    )