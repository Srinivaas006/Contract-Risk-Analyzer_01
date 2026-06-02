import spacy
import json
import os

def extract_contract_entities(text: str):
    """
    Uses a baseline spaCy NER pipeline to extract key legal variables.
    Fulfills Week 1: Day 6-7 of the project timeline.
    """
    # Load the lightweight English language processing brain
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    extracted_data = {
        "ORGANIZATIONS": [],
        "DATES": [],
        "MONETARY_VALUES": []
    }
    
    # Iterate through entities identified by spaCy's default statistical model
    for ent in doc.ents:
        if ent.label_ == "ORG":
            extracted_data["ORGANIZATIONS"].append(ent.text)
        elif ent.label_ in ["DATE", "TIME"]:
            extracted_data["DATES"].append(ent.text)
        elif ent.label_ == "MONEY":
            extracted_data["MONETARY_VALUES"].append(ent.text)
            
    # Clean duplicates out of our arrays
    for key in extracted_data:
        extracted_data[key] = list(set(extracted_data[key]))
        
    return extracted_data

if __name__ == "__main__":
    # Test our baseline extractor with a complex mockup legal statement
    sample_legal_text = """
    This Master Services Agreement is entered into on January 15, 2024, by and between 
    Acme Global Logistics Inc. and TechStart Solutions LLC. The total liability caps 
    for damages under Section 4 shall not exceed $500,000 USD.
    """
    
    print("Running baseline NER pipeline analysis...")
    results = extract_contract_entities(sample_legal_text)
    print("\n--- Baseline Extraction Results ---")
    print(json.dumps(results, indent=4))