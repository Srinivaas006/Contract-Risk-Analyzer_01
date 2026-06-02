import pytesseract
from pdf2image import convert_from_path
import os

def process_contract_pdf(pdf_path):
    """
    Converts a PDF contract into searchable text.
    Matches Week 1: Day 3-5 of the Zalima Project Plan.
    """
    print(f"Starting OCR for: {pdf_path}")
    
    # 1. Convert PDF pages to images
    try:
        pages = convert_from_path(pdf_path)
        full_text = ""

        # 2. Loop through pages and extract text
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            full_text += f"\n--- Page {i+1} ---\n" + text
            
        return full_text
    except Exception as e:
        return f"Error processing PDF: {e}"

if __name__ == "__main__":
    # Placeholder for testing
    sample_path = "data/raw/sample_contract.pdf"
    if os.path.exists(sample_path):
        content = process_contract_pdf(sample_path)
        print(content[:500]) # Print first 500 characters
    else:
        print("Please drop a PDF file into data/raw/ to test the OCR pipeline.")