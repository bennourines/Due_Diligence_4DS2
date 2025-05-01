import os
import PyPDF2
import pandas as pd
from pathlib import Path

def create_output_directory():
    """Create the text_extractions directory if it doesn't exist"""
    output_dir = Path("text_extractions")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
    return text

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error processing {txt_path}: {str(e)}")
        return ""

def extract_text_from_csv(csv_path):
    """Extract text from a CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Convert DataFrame to string representation
        text = df.to_string(index=False)
        return text
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return ""

def extract_text_from_excel(excel_path):
    """Extract text from an Excel file"""
    try:
        df = pd.read_excel(excel_path)
        # Convert DataFrame to string representation
        text = df.to_string(index=False)
        return text
    except Exception as e:
        print(f"Error processing {excel_path}: {str(e)}")
        return ""

def process_files():
    """Process all files in the data directory"""
    output_dir = create_output_directory()
    data_dir = Path("data")
    
    for file_path in data_dir.glob("*"):
        if file_path.is_file():
            # Create output filename
            output_filename = file_path.stem + "_extracted.txt"
            output_path = output_dir / output_filename
            
            # Extract text based on file extension
            if file_path.suffix.lower() == '.pdf':
                text = extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == '.txt':
                text = extract_text_from_txt(file_path)
            elif file_path.suffix.lower() == '.csv':
                text = extract_text_from_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                text = extract_text_from_excel(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            
            # Save extracted text
            if text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Processed: {file_path.name} -> {output_filename}")

if __name__ == "__main__":
    process_files() 