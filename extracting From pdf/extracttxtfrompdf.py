import os
import fitz  # pymupdf for text extraction
import pdfplumber  # Extract tables from PDF

def extract(pdf_path, output_txt_path):
    """Extract text and tables from a PDF and save to a text file."""
    # Check if file is empty
    if os.stat(pdf_path).st_size == 0:
        print(f"⚠️ Skipping empty file: {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    pdf_tables = pdfplumber.open(pdf_path)
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for page_num in range(len(doc)):
            f.write(f"--- Page {page_num + 1} ---\n\n")
            # Extract text with pymupdf
            text = doc[page_num].get_text("text")
            f.write(text + "\n")
            
            # Extract tables with pdfplumber
            page = pdf_tables.pages[page_num]
            tables = page.extract_table()
            
            if tables:
                f.write("\n--- Tables Found ---\n")
                for row in tables:
                    cleaned_row = [cell if cell is not None else "" for cell in row]
                    f.write("\t".join(cleaned_row) + "\n")
            
            f.write("\n" + "-" * 80 + "\n")
    
    doc.close()
    pdf_tables.close()
    print(f"✅ Extraction complete. Data saved to: {output_txt_path}")

# Define directories
source_directory = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\preproc_nlp\raw_data_papers"
destination_directory = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\preproc_nlp\cleaned_data"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Process each PDF file in the source directory
for filename in os.listdir(source_directory):
   
    if filename.endswith(".pdf"):
        source_file_path = os.path.join(source_directory, filename)
        destination_file_path = os.path.join(destination_directory, f"{os.path.splitext(filename)[0]}_processed.txt")
        extract(source_file_path, destination_file_path)