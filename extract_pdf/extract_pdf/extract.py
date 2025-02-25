# Install required libraries

import fitz  # pymupdf for text extraction
import pdfplumber  # Extract tables from PDF
import os

# Define the PDF file path (Update this if using Google Drive)
pdf_paths = [
    "G:\\Ela\\study\\4-Ds\\S2\\PIDS\\DATA\\us-crypto-regulatory-whitepaper.pdf",
    "G:\\Ela\\study\\4-Ds\\S2\\PIDS\\DATA\\WEF_Digital_Assets_Regulation_2024.pdf"
]


for pdf_path in pdf_paths:
    # Extract filename without extension
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_txt_path = f"{file_name}.txt"


    # Open PDF with pymupdf (for text extraction)
    doc = fitz.open(pdf_path)

    # Open PDF with pdfplumber (for table extraction)
    pdf_tables = pdfplumber.open(pdf_path)

    # Open text file to save results
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
                    # Replace None values with an empty string before joining
                    cleaned_row = [cell if cell is not None else "" for cell in row]
                    f.write("\t".join(cleaned_row) + "\n")  # Save table rows with tab spacing

            f.write("\n" + "-" * 80 + "\n")  # Page separator

    # Close PDFs
doc.close()
pdf_tables.close()

print(f"✅ Extraction complete. Data saved to: {output_txt_path}")
