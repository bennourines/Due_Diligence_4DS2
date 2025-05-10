import os
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

def extract_and_save_texts(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(input_folder, filename)
        try:
            reader = PdfReader(filepath)
        except PdfReadError as e:
            print(f"  ▶ Could not read {filename}: {e}")
            continue

        # If encrypted, try decrypting with empty password, else skip
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # empty string password
            except Exception:
                print(f"  ▶ Skipping encrypted file {filename}")
                continue

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✔ Extracted {filename} → {txt_filename}")

    print(f"\nAll done! Text files are in: {output_folder}")

if __name__ == "__main__":
    pdf_folder = "pdfs"
    output_folder = "Extracted_Texts"
    extract_and_save_texts(pdf_folder, output_folder)
