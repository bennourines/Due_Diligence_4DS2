# filepath: c:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\DeployTrial\index_documents.py
import os
import glob
import uuid
from document_processor import DocumentProcessor
from search_engine import HybridSearchEngine

# --- Configuration ---
# !!! IMPORTANT: Set this to the directory containing your .txt, .pdf, etc. files !!!
DATA_DIR = "../DATA_COMPLETE/cleaned_texts"  # Example: Adjust this path as needed

# Assign a fixed Project ID for these pre-indexed documents
# You will need to use this ID later when querying via the API/UI
# Or generate one if you prefer: str(uuid.uuid4())
PROJECT_ID = "pre_indexed_project_01"

# --- Initialization ---
doc_processor = DocumentProcessor()
search_engine = HybridSearchEngine()

# --- Processing ---
def run_indexing():
    print(f"Starting indexing for project: {PROJECT_ID}")
    print(f"Looking for documents in: {os.path.abspath(DATA_DIR)}")

    all_docs = []
    # Use glob to find all supported files (add more extensions if needed)
    file_patterns = ["*.txt", "*.pdf", "*.docx", "*.xlsx", "*.md"]
    found_files = []
    for pattern in file_patterns:
        found_files.extend(glob.glob(os.path.join(DATA_DIR, pattern)))

    if not found_files:
        print(f"Error: No documents found in {DATA_DIR}. Please check the DATA_DIR path.")
        return

    print(f"Found {len(found_files)} files to process.")

    for file_path in found_files:
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            # Load and split the document
            docs = doc_processor.load_and_split(file_path)
            if docs:
                # Add project_id to metadata if needed (optional, search_engine might not use it directly here)
                # for doc in docs:
                #     doc.metadata["project_id"] = PROJECT_ID
                all_docs.extend(docs)
            else:
                print(f"Warning: No content extracted from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if all_docs:
        print(f"\nAdding {len(all_docs)} document chunks to the search engine...")
        try:
            # Add documents to the search engine and persist the index
            search_engine.add_documents(all_docs, PROJECT_ID)
            print(f"Successfully indexed documents for project_id: {PROJECT_ID}")
            print(f"Vector store saved in: {os.path.abspath('vector_stores/')}")
        except Exception as e:
            print(f"Error adding documents to search engine: {e}")
    else:
        print("No document chunks were generated. Index not updated.")

if __name__ == "__main__":
    run_indexing()