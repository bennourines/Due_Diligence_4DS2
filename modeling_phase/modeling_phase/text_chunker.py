"""
text_chunker.py

Handles loading and preprocessing whitepapers, then splitting them into
domain-specific text chunks.
"""

import os
import glob
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import utility  # Assumes utility.py is in the same directory

def preprocess_whitepaper(text):
    """Clean and preprocess whitepaper text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers (common in PDFs)
    text = re.sub(r'\b\d+\s*\|\s*[pP]age\b', '', text)
    # Replace special characters
    text = text.replace('•', '* ')
    # Clean up references and citations
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def load_whitepapers(directory="data"):
    """Load and preprocess all text files from a directory."""
    all_text = ""
    # Get all .txt files
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    if not txt_files:
        print(f"No text files found in {directory}")
        return all_text
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                print(f"Loading: {file_path}")
                file_text = file.read()
                cleaned_text = preprocess_whitepaper(file_text)
                all_text += cleaned_text + "\n\n"
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_text

def create_domain_specific_chunks(text):
    """Split text into chunks with domain-specific handling."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Larger chunks for better context
        chunk_overlap=200,    # Overlap to maintain context
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # Filter chunks using the utility function to check relevance
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        if utility.is_relevant_chunk(chunk):
            # Include section numbering for traceability
            relevant_chunks.append(f"[Section {i+1}] {chunk}")
    print(f"Created {len(relevant_chunks)} relevant chunks from {len(chunks)} total chunks")
    return relevant_chunks
