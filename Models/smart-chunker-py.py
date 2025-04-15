import os
import glob
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Union, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from entity_extractor import EntityExtractor
from utility import FinanceUtility

class SmartChunker:
    """Advanced document processing with domain-specific chunking and search capabilities"""

    def __init__(self, chunk_size=1000, chunk_overlap=200, data_dir="faiss_index"):
        """Initialize chunker with optional configs"""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.data_dir = data_dir
        self.entity_extractor = EntityExtractor()
        self.utility = FinanceUtility()
        self.vectordb = None
        self.embeddings = None

    def preprocess_text(self, text: str) -> str:
        """Cleans up raw text before chunking."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d+\s*\|\s*[pP]age\b', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        return text.replace('•', '* ').strip()

    def chunk_text(self, text: str, metadata: Dict[str, str] = None) -> List[Dict[str, Union[str, Dict]]]:
        """Splits and filters text into meaningful domain-specific chunks with entity extraction."""
        chunks = self.splitter.split_text(text)
        relevant_chunks = []

        for i, chunk in enumerate(chunks):
            if self.utility.is_relevant_chunk(chunk):
                # Extract entities and risk features
                extracted_entities = self.entity_extractor.extract_named_entities(chunk)
                risk_features, risk_score = self.entity_extractor.extract_risk_features(chunk)

                # Combine metadata
                chunk_metadata = {
                    "chunk_id": hashlib.md5(chunk.encode()).hexdigest(),
                    "entities": extracted_entities,
                    "risk_features": risk_features,
                    "risk_score": risk_score,
                    **(metadata or {})
                }

                relevant_chunks.append({
                    "text": f"[Section {i+1}] {chunk}",
                    "metadata": chunk_metadata
                })

        print(f"✅ Kept {len(relevant_chunks)} relevant chunks from {len(chunks)} total.")
        return relevant_chunks

    def save_chunks_to_json(self, chunks, output_path, drive_output_folder=None):
        """Saves chunks to local directory and optionally to a drive directory."""
        # Ensure the local directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Save to local path
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"📦 Saved chunks to {output_path}")

        # If drive output folder is specified, save there too
        if drive_output_folder:
            drive_folder = Path(drive_output_folder)
            drive_folder.mkdir(parents=True, exist_ok=True)  # Make sure the folder exists

            # Save to drive path
            drive_path = os.path.join(drive_output_folder, os.path.basename(output_path))
            with open(drive_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print(f"📦 Saved chunks to {drive_path}")

    def load_chunks_from_json(self, input_path):
        """Loads previously saved chunks."""
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_cleaned_texts(self, folder="data/cleaned_texts") -> Dict[str, str]:
        """Loads all .txt files from a folder."""
        texts = {}
        for file_path in glob.glob(os.path.join(folder, "*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    texts[os.path.basename(file_path)] = f.read()
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                with open(file_path, "r", encoding="latin-1") as f:
                    texts[os.path.basename(file_path)] = f.read()
        return texts

    def process_all_texts(self, input_folder="data/cleaned_texts", output_folder="chunks", drive_output_folder=None):
        """Process all texts in a folder and save chunks"""
        print(f"Looking for text files in: {input_folder}")
        texts = self.load_cleaned_texts(input_folder)

        if not texts:
            print(f"⚠️ No text files found in {input_folder}. Please check the path and file extensions.")
            return 0

        print(f"Found {len(texts)} text files to process")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        for filename, content in texts.items():
            print(f"\n📄 Processing {filename} (content length: {len(content)})")
            cleaned = self.preprocess_text(content)
            chunks = self.chunk_text(cleaned, metadata={"source": filename})
            self.save_chunks_to_json(
                chunks, 
                f"{output_folder}/{filename.replace('.txt', '_chunks.json')}", 
                drive_output_folder
            )

        print(f"\n✅ Completed processing {len(texts)} files")
        return len(texts)
