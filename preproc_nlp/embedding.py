import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pymongo
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TextProcessor:
    def __init__(self, mongodb_uri: str = None):
        """Initialize the text processor with MongoDB connection and models."""
        # Initialize MongoDB connection
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        if not self.mongodb_uri:
            raise ValueError("MongoDB URI not provided in environment variable MONGODB_URI")
        try:
            self.client = pymongo.MongoClient(self.mongodb_uri)
            self.client.server_info()  # Test connection
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        self.db = self.client["text_embeddings"]
        self.collection = self.db["document_chunks"]
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize the embedding model
        logger.info("Loading sentence transformer model...")
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def __del__(self):
        """Close MongoDB connection when the object is destroyed."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")

    def read_text_file(self, file_path: str) -> str:
        """Read content from a text file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                logger.info(f"Successfully read file: {file_path}")
                return text
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using the text splitter."""
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise

    def create_embeddings(self, chunks: List[str], batch_size: int = 32) -> List[List[float]]:
        """Create embeddings for text chunks."""
        try:
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(
                chunks,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info("Embeddings generated successfully")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def store_in_mongodb(self, file_path: str, chunks: List[str], embeddings: List[List[float]]):
        """Store chunks and their embeddings in MongoDB."""
        try:
            logger.info("Preparing to store in MongoDB...")
            documents = []
            for chunk, embedding in zip(chunks, embeddings):
                doc = {
                    "source_file": file_path,
                    "chunk_text": chunk,
                    "embedding": embedding
                }
                documents.append(doc)
            
            if not documents:
                logger.warning("No documents to store")
                return
            self.collection.insert_many(documents)
            logger.info(f"Successfully stored {len(documents)} documents in MongoDB")
        except Exception as e:
            logger.error(f"Error storing in MongoDB: {str(e)}")
            raise

    def process_text_file(self, file_path: str):
        """Process a single text file: read, chunk, embed, and store."""
        logger.info(f"Processing file: {file_path}")
        text = self.read_text_file(file_path)
        chunks = self.chunk_text(text)
        embeddings = self.create_embeddings(chunks)
        self.store_in_mongodb(file_path, chunks, embeddings)
        logger.info(f"Successfully processed file: {file_path}")
        return len(chunks)

    def process_directory(self, directory_path: str, max_workers: int = 4):
        """Process all text files in a directory."""
        processed_files = 0
        file_paths = [str(fp) for fp in Path(directory_path).glob('**/*.txt')]
        logger.info(f"Found {len(file_paths)} text files to process")

        def process_single_file(fp):
            try:
                chunks_created = self.process_text_file(fp)
                return 1, chunks_created
            except Exception as e:
                logger.error(f"Error processing {fp}: {str(e)}")
                return 0, 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_file, file_paths))
        
        for success, chunks in results:
            if success:
                processed_files += 1
            logger.info(f"Processed file with {chunks} chunks created")

        return processed_files

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process text files and store embeddings in MongoDB')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Path to a single text file to process')
    group.add_argument('--dir', type=str, help='Path to directory containing text files to process')
    args = parser.parse_args()
    
    try:
        processor = TextProcessor()
        if args.file:
            processor.process_text_file(args.file)
        else:
            processed_files = processor.process_directory(args.dir)
            logger.info(f"Successfully processed {processed_files} files")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()