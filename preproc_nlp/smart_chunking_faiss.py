import os
import re
import sys
import traceback
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"Warning: Could not download NLTK data: {str(e)}")

class SmartChunker:
    def __init__(
        self,
        model_name: str = 'multi-qa-mpnet-base-dot-v1',
        chunk_size: int = 250,
        overlap_size: int = 50,
        data_dir: str = 'vector_store'
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.data_dir = data_dir
        self.model = None
        self.index = None
        self.metadata = []
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define paths
        self.index_path = os.path.join(self.data_dir, 'faiss.index')
        self.metadata_path = os.path.join(self.data_dir, 'metadata.pkl')
        
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SentenceTransformer model."""
        print(f"Loading SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _initialize_index(self):
        """Initialize FAISS index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product index for cosine similarity

    def read_text_file(self, file_path: str) -> str:
        """Read and return text from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist at path: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def clean_text(self, text: str) -> str:
        """Clean text with enhanced preprocessing."""
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\s.,!?:;()\-\'"]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(word_tokenize(text))

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Create smart chunks from text."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size and we have content
            if current_size + sentence_tokens > self.chunk_size and current_size > 0:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_size
                })
                
                # Calculate overlap
                overlap_tokens = 0
                overlap_sentences = []
                
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap_size:
                        overlap_tokens += sent_tokens
                        overlap_sentences.insert(0, sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_tokens
            
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "token_count": current_size
            })
        
        return chunks

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate embeddings for chunks."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        return embeddings, chunks

    def process_file(self, file_path: str) -> None:
        """Process a single file."""
        try:
            print(f"Processing file: {file_path}")
            
            # Read and clean text
            text = self.read_text_file(file_path)
            cleaned_text = self.clean_text(text)
            
            # Create chunks
            chunks = self.chunk_text(cleaned_text)
            print(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings, chunks = self.generate_embeddings(chunks)
            
            # Add file information to chunks
            file_name = os.path.basename(file_path)
            for i, chunk in enumerate(chunks):
                chunk.update({
                    "file_name": file_name,
                    "chunk_id": len(self.metadata) + i,
                    "file_path": file_path
                })
            
            # Add to index
            if self.index is None:
                self._initialize_index()
            
            self.index.add(embeddings)
            self.metadata.extend(chunks)
            
            print(f"Successfully processed {file_name}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            traceback.print_exc()

    def process_directory(self, directory_path: str) -> None:
        """Process all text files in directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        txt_files = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith('.txt') and os.path.isfile(os.path.join(directory_path, f))
        ]
        
        if not txt_files:
            raise ValueError(f"No .txt files found in {directory_path}")
        
        print(f"Found {len(txt_files)} files to process")
        
        for file_path in tqdm(txt_files, desc="Processing files"):
            self.process_file(file_path)
        
        print("\nCompleted processing all files")
        self.save_index()

    def save_index(self) -> None:
        """Save FAISS index and metadata."""
        if self.index is not None:
            print(f"\nSaving data to {self.data_dir}")
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save config
            config = {
                'model_name': self.model_name,
                'chunk_size': self.chunk_size,
                'overlap_size': self.overlap_size,
                'embedding_dim': self.embedding_dim
            }
            
            config_path = os.path.join(self.data_dir, 'config.pkl')
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
            
            print(f"Saved index, metadata, and config to {self.data_dir}/")

    def load_index(self) -> None:
        """Load saved FAISS index and metadata."""
        print(f"\nLoading data from {self.data_dir}")
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Index or metadata file not found in {self.data_dir}/")
        
        # Load index
        self.index = faiss.read_index(self.index_path)
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load config
        config_path = os.path.join(self.data_dir, 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                print("Loaded configuration:")
                for key, value in config.items():
                    print(f"- {key}: {value}")
        
        print(f"Successfully loaded {len(self.metadata)} chunks from index")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        if self.index is None:
            raise ValueError("No index loaded. Either process files or load an existing index.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for invalid indices
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results

if __name__ == "__main__":
    # Configuration
    config = {
        'directory_path': r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\preproc_nlp\cleaned_data",
        'model_name': 'multi-qa-mpnet-base-dot-v1',
        'chunk_size': 250,
        'overlap_size': 50,
        'data_dir': 'vector_store'  # Data will be stored in this directory
    }
    
    # Initialize and run chunker
    chunker = SmartChunker(
        model_name=config['model_name'],
        chunk_size=config['chunk_size'],
        overlap_size=config['overlap_size'],
        data_dir=config['data_dir']
    )
    
    # Process files and build index
    chunker.process_directory(config['directory_path'])
    
    # Test search
    test_query = "What is cryptocurrency?"
    results = chunker.search(test_query, k=3)
    
    print("\nSearch Results:")
    for result in results:
        print(f"\nFile: {result['file_name']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:200]}...")