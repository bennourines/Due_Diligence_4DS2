"""
Script for processing text files using sliding window chunking with sentence boundaries,
generating embeddings with SentenceTransformers, and storing in MongoDB.
"""

import os
import re
import sys
import traceback
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from pymongo import MongoClient, errors
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data (uncomment if needed)
# nltk.download('punkt')

def read_text_file(file_path: str) -> str:
    """
    Reads text from the provided file path.
    :param file_path: Path to the .txt file.
    :return: String content of the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist at path: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_text(text: str) -> str:
    """
    Cleans the text by removing special characters and excess whitespace.
    :param text: Raw text string.
    :return: Cleaned text.
    """
    # Remove special characters except standard punctuation and alphanumerics
    text = re.sub(r'[^\w\s.,!?:;()\-\']', ' ', text)
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def count_tokens(text: str) -> int:
    """
    Count the approximate number of tokens in the text.
    :param text: Text to count tokens for.
    :return: Number of tokens.
    """
    # Simple word-based tokenization for estimation
    return len(word_tokenize(text))

def sliding_window_chunking(text: str, chunk_size: int = 250, overlap_size: int = 50) -> List[Dict[str, Any]]:
    """
    Chunks text using sliding window with respect to sentence boundaries.
    
    :param text: The full text to chunk.
    :param chunk_size: Target number of tokens per chunk.
    :param overlap_size: Number of tokens to overlap between chunks.
    :return: List of dictionaries containing chunks and metadata.
    """
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize variables
    chunks = []
    current_chunk = []
    current_size = 0
    sentence_idx = 0
    
    while sentence_idx < len(sentences):
        sentence = sentences[sentence_idx]
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence would exceed the chunk size and we already have content,
        # finalize the current chunk
        if current_size + sentence_tokens > chunk_size and current_size > 0:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "token_count": current_size
            })
            
            # Calculate how many sentences to go back for the overlap
            overlap_token_count = 0
            sentences_to_keep = []
            
            # Go backwards through current_chunk to determine overlap
            for sent in reversed(current_chunk):
                sent_tokens = count_tokens(sent)
                if overlap_token_count + sent_tokens <= overlap_size:
                    overlap_token_count += sent_tokens
                    sentences_to_keep.insert(0, sent)
                else:
                    break
            
            # Reset with overlap sentences
            current_chunk = sentences_to_keep
            current_size = overlap_token_count
        
        # Add the current sentence
        current_chunk.append(sentence)
        current_size += sentence_tokens
        sentence_idx += 1
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "token_count": current_size
        })
    
    return chunks

def generate_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    Generates embeddings for each text chunk.
    :param chunks: List of text chunks.
    :param model: SentenceTransformer model.
    :return: List of chunks with added embeddings.
    """
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batches (better for performance)
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Add embeddings to the chunks
    for i, embedding in enumerate(embeddings):
        chunks[i]["embedding"] = embedding.tolist()
    
    return chunks

def connect_to_mongodb(uri: str, db_name: str) -> Any:
    """
    Connects to MongoDB database.
    :param uri: MongoDB connection URI.
    :param db_name: Name of the database.
    :return: MongoDB database object.
    """
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Check connection
        client.server_info()
        db = client[db_name]
        print("Successfully connected to MongoDB.")
        return db
    except errors.ServerSelectionTimeoutError:
        print("Error: Could not connect to MongoDB.")
        traceback.print_exc()
        sys.exit(1)

def store_in_mongodb(db: Any, collection_name: str, chunks: List[Dict[str, Any]], file_metadata: Dict[str, Any]) -> None:
    """
    Stores chunks with embeddings in MongoDB.
    :param db: MongoDB database object.
    :param collection_name: Collection name.
    :param chunks: List of text chunks with embeddings.
    :param file_metadata: Metadata about the processed file.
    """
    # Get or create collection
    collection = db[collection_name]
    
    # Create a vector index if it doesn't exist
    # Note: Assuming MongoDB 6.0+ with vector search capability
    # You may need to adjust this based on your MongoDB version and setup
    try:
        collection.create_index([("embedding", "2dsphere")])
        print("Created vector index on 'embedding' field.")
    except Exception as e:
        print(f"Note: Could not create index: {str(e)}")
    
    # Prepare documents for insertion
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "chunk_id": i,
            "file_id": file_metadata["file_id"],
            "file_name": file_metadata["file_name"],
            "text": chunk["text"],
            "token_count": chunk["token_count"],
            "embedding": chunk["embedding"]
        }
        documents.append(doc)
    
    # Insert documents
    if documents:
        result = collection.insert_many(documents)
        print(f"Inserted {len(result.inserted_ids)} chunks into MongoDB.")
    else:
        print("No chunks to insert.")

def process_text_file(
    file_path: str, 
    mongo_uri: str, 
    db_name: str, 
    collection_name: str,
    model_name: str = 'all-MiniLM-L6-v2',
    chunk_size: int = 250,
    overlap_size: int = 50
) -> None:
    """
    Main function to process a text file and store chunks with embeddings in MongoDB.
    
    :param file_path: Path to the input text file.
    :param mongo_uri: MongoDB connection URI.
    :param db_name: MongoDB database name.
    :param collection_name: MongoDB collection name.
    :param model_name: SentenceTransformer model name.
    :param chunk_size: Target number of tokens per chunk.
    :param overlap_size: Number of tokens to overlap between chunks.
    """
    try:
        # Read and clean the text
        print(f"Reading file: {file_path}")
        raw_text = read_text_file(file_path)
        cleaned_text = clean_text(raw_text)
        
        # Get file metadata
        file_name = os.path.basename(file_path)
        file_id = file_name.replace('.', '_')
        file_metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "file_path": file_path
        }
        
        # Create chunks using sliding window
        print(f"Chunking text with size {chunk_size} and overlap {overlap_size}...")
        chunks = sliding_window_chunking(cleaned_text, chunk_size, overlap_size)
        print(f"Created {len(chunks)} chunks.")
        
        # Load SentenceTransformer model
        print(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Generate embeddings
        print("Generating embeddings...")
        chunks_with_embeddings = generate_embeddings(chunks, model)
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB: {db_name}.{collection_name}")
        db = connect_to_mongodb(mongo_uri, db_name)
        
        # Store chunks with embeddings
        print("Storing chunks in MongoDB...")
        store_in_mongodb(db, collection_name, chunks_with_embeddings, file_metadata)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()

def process_directory(
    directory_path: str,
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    model_name: str = 'all-MiniLM-L6-v2',
    chunk_size: int = 250,
    overlap_size: int = 50
) -> None:
    """
    Process all .txt files in the given directory.
    
    :param directory_path: Path to the directory containing .txt files.
    :param mongo_uri: MongoDB connection URI.
    :param db_name: MongoDB database name.
    :param collection_name: MongoDB collection name.
    :param model_name: SentenceTransformer model name.
    :param chunk_size: Target number of tokens per chunk.
    :param overlap_size: Number of tokens to overlap between chunks.
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    
    # Get all .txt files in the directory
    txt_files = [
        os.path.join(directory_path, file) 
        for file in os.listdir(directory_path) 
        if file.endswith('.txt') and os.path.isfile(os.path.join(directory_path, file))
    ]
    
    if not txt_files:
        print(f"No .txt files found in {directory_path}")
        return
    
    print(f"Found {len(txt_files)} .txt files to process")
    
    # Load model once to reuse for all files
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Connect to MongoDB once
    print(f"Connecting to MongoDB: {db_name}.{collection_name}")
    db = connect_to_mongodb(mongo_uri, db_name)
    
    # Process each file
    for i, file_path in enumerate(txt_files):
        try:
            print(f"\n[{i+1}/{len(txt_files)}] Processing file: {os.path.basename(file_path)}")
            
            # Read and clean the text
            raw_text = read_text_file(file_path)
            cleaned_text = clean_text(raw_text)
            
            # Get file metadata
            file_name = os.path.basename(file_path)
            file_id = file_name.replace('.', '_')
            file_metadata = {
                "file_id": file_id,
                "file_name": file_name,
                "file_path": file_path
            }
            
            # Create chunks using sliding window
            print(f"Chunking text with size {chunk_size} and overlap {overlap_size}...")
            chunks = sliding_window_chunking(cleaned_text, chunk_size, overlap_size)
            print(f"Created {len(chunks)} chunks.")
            
            # Generate embeddings
            print("Generating embeddings...")
            chunks_with_embeddings = generate_embeddings(chunks, model)
            
            # Store chunks with embeddings
            print("Storing chunks in MongoDB...")
            store_in_mongodb(db, collection_name, chunks_with_embeddings, file_metadata)
            
            print(f"Successfully processed {file_name}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            traceback.print_exc()
            print("Continuing with next file...")
            continue
    
    print(f"\nCompleted processing all files in {directory_path}")

if __name__ == "__main__":
    # Configuration
    directory_path = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\preproc_nlp\cleaned_data"
    mongo_uri = "mongodb+srv://Feriel:Feriel@cluster0.81oai.mongodb.net/" 
    db_name = "Crypto"
    collection_name = "whitepaper_chunks"
    
    # Process all files in the directory
    process_directory(
        directory_path=directory_path,
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
        model_name='multi-qa-mpnet-base-dot-v1',
        chunk_size=250,
        overlap_size=50
    )