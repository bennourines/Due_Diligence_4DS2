#version 17 avr avc all data from ines
import os
import numpy as np
import warnings
import requests
import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import glob

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='tensorflow')

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"  # Embedding model
OLLAMA_MODEL = "phi4:latest"  # LLM model
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
FAISS_INDEX_PATH = "./faiss_index_ines/index.faiss"
METADATA_PATH = "./faiss_index_ines/index_metadata.json"
DATA_DIR = "./DATA_COMPLETE/cleaned_texts"  # Directory with text files to index

# Load SentenceTransformer model for generating embeddings
def load_embedding_model(model_name: str):
    print(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)

# Create FAISS index from documents
def create_faiss_index(data_dir: str, model_name: str, index_path: str, metadata_path: str):
    print(f"Creating FAISS index from documents in {data_dir}...")
    
    # Load model
    model = load_embedding_model(model_name)
    
    # Initialize lists to store document chunks and metadata
    texts = []
    metadata = []
    
    # Process all text files in the data directory
    for file_path in glob.glob(f"{data_dir}/*.txt"):
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Simple chunking approach - split by paragraphs and limit size
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():  # Skip empty paragraphs
                    texts.append(para)
                    metadata.append({
                        "text": para,
                        "file_name": file_name,
                        "chunk_id": i
                    })
    
    if not texts:
        print("No texts found to index!")
        return None, None
    
    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create FAISS index - using L2 distance (can be changed to inner product)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Add vectors to the index
    index.add(np.array(embeddings).astype('float32'))
    
    # Save index and metadata
    print(f"Saving index to {index_path} and metadata to {metadata_path}")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Created FAISS index with {index.ntotal} vectors")
    return index, metadata

# Load existing FAISS index
def load_faiss_index(index_path: str, metadata_path: str):
    print(f"Loading FAISS index from {index_path}...")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("Index or metadata file not found. Creating new index...")
        return None, None
    
    try:
        index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
        return index, metadata
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        return None, None

# Retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(question: str, model, index, metadata, top_k: int = 3):
    try:
        # Generate query embedding
        query_embedding = model.encode([question], show_progress_bar=False)
        
        # Search the index
        distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
        
        # Get the relevant chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                doc = metadata[idx]
                # Add distance score (convert to similarity score)
                doc_with_score = doc.copy()
                doc_with_score["score"] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity
                results.append(doc_with_score)
        
        if results:
            print(f"Found {len(results)} relevant chunks using FAISS")
            return results
        else:
            print("No relevant chunks found")
            return []
            
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        return []

# Generate answer using Ollama
def generate_answer_with_llm(question: str, context: str, model: str = OLLAMA_MODEL):
    prompt = (
        "You are a highly knowledgeable expert in cryptocurrency, blockchain technology, "
        "and decentralized finance (DeFi), with extensive experience "
        "in both technical and practical aspects of the field. "
        "Your expertise includes understanding of consensus mechanisms, smart contracts, "
        "tokenomics, mining/staking processes, cryptocurrency exchanges, and regulatory "
        "frameworks across different jurisdictions. You will analyze the provided context thoroughly "
        "and respond to questions with precise, factual information supported by the given context. "
        "If the information cannot be found in the context, you will clearly state this limitation rather than making assumptions or providing speculative answers."
        " When answering, use clear, technical language while maintaining accessibility for both beginners and advanced users. "
        "Break down complex concepts when necessary, and provide relevant examples from the context to support your explanations."

        "Use the following context to answer the question. If you cannot find the answer in the context, "
        "explicitly state 'I cannot find this information in the provided context' and do not make up"
        "information that isn't directly supported by the context."

        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer: "
    )
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error calling Ollama: {response.text}")
    except Exception as e:
        print(f"Error in generate_answer_with_llm: {str(e)}")
        return f"Error generating answer: {str(e)}"

def main():
    print("\nInitializing RAG system...")
    
    # Load embedding model
    print("Loading embedding model...")
    embedding_model = load_embedding_model(MODEL_NAME)
    
    # Load or create FAISS index
    print("Loading or creating FAISS index...")
    index, metadata = load_faiss_index(FAISS_INDEX_PATH, METADATA_PATH)
    
    if index is None or metadata is None:
        print("Creating new index...")
        index, metadata = create_faiss_index(DATA_DIR, MODEL_NAME, FAISS_INDEX_PATH, METADATA_PATH)
        if index is None or metadata is None:
            print("Failed to create index. Exiting.")
            return
    
    print("\nRAG system is ready! You can start asking questions.")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("\nGoodbye!")
            break
            
        if not question:
            print("Please enter a valid question.")
            continue
            
        print("\nProcessing your question...")
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(question, embedding_model, index, metadata, top_k=3)
        
        if not relevant_chunks:
            print("No relevant information found in the documents.")
            continue
        
        # Combine retrieved chunks into context
        context = "\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Generate answer using LLM
        print("\nGenerating answer...")
        try:
            answer = generate_answer_with_llm(question, context, OLLAMA_MODEL)
            print("\nAnswer:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
        except Exception as e:
            print(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()