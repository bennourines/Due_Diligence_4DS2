import os
from smart_chunker import SmartChunker
from vector_store import add_vector_store_methods

def setup_directories(dirs):
    """Create required directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory {dir_path} is ready")

def process_and_index(input_folder="data/cleaned_texts",
                      chunks_folder="chunks",
                      index_folder="faiss_index"):
    """Complete pipeline: process texts, create chunks, and build index"""
    print("🚀 Starting document processing pipeline...")

    # Create directories
    setup_directories([input_folder, chunks_folder, index_folder])

    # Initialize chunker with vector store capabilities
    VectorEnabledChunker = add_vector_store_methods(SmartChunker)
    chunker = VectorEnabledChunker(data_dir=index_folder)

    # Process texts
    print(f"\n📑 Processing texts from {input_folder}...")
    num_processed = chunker.process_all_texts(input_folder, chunks_folder)
    print(f"✅ Processed {num_processed} document(s)")

    # Create vectorstore
    print(f"\n🔍 Creating vectorstore in {index_folder}...")
    chunker.create_vectorstore(chunks_folder, index_folder)
    print("✅ Pipeline complete!")

    return chunker

def create_sample_file():
    """Create a sample text file for testing"""
    os.makedirs("data/cleaned_texts", exist_ok=True)
    
    with open("data/cleaned_texts/sample.txt", "w") as f:
        f.write("""
        Cryptocurrency compliance has become a major concern for blockchain projects. Bitcoin and Ethereum transactions require proper KYC and AML procedures.
        Financial institutions partnering with crypto exchanges need to implement risk assessment protocols to prevent money laundering.
        Blockchain technology enables transparency but also presents unique regulatory challenges.
        Many DeFi protocols operate in regulatory gray areas which increases their risk profile.
        """)
    
    print("Created sample file at data/cleaned_texts/sample.txt")

if __name__ == "__main__":
    # Create a sample file for testing
    create_sample_file()
    
    # Run the processing pipeline
    chunker = process_and_index()
