"""
Script to test the FAISS vectorstore functionality by loading saved index and performing searches.
"""

import os
from smart_chunking_faiss import SmartChunker

def test_search(chunker: SmartChunker, query: str, k: int = 3) -> None:
    """
    Test search functionality and print results.
    
    Args:
        chunker: Initialized SmartChunker instance
        query: Search query
        k: Number of results to return
    """
    print(f"\nSearching for: {query}")
    print("=" * 50)
    
    results = chunker.search(query, k=k)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. File: {result['file_name']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Text: {result['text'][:200]}...")
        print("-" * 50)

def main():
    # Initialize chunker with saved data
    data_dir = "vector_store"  # Directory where the index and metadata are stored
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            "Please run smart_chunking_faiss.py first to create the vectorstore."
        )
    
    print(f"Loading vectorstore from {data_dir}")
    chunker = SmartChunker(data_dir=data_dir)
    chunker.load_index()
    
    # Test queries
    test_queries = [
        "What is cryptocurrency?",
        "How do cryptocurrency wallets work?",
        "What are the regulatory requirements for crypto?",
        "Explain blockchain technology",
        "What are the risks of cryptocurrency investment?"
    ]
    
    # Run test searches
    for query in test_queries:
        test_search(chunker, query)

if __name__ == "__main__":
    main()