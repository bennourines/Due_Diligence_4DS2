import os
from smart_chunker import SmartChunker
from vector_store import add_vector_store_methods
from process_pipeline import create_sample_file

def test_search(chunker, query: str, k: int = 3) -> None:
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
        print(f"   Risk Score: {result.get('risk_score', 'N/A')}")

        # Print entities if available
        if result.get('entities'):
            print("   Entities:")
            for etype, elist in result['entities'].items():
                if elist:
                    print(f"     - {etype.capitalize()}: {', '.join(elist[:3])}")

        # Print text preview
        print(f"   Text: {result['text'][:200]}...")
        print("-" * 50)

def run_test_queries():
    """Run a series of test queries on the vector store"""
    # Test queries
    test_queries = [
        "What is cryptocurrency?",
        "How do cryptocurrency wallets work?",
        "What are the regulatory requirements for crypto?",
        "Explain blockchain technology",
        "What are the risks of cryptocurrency investment?",
        "Identify potential fraud in crypto transactions",
        "What are KYC requirements for exchanges?"
    ]
    
    # Initialize the chunker with vector store capabilities
    VectorEnabledChunker = add_vector_store_methods(SmartChunker)
    chunker = VectorEnabledChunker()
    
    # Check if index exists, if not run the pipeline
    if not os.path.exists("faiss_index") or not os.listdir("faiss_index"):
        print("No index found, creating sample data and building index...")
        create_sample_file()
        
        # Process text
        chunker.process_all_texts("data/cleaned_texts", "chunks")
        
        # Create vectorstore
        chunker.create_vectorstore("chunks", "faiss_index")
    else:
        # Load existing index
        chunker.load_index("faiss_index")
    
    # Run test searches
    for query in test_queries:
        test_search(chunker, query, k=3)

if __name__ == "__main__":
    run_test_queries()
