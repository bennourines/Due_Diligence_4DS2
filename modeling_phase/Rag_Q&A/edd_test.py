# edd_test.py
"""
Test script specifically for Enhanced Due Diligence (EDD) functionality
"""

import os
from dotenv import load_dotenv
from chunk_2 import load_whitepapers, create_domain_specific_chunks, build_vector_database, setup_qa_system, query_system

# Load environment variables
load_dotenv()

def test_edd_queries():
    """Test specific queries related to Enhanced Due Diligence"""
    # Load whitepaper texts
    print("Loading whitepaper texts...")
    raw_text = load_whitepapers()
    
    # Create chunks
    print("Creating domain-specific chunks...")
    chunks = create_domain_specific_chunks(raw_text)
    
    # Build vector database
    print("Building vector database...")
    vectorstore = build_vector_database(chunks)
    
    # Set up QA system
    print("Setting up QA system...")
    qa_chain = setup_qa_system(vectorstore)
    
    # Test EDD-specific queries
    edd_queries = [
        "What is Enhanced Due Diligence (EDD) in the context of cryptocurrency compliance?",
        "How does EDD differ from standard KYC procedures?",
        "What are the key risk indicators that trigger EDD for cryptocurrency transactions?",
        "What documentation is typically required for EDD in cryptocurrency exchanges?",
        "How should ongoing monitoring be implemented for high-risk crypto customers?"
    ]
    
    print("\n=== Enhanced Due Diligence (EDD) Test Results ===\n")
    
    for i, query in enumerate(edd_queries, 1):
        print(f"Query {i}: {query}")
        result = query_system(qa_chain, query)
        print(f"Answer: {result['answer']}\n")
        print("Top source excerpt:")
        if result["sources"]:
            print(result["sources"][0])
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    test_edd_queries()