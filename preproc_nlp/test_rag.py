from vector_database import RAGSystem
import os

# Set OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-xxxxx"  # Replace with actual key

def main():
    # Initialize the RAG system
    rag = RAGSystem(
        embedding_model_name="text-embedding-3-small",
        llm_model="meta-llama/llama-4-maverick:free",
        chunk_size=500,  # Smaller chunks for more precise retrieval
        chunk_overlap=50,
        index_path="./test_vectorstore",
        temperature=0.7,
        max_tokens=512
    )

    # Process the specific document
    input_dir = "./nlp_cleaned_data"
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    print(f"\nProcessing documents from {input_dir}...")
    rag.process_documents(input_dir)
    
    # Test queries
    test_questions = [
        "What are the main impacts of cryptocurrency on macroeconomic stability?",
        "How does cryptocurrency affect inflation and deflation?",
        "What are the different regulatory approaches for cryptocurrency?",
        "How does cryptocurrency influence exchange rates?",
        "What is the role of cryptocurrency in global finance?"
    ]
    
    # Run queries and display results
    for question in test_questions:
        print("\n" + "="*80)
        print(f"\nQuestion: {question}")
        try:
            result = rag.query(question, top_k=3)
            print("\nAnswer:")
            print(result["answer"])
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n{i}. From {doc['source']}:")
                print(f"   {doc['content'][:200]}...")
        except Exception as e:
            print(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()