from text_processing_pipeline import TextProcessor, VectorDatabase, process_whitepapers
import os

def test_pipeline():
    # Initialize the text processor
    processor = TextProcessor()
    
    # Test with a sample text
    sample_text = """
    Blockchain technology enables decentralized applications and smart contracts.
    The Ethereum network is a popular platform for building dApps.
    Cryptocurrencies like Bitcoin and Ethereum use blockchain for secure transactions.
    """
    
    # Test text cleaning
    cleaned_text = processor.clean_text(sample_text)
    print("Cleaned Text:")
    print(cleaned_text)
    print("\n" + "="*50 + "\n")
    
    # Test text processing
    chunks = processor.process_text(sample_text)
    print("Processed Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
    print("\n" + "="*50 + "\n")
    
    # Test embeddings
    embeddings = processor.create_embeddings(chunks)
    print("Embeddings shape:", embeddings.shape)
    print("\n" + "="*50 + "\n")
    
    # Test vector database
    vector_db = VectorDatabase()
    metadata = [{"source": "test_sample"} for _ in range(len(chunks))]
    vector_db.store_embeddings(chunks, embeddings, metadata)
    
    # Save and load the database
    vector_db.save("test_db")
    loaded_db = VectorDatabase()
    loaded_db.load("test_db")
    
    print("Vector database test completed successfully!")
    print("Number of documents stored:", len(loaded_db.documents))
    
    # Process actual whitepapers
    print("\nProcessing whitepapers...")
    whitepapers_dir = "Amine/whitepapers/txt_whitepapers_np"
    process_whitepapers(whitepapers_dir)
    print("Whitepaper processing completed!")

if __name__ == "__main__":
    test_pipeline() 