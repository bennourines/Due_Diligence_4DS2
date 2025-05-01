from text_processing_pipeline import VectorDatabase, TextProcessor
import numpy as np

def check_database():
    # Load the database
    vector_db = VectorDatabase()
    vector_db.load("vector_db")
    
    # Initialize text processor for queries
    processor = TextProcessor()
    
    print("\n=== Database Statistics ===")
    print(f"Total number of chunks: {len(vector_db.documents)}")
    print(f"Total number of unique sources: {len(set(meta['source'] for meta in vector_db.metadata))}")
    
    # Get unique sources and their chunk counts
    source_counts = {}
    for meta in vector_db.metadata:
        source = meta['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\n=== Documents by Source ===")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {count} chunks")
    
    while True:
        print("\n=== Database Explorer ===")
        print("1. Search for similar content")
        print("2. View random chunks")
        print("3. View chunks from specific source")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            # Search for similar content
            query = input("\nEnter your search query: ")
            query_embedding = processor.create_embeddings([query])[0]
            
            # Find similar chunks
            distances, indices = vector_db.index.kneighbors([query_embedding])
            
            print("\nTop 5 most similar chunks:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                print(f"\n--- Result {i+1} (Similarity: {1-dist:.2f}) ---")
                print(f"Source: {vector_db.metadata[idx]['source']}")
                print(f"Content: {vector_db.documents[idx][:200]}...")
        
        elif choice == "2":
            # View random chunks
            num_chunks = int(input("\nHow many random chunks to view? (1-10): "))
            num_chunks = min(max(1, num_chunks), 10)
            
            random_indices = np.random.choice(len(vector_db.documents), num_chunks, replace=False)
            for i, idx in enumerate(random_indices):
                print(f"\n--- Random Chunk {i+1} ---")
                print(f"Source: {vector_db.metadata[idx]['source']}")
                print(f"Content: {vector_db.documents[idx][:200]}...")
        
        elif choice == "3":
            # View chunks from specific source
            source = input("\nEnter source filename (e.g., 'Polkadot.txt'): ")
            source_chunks = [(i, doc) for i, doc in enumerate(vector_db.documents) 
                           if vector_db.metadata[i]['source'] == source]
            
            if not source_chunks:
                print(f"No chunks found for source: {source}")
                continue
            
            print(f"\nFound {len(source_chunks)} chunks from {source}")
            num_to_view = min(5, len(source_chunks))
            
            for i, (idx, chunk) in enumerate(source_chunks[:num_to_view]):
                print(f"\n--- Chunk {i+1} from {source} ---")
                print(f"Content: {chunk[:200]}...")
        
        elif choice == "4":
            print("Exiting database explorer...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    check_database() 