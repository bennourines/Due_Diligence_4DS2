import os
import warnings
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import requests

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# Suppress specific deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='pymongo')

# Configuration
MONGO_URI = "mongodb+srv://Feriel:Feriel@cluster0.81oai.mongodb.net/"
DB_NAME = "Crypto"
COLLECTION_NAME = "whitepaper_chunks"
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"  # Same model used for embedding
OPENROUTER_API_KEY = "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"  
MISTRAL_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1:free"  # Mistral model on OpenRouter
#MISTRAL_MODEL = "meta-llama/llama-4-maverick:free"
# Connect to MongoDB
def connect_to_mongodb(uri: str, db_name: str):
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
        print("Successfully connected to MongoDB.")
        return client[db_name]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

# Load SentenceTransformer model for generating query embeddings
def load_embedding_model(model_name: str):
    print(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)

# Retrieve relevant chunks from MongoDB
def retrieve_relevant_chunks(db, collection_name: str, query_embedding: np.ndarray, top_k: int = 3):
    try:
        collection = db[collection_name]
        
        # Debug: Print collection info
        print(f"\nCollection stats:")
        doc_count = collection.count_documents({})
        print(f"Document count: {doc_count}")
        
        # Print index information in a more readable format
        print("\nIndexes:")
        for index in collection.list_indexes():
            print(f"- {index['name']}: {index['key']}")
        print()
        
        # Check for vector search capability
        try:
            db.command('listSearchIndexes', COLLECTION_NAME)
            print("Vector search is available")
        except Exception as e:
            print(f"Vector search might not be available: {str(e)}")
        
        # Convert query embedding to list for MongoDB query
        query_embedding_list = query_embedding.tolist()
        
        # Implement manual dot product similarity search
        pipeline = [
            {
                "$addFields": {
                    "similarity": {
                        "$reduce": {
                            "input": {"$range": [0, {"$size": "$embedding"}]},
                            "initialValue": 0,
                            "in": {
                                "$add": [
                                    "$$value",
                                    {"$multiply": [
                                        {"$arrayElemAt": ["$embedding", "$$this"]},
                                        {"$arrayElemAt": [query_embedding_list, "$$this"]}
                                    ]}
                                ]
                            }
                        }
                    }
                }
            },
            {"$sort": {"similarity": -1}},
            {"$limit": top_k},
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "file_name": 1,
                    "score": "$similarity"
                }
            }
        ]
        
        try:
            results = list(collection.aggregate(pipeline))
            if results:
                print("Found results using dot product similarity search")
                return results
        except Exception as e:
            print(f"Dot product similarity search failed: {str(e)}")
        
        # Fallback to random sampling if similarity search fails
        print("\nFalling back to random sampling...")
        fallback_pipeline = [
            {"$sample": {"size": top_k}},
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "file_name": 1,
                    "score": {"$literal": 0.0}
                }
            }
        ]
        
        results = list(collection.aggregate(fallback_pipeline))
        print("Retrieved random samples as fallback")
        return results
            
    except Exception as e:
        print(f"Error accessing collection: {str(e)}")
        return []

# Generate answer using Mistral LLM
def generate_answer_with_mistral(question: str, context: str, api_key: str, model: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = (
        "You are a helpful expert in cryptocurrency and blockchain technology. "
        "Use the following context to answer the question. If you cannot find "
        "the answer in the context, say so. Do not make up information.\n\n"

        "Use the following context to answer the question. If you cannot find the answer in the context, "
        "explicitly state 'I cannot find this information in the provided context and do not make up "
        "information that isn't directly supported by the context."

        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer: "
    )
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 5000,
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error calling Mistral: {response.text}")

# Main function to test the RAG system
def test_rag_system(question: str):
    # Connect to MongoDB
    db = connect_to_mongodb(MONGO_URI, DB_NAME)
    
    # Load embedding model
    embedding_model = load_embedding_model(MODEL_NAME)
    
    # Generate query embedding
    query_embedding = embedding_model.encode([question], show_progress_bar=False)
    
    # Retrieve relevant chunks
    try:
        relevant_chunks = retrieve_relevant_chunks(db, COLLECTION_NAME, query_embedding[0], top_k=3)
        if not relevant_chunks:
            print("No relevant chunks found in the database.")
            print("\nTrying to analyze the database structure...")
            try:
                # List all collections in the database
                collections = db.list_collection_names()
                print(f"Available collections: {collections}")
                if COLLECTION_NAME in collections:
                    collection = db[COLLECTION_NAME]
                    print(f"\nSample document from {COLLECTION_NAME}:")
                    sample_doc = collection.find_one()
                    if sample_doc:
                        print(f"Document structure: {list(sample_doc.keys())}")
            except Exception as e:
                print(f"Error analyzing database structure: {str(e)}")
            return
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        return
    
    print("Retrieved Relevant Chunks:")
    for chunk in relevant_chunks:
        score_text = f", Score: {chunk['score']}" if 'score' in chunk else ""
        print(f"- File: {chunk['file_name']}{score_text}")
        print(chunk['text'][:200])  # Print first 200 characters of the chunk
        print("-" * 50)
    
    # Combine retrieved chunks into context
    context = "\n".join([chunk["text"] for chunk in relevant_chunks])
    
    # Generate answer using Mistral
    print("Generating answer with Mistral...")
    try:
        answer = generate_answer_with_mistral(question, context, OPENROUTER_API_KEY, MISTRAL_MODEL)
        print("Generated Answer:")
        print(answer)
    except Exception as e:
        print(f"Error generating answer: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Test with a simple question first
    #print("\nTesting with a simple question...")
    #test_rag_system("What is cryptocurrency?")
    
    # If the simple test works, try the more complex question
    print("\nTesting with the original question...\n")
    #test_rag_system("generate a set of 5 questions that you will extract from the sources in the RAG system")
    test_rag_system("How safe and reliable are online and virtual payment and wallet platforms for cryptocurrency transactions?")

    #test_rag_system("What are the main impacts of cryptocurrency on macroeconomic stability?")
    test_rag_system("How does cryptocurrency affect inflation and deflation?")
    #test_rag_system("What are the different regulatory approaches for cryptocurrency?")
    #test_rag_system("How does cryptocurrency influence exchange rates?")
    #test_rag_system("What is the role of cryptocurrency in global finance?")
    #test_rag_system("How does cryptocurrency impact inflation and deflation?")
    #test_rag_system("What are the different regulatory approaches for cryptocurrency?")
    #How safe and reliable are online and virtual payment and wallet platforms for cryptocurrency transactions?