"""
Script for processing and storing text data with embeddings in a MongoDB vector database.

Assumptions:
- The .txt file to process is located at 'preproc_nlp/nlp_cleaned_data/document-1_processed_processed.txt'.
- MongoDB is running and accessible with a connection URI.
- A Sentence Transformers model is used for generating text embeddings.
- NLTK is used for tokenization and sentence segmentation.
"""

import os
import re
import sys
import traceback
from pymongo import MongoClient, errors
from sentence_transformers import SentenceTransformer
import nltk

# Download necessary NLTK data (only if not downloaded previously)
#nltk.download('punkt')

def read_text_file(file_path):
    """
    Reads text from the provided file path.
    :param file_path: Path to the .txt file.
    :return: String content of the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist at path: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_text(text):
    """
    Cleans the text by removing special characters and excess whitespace.
    :param text: Raw text string.
    :return: Cleaned text.
    """
    # Remove special characters except standard punctuation and alphanumerics
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def segment_sentences(text):
    """
    Tokenizes text into sentences using NLTK.
    :param text: The cleaned text.
    :return: List of sentences.
    """
    sentences = nltk.sent_tokenize(text)
    return sentences

def tokenize_text(text):
    """
    Tokenizes text into words using NLTK's word tokenizer.
    :param text: A sentence string.
    :return: List of tokens.
    """
    return nltk.word_tokenize(text)

def generate_embeddings(sentences, model):
    """
    Generates embeddings for each sentence.
    :param sentences: List of sentences.
    :param model: The sentence transformer model for generating embeddings.
    :return: List of embedding vectors.
    """
    # Using the model to encode sentences to vector embeddings
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

def connect_mongo(uri, db_name):
    """
    Establishes a connection to MongoDB with error handling.
    :param uri: MongoDB connection URI.
    :param db_name: Name of the database to connect to.
    :return: MongoDB database instance.
    """
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Trigger connection on a request as the connect=True parameter of MongoClient is deprecated
        client.server_info()  
        db = client[db_name]
        print("Successfully connected to MongoDB.")
        return db
    except errors.ServerSelectionTimeoutError as err:
        print("Error: Could not connect to MongoDB.")
        traceback.print_exc()
        sys.exit(1)

def create_vector_collection(db, collection_name):
    """
    Creates a collection and ensures that a vector index is created.
    For MongoDB 6.0+ with vector search support, you might need to configure your index accordingly.
    :param db: MongoDB database instance.
    :param collection_name: Name of the collection.
    :return: MongoDB collection instance.
    """
    try:
        collection = db[collection_name]
        
        # Drop collection if exists (for demo purposes only; remove in production)
        collection.drop()
        print(f"Collection '{collection_name}' dropped (if it existed) and will be recreated.")
        
        # Create the collection and set an index on the embedding vector field.
        # The following index creation is an example. Adjust the index type and key depending on your MongoDB version and use-case.
        # For example, if using MongoDB with vector search, you might use a "knnVector" index.
        index_name = collection.create_index([("embedding", "2dsphere")])
        print(f"Index created: {index_name}")
        return collection
    except Exception as e:
        print("Error creating vector collection and index:")
        traceback.print_exc()
        sys.exit(1)

def store_embeddings(collection, sentences, embeddings):
    """
    Stores the sentences and their corresponding embeddings in MongoDB.
    :param collection: MongoDB collection instance.
    :param sentences: List of sentences.
    :param embeddings: List of embeddings.
    """
    if len(sentences) != len(embeddings):
        raise ValueError("The number of sentences and embeddings must match.")
    
    documents = []
    for sentence, embedding in zip(sentences, embeddings):
        # Create a document structure; adjust fields as needed.
        doc = {
            "sentence": sentence,
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding  # Ensure numpy arrays are stored as lists
        }
        documents.append(doc)
    
    try:
        result = collection.insert_many(documents)
        print(f"Inserted {len(result.inserted_ids)} documents into the collection.")
    except Exception as e:
        print("Error inserting documents into MongoDB:")
        traceback.print_exc()

def main():
    # Configuration
    file_path = r"C:\Users\ferie\Documents\ferielWork\4ds2PIDS\Due_Diligence_4DS2\preproc_nlp\cleaned_data\ssrn-3305362_processed.txt"
    mongo_uri = "mongodb+srv://Feriel:Feriel@cluster0.81oai.mongodb.net/" 
    db_name = "Crypto"
    collection_name = "VectorDB"
    
    try:
        # Read and process the file
        raw_text = read_text_file(file_path)
        cleaned_text = clean_text(raw_text)
        sentences = segment_sentences(cleaned_text)
        print(f"Segmented into {len(sentences)} sentences.")
        
        # Optionally, you could further tokenize each sentence if needed
        # tokens = [tokenize_text(sentence) for sentence in sentences]
        
        # Load the Sentence Transformer model 
        # (These models are part of the Sentence Transformers library, which is built on top of Hugging Face's Transformers.)
        model_name = 'all-mpnet-base-v2'
        model = SentenceTransformer(model_name)
        print(f"Model '{model_name}' loaded.")
        
        # Generate embeddings for each sentence
        embeddings = generate_embeddings(sentences, model)
        
        # Connect to MongoDB and setup the collection
        db = connect_mongo(mongo_uri, db_name)
        collection = create_vector_collection(db, collection_name)
        
        # Store the sentences and embeddings into MongoDB
        store_embeddings(collection, sentences, embeddings)
        
    except Exception as e:
        print("An unexpected error occurred:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
