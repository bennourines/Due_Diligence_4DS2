"""
vector.py

Builds a vector database from a list of text chunks using embeddings.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def build_vector_database(chunks):
    """Build and save a vector database from text chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Suitable model for financial content
        model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU is available
    )
    # Convert each chunk into a Document with metadata
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i, "source": "whitepaper"})
        for i, chunk in enumerate(chunks)
    ]
    # Create a FAISS vector store based on the documents and embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)
    # Optionally save the vector store for persistence
    vectorstore.save_local("faiss_finance_index")
    print("Vector database saved to 'faiss_finance_index'")
    return vectorstore
