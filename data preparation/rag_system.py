import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import faiss
import json
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        model_name: str = "llama3:8b",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_type: str = "ollama",
        model_kwargs: Optional[Dict] = None,
        similarity_threshold: float = 0.5,
        vector_store_type: str = "faiss"  # Options: "faiss", "semantic", "llm"
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.model_type = model_type
        self.model_kwargs = model_kwargs or {}
        self.similarity_threshold = similarity_threshold
        self.vector_store_type = vector_store_type
        
        # Initialize components
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize LLM based on model type
        self.initialize_llm()
        
        # Load vector store and chunks
        self.load_vector_store()
        self.chunks = self.load_chunks()

    def initialize_llm(self) -> None:
        """Initialize the language model based on the specified type."""
        try:
            if self.model_type == "ollama":
                # Initialize Ollama with streaming support
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                self.llm = Ollama(
                    model=self.model_name,
                    callback_manager=callback_manager,
                    **self.model_kwargs
                )
            elif self.model_type == "huggingface":
                # Add HuggingFace initialization here
                pass
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            logger.info(f"Successfully initialized {self.model_type} model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    def load_vector_store(self) -> None:
        """Load the appropriate vector store based on the selected type."""
        try:
            vector_store_path = Path("vector_store")
            
            if self.vector_store_type == "faiss":
                # Load main FAISS index
                faiss_path = vector_store_path / "faiss_index"
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info("Loaded main FAISS index")
                
            elif self.vector_store_type == "semantic":
                # Load semantic index
                semantic_path = vector_store_path / "semantic_index.faiss"
                self.faiss_index = faiss.read_index(str(semantic_path))
                logger.info("Loaded semantic FAISS index")
                
            elif self.vector_store_type == "llm":
                # Load LLM index
                llm_path = vector_store_path / "llm_index.faiss"
                self.faiss_index = faiss.read_index(str(llm_path))
                logger.info("Loaded LLM FAISS index")
                
            else:
                raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
            
            # Load embeddings
            embeddings_path = vector_store_path / "embeddings.pt"
            self.embeddings = torch.load(str(embeddings_path))
            
            logger.info(f"Successfully loaded {self.vector_store_type} vector store")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunks from the chunks directory."""
        chunks = []
        chunks_dir = Path("chunks")
        
        for file_path in chunks_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_chunks = json.load(f)
                    chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Error loading chunks from {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks

    def get_relevant_chunks(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Get the most relevant chunks for a query using the selected vector store."""
        try:
            # Get query embedding
            query_embedding = self.embedder.encode([query])[0]
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                k
            )
            
            # Get relevant chunks
            relevant_chunks = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    similarity = float(1 / (1 + distance))  # Convert distance to similarity
                    if similarity >= self.similarity_threshold:  # Only include chunks above threshold
                        chunk["similarity"] = similarity
                        relevant_chunks.append(chunk)
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks with similarity >= {self.similarity_threshold}")
            return relevant_chunks
        except Exception as e:
            logger.error(f"Error getting relevant chunks: {str(e)}")
            return []

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format the context from relevant chunks."""
        if not chunks:
            return "No relevant context found."
        
        formatted_context = "Relevant information from our knowledge base:\n\n"
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown source")
            formatted_context += f"Document {i} (Source: {source}, Similarity: {chunk['similarity']:.2f}):\n"
            formatted_context += f"{chunk['text']}\n\n"
        
        return formatted_context

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system and get a response with source information."""
        try:
            # Get relevant chunks
            relevant_chunks = self.get_relevant_chunks(question)
            
            # Format context
            context = self.format_context(relevant_chunks)
            
            # Create prompt template
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert in digital assets and blockchain-based financial products.
                Your goal is to answer user questions by synthesizing information from the provided context.
                
                Guidelines:
                - Use the provided context as your primary source of information
                - If the answer can be found in the context, use it directly
                - If the context doesn't contain enough information, say so clearly
                - Keep answers concise (3-5 sentences) unless more detail is requested
                - Use bullet points for lists
                - Highlight key metrics or dates in **bold**
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:"""
            )
            
            # Create LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Generate response
            response = chain.run(context=context, question=question)
            
            # Determine if the answer is from context or base knowledge
            is_from_context = bool(relevant_chunks)  # If we have relevant chunks, assume answer is from context
            
            return {
                "answer": response,
                "source": "context" if is_from_context else "base_knowledge",
                "relevant_chunks": relevant_chunks,
                "vector_store_type": self.vector_store_type
            }
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise

def main():
    # Initialize RAG system with different vector stores
    vector_stores = ["faiss", "semantic", "llm"]
    
    for store_type in vector_stores:
        try:
            print(f"\nTrying vector store: {store_type}")
            rag = RAGSystem(
                model_name="llama3:8b",
                model_type="ollama",
                model_kwargs={"temperature": 0.1},
                similarity_threshold=0.5,
                vector_store_type=store_type
            )
            
            # Interactive query loop
            print(f"RAG System is ready with {store_type} vector store. Type 'exit' to try next store.")
            while True:
                question = input("\nEnter your question: ")
                if question.lower() == 'exit':
                    break
                
                try:
                    result = rag.query(question)
                    print("\nAnswer:")
                    print(f"Source: {result['source']}")
                    print(f"Vector Store: {result['vector_store_type']}")
                    print(result['answer'])
                    
                    if result['relevant_chunks']:
                        print("\nRelevant sources:")
                        for chunk in result['relevant_chunks']:
                            source = chunk.get("metadata", {}).get("source", "Unknown source")
                            print(f"- {source} (similarity: {chunk['similarity']:.2f})")
                except Exception as e:
                    print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Failed to initialize RAG system with {store_type} vector store: {str(e)}")
            continue

if __name__ == "__main__":
    main() 