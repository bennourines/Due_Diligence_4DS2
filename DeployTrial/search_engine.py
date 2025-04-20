# search_engine.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from typing import List, Dict, Any, Optional
import os
import pickle
import logging

# Set up logging
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Hybrid search engine combining vector search and keyword search"""
    
    def __init__(self):
        logger.info("Initializing HybridSearchEngine")
        # Initialize HuggingFace embeddings (sentence-transformers)
        logger.debug("Setting up HuggingFace embeddings")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Maps project IDs to search engines
        self.vector_stores = {}
        self.bm25_retrievers = {}
        self.ensemble_retrievers = {}
        
        # Create storage directory
        os.makedirs("vector_stores", exist_ok=True)
        logger.info("Vector stores directory initialized")
    
    def add_documents(self, documents: List, project_id: str):
        """Add documents to vector store and BM25 retriever"""
        try:
            logger.info(f"Adding documents for project: {project_id}")
            logger.debug(f"Processing {len(documents)} documents")
            
            # Create FAISS vector store
            logger.debug("Creating FAISS vector store")
            vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info("FAISS vector store created successfully")
            
            # Create BM25 retriever
            logger.debug("Creating BM25 retriever")
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 5
            logger.info("BM25 retriever created successfully")
            
            # Create ensemble retriever
            logger.debug("Creating ensemble retriever")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                    bm25_retriever
                ],
                weights=[0.7, 0.3]
            )
            logger.info("Ensemble retriever created successfully")
            
            # Store retrievers
            self.vector_stores[project_id] = vector_store
            self.bm25_retrievers[project_id] = bm25_retriever
            self.ensemble_retrievers[project_id] = ensemble_retriever
            
            # Save vector store for persistence
            self._save_vector_store(vector_store, project_id)
            logger.info(f"All components initialized and saved for project: {project_id}")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise
    
    def hybrid_search(self, query: str, project_id: str, k: int = 5) -> List:
        """Perform hybrid search using ensemble retriever"""
        try:
            logger.info(f"Performing hybrid search for project: {project_id}")
            logger.debug(f"Query: {query}, k={k}")
            
            if project_id not in self.ensemble_retrievers:
                logger.debug(f"Ensemble retriever not found for project {project_id}, attempting to load")
                self._load_vector_store(project_id)
                
            if project_id not in self.ensemble_retrievers:
                logger.error(f"No documents found for project ID: {project_id}")
                raise ValueError(f"No documents found for project ID: {project_id}")
            
            # Retrieve documents with ensemble retriever
            logger.debug("Executing ensemble retriever search")
            retriever = self.ensemble_retrievers[project_id]
            results = retriever.invoke(query)
            logger.info(f"Found {len(results)} relevant documents")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}", exc_info=True)
            raise
    
    def _save_vector_store(self, vector_store, project_id: str):
        """Save vector store to disk"""
        try:
            logger.debug(f"Saving vector store for project: {project_id}")
            file_path = f"vector_stores/{project_id}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(vector_store, f)
            logger.info(f"Vector store saved successfully: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}", exc_info=True)
            raise
    
    def _load_vector_store(self, project_id: str):
        """Load vector store from disk"""
        try:
            logger.debug(f"Loading vector store for project: {project_id}")
            file_path = f"vector_stores/{project_id}.pkl"
            
            if os.path.exists(file_path):
                logger.debug(f"Found vector store file: {file_path}")
                with open(file_path, "rb") as f:
                    vector_store = pickle.load(f)
                logger.info("Vector store loaded successfully")
                    
                # Create BM25 retriever from vector store documents
                logger.debug("Recreating BM25 retriever")
                documents = vector_store.similarity_search("", k=1000)  # Get all documents
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 5
                logger.info("BM25 retriever recreated successfully")
                
                # Create ensemble retriever
                logger.debug("Recreating ensemble retriever")
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[
                        vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                        bm25_retriever
                    ],
                    weights=[0.7, 0.3]
                )
                logger.info("Ensemble retriever recreated successfully")
                
                # Store retrievers
                self.vector_stores[project_id] = vector_store
                self.bm25_retrievers[project_id] = bm25_retriever
                self.ensemble_retrievers[project_id] = ensemble_retriever
                
            else:
                logger.warning(f"No vector store file found at: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            logger.warning("Continuing without loaded vector store")