# search_engine.py - Enhanced with external FAISS DB integration
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Union
import os
import pickle
import faiss
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedHybridSearchEngine:
    """Enhanced Hybrid search engine with external FAISS DB integration"""
    
    def __init__(self, 
                 external_faiss_path: Optional[str] = None,
                 external_faiss_index_mapping_path: Optional[str] = None):
        """
        Initialize the enhanced hybrid search engine
        
        Args:
            external_faiss_path: Path to external FAISS index (.faiss file)
            external_faiss_index_mapping_path: Path to external FAISS index mapping (.pkl file)
        """
        # Initialize HuggingFace embeddings (sentence-transformers)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Maps project IDs to search engines
        self.vector_stores = {}
        self.bm25_retrievers = {}
        self.ensemble_retrievers = {}
        
        # External FAISS index (user's existing database)
        self.external_faiss_index = None
        self.external_faiss_mapping = None
        
        # Load external FAISS index if provided
        if external_faiss_path and os.path.exists(external_faiss_path):
            self._load_external_faiss(external_faiss_path, external_faiss_index_mapping_path)
        
        # Create storage directory
        os.makedirs("vector_stores", exist_ok=True)
    
    def _load_external_faiss(self, index_path: str, mapping_path: Optional[str] = None):
        """
        Load external FAISS index and mapping
        
        Args:
            index_path: Path to the FAISS index file
            mapping_path: Path to the index mapping file (document mapping)
        """
        try:
            logger.info(f"Loading external FAISS index from {index_path}")
            self.external_faiss_index = faiss.read_index(index_path)
            
            # Load document mapping if available
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.external_faiss_mapping = pickle.load(f)
                logger.info(f"Loaded external FAISS mapping with {len(self.external_faiss_mapping)} documents")
            else:
                logger.warning("No mapping file provided for external FAISS index")
        except Exception as e:
            logger.error(f"Error loading external FAISS index: {str(e)}")
            self.external_faiss_index = None
            self.external_faiss_mapping = None
    
    def add_documents(self, documents: List[Document], project_id: str):
        """
        Add documents to vector store and BM25 retriever
        
        Args:
            documents: List of LangChain Document objects
            project_id: Project identifier
        """
        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                bm25_retriever
            ],
            weights=[0.7, 0.3]
        )
        
        # Store retrievers
        self.vector_stores[project_id] = vector_store
        self.bm25_retrievers[project_id] = bm25_retriever
        self.ensemble_retrievers[project_id] = ensemble_retriever
        
        # Save vector store for persistence
        self._save_vector_store(vector_store, project_id)
        
        logger.info(f"Added {len(documents)} documents to project {project_id}")
    
    def hybrid_search(self, query: str, project_id: str, k: int = 5, 
                     include_external: bool = True) -> List[Document]:
        """
        Perform hybrid search using ensemble retriever and external FAISS
        
        Args:
            query: Query string
            project_id: Project identifier
            k: Number of results to return
            include_external: Whether to include results from external FAISS
            
        Returns:
            List of relevant Document objects
        """
        results = []
        
        # Check if project exists or needs to be loaded
        if project_id not in self.ensemble_retrievers:
            self._load_vector_store(project_id)
            
        # Search in project-specific database
        if project_id in self.ensemble_retrievers:
            # Retrieve documents with ensemble retriever
            retriever = self.ensemble_retrievers[project_id]
            project_results = retriever.get_relevant_documents(query)
            
            # Add source to metadata
            for doc in project_results:
                if 'source_type' not in doc.metadata:
                    doc.metadata['source_type'] = 'uploaded_document'
                
            results.extend(project_results)
        
        # Search in external FAISS if available and requested
        if include_external and self.external_faiss_index is not None:
            external_results = self._search_external_faiss(query, k=k)
            if external_results:
                # Add source to metadata
                for doc in external_results:
                    doc.metadata['source_type'] = 'external_database'
                
                results.extend(external_results)
        
        # Deduplicate and sort results by relevance (if we have a combined approach)
        if include_external and self.external_faiss_index is not None and project_id in self.ensemble_retrievers:
            # This is a simple approach - ideally would re-rank based on combined score
            seen_content = set()
            unique_results = []
            
            for doc in results:
                # Create a hash of the content to deduplicate
                content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(doc)
            
            results = unique_results[:k]  # Limit to k results
        
        return results
    
    def _search_external_faiss(self, query: str, k: int = 5) -> List[Document]:
        """
        Search in external FAISS index
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if self.external_faiss_index is None:
            return []
        
        try:
            # Generate embeddings for the query
            query_embedding = self.embeddings.embed_query(query)
            query_embedding_np = np.array([query_embedding], dtype=np.float32)
            
            # Search in the FAISS index
            distances, indices = self.external_faiss_index.search(query_embedding_np, k)
            
            # Convert results to Document objects
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # -1 indicates no match
                    if self.external_faiss_mapping and idx < len(self.external_faiss_mapping):
                        doc_data = self.external_faiss_mapping[idx]
                        
                        # Check if doc_data is already a Document object
                        if isinstance(doc_data, Document):
                            doc = doc_data
                        elif isinstance(doc_data, dict) and 'content' in doc_data:
                            # Create Document from dict
                            doc = Document(
                                page_content=doc_data['content'],
                                metadata=doc_data.get('metadata', {'source': 'external_faiss'})
                            )
                        else:
                            # Just use the index as a placeholder
                            doc = Document(
                                page_content=f"Content from external FAISS index at position {idx}",
                                metadata={'source': 'external_faiss', 'index': int(idx), 'score': float(distances[0][i])}
                            )
                        
                        results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error searching external FAISS: {str(e)}")
            return []
    
    def _save_vector_store(self, vector_store, project_id: str):
        """Save vector store to disk"""
        try:
            with open(f"vector_stores/{project_id}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            logger.info(f"Saved vector store for project {project_id}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def _load_vector_store(self, project_id: str):
        """Load vector store from disk"""
        try:
            file_path = f"vector_stores/{project_id}.pkl"
            if os.path.exists(file_path):
                logger.info(f"Loading vector store for project {project_id}")
                with open(file_path, "rb") as f:
                    vector_store = pickle.load(f)
                    
                # Create BM25 retriever from vector store documents
                documents = vector_store.similarity_search("", k=1000)  # Get all documents
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 5
                
                # Create ensemble retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[
                        vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                        bm25_retriever
                    ],
                    weights=[0.7, 0.3]
                )
                
                # Store retrievers
                self.vector_stores[project_id] = vector_store
                self.bm25_retrievers[project_id] = bm25_retriever
                self.ensemble_retrievers[project_id] = ensemble_retriever
                
                logger.info(f"Loaded vector store for project {project_id}")
            else:
                logger.warning(f"No vector store found for project {project_id}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")