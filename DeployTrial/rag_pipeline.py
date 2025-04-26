# rag_pipeline.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import os
import logging
import pickle
import faiss
import json 
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document # Import Document

# --- Constants ---
FAISS_INDEX_PATH = "../faiss_index_download/index.faiss" # Path to your FAISS index file
FAISS_METADATA_PATH = "../faiss_index_download/index_metadata.json" # Path to your metadata file (adjust if different)
EMBEDDING_MODEL_NAME = 'sentence-transformers/multi-qa-mpnet-base-dot-v1' # Ensure this matches the model used for indexing
DEFAULT_TOP_K_FAISS = 5 # Number of results to retrieve from FAISS


# Set up logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline using LangChain and OpenRouter"""
    
    def __init__(self):
        logger.info("Initializing RAG Pipeline")
        try:
            # Initialize LLM using OpenRouter
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key")
            logger.debug("Setting up ChatOpenAI with OpenRouter")
            
            self.llm = ChatOpenAI(
                model="meta-llama/llama-4-maverick:free",  # Can be adjusted to other models
                temperature=0.2,
                max_tokens=1000,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=openrouter_api_key,
                
            )
            logger.info("LLM initialized successfully")

            # --- Load FAISS Index and related components ---
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded.")

            logger.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
            if not os.path.exists(FAISS_INDEX_PATH):
                raise FileNotFoundError(f"FAISS index file not found at {FAISS_INDEX_PATH}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"FAISS index loaded with {self.faiss_index.ntotal} vectors.")

           # --- Correctly load metadata from .json file ---
            logger.info(f"Loading FAISS metadata from JSON file: {FAISS_METADATA_PATH}")
            if not os.path.exists(FAISS_METADATA_PATH):
                raise FileNotFoundError(f"FAISS metadata JSON file not found at {FAISS_METADATA_PATH}")
            try:
                # Use 'r' for text mode and json.load for .json files
                with open(FAISS_METADATA_PATH, 'r', encoding='utf-8') as f_meta: 
                    self.faiss_metadata = json.load(f_meta) 
            except json.JSONDecodeError as json_err:
                 logger.error(f"Error decoding JSON metadata file {FAISS_METADATA_PATH}: {json_err}")
                 raise ValueError(f"Invalid JSON format in metadata file: {FAISS_METADATA_PATH}") from json_err
            except Exception as meta_err:
                 logger.error(f"Error loading metadata file {FAISS_METADATA_PATH}: {meta_err}")
                 raise meta_err

            # Ensure metadata is a list or dict (adjust based on your actual structure)
            if not isinstance(self.faiss_metadata, (list, dict)):
                 raise TypeError(f"Loaded metadata from {FAISS_METADATA_PATH} is not a list or dictionary.")
            logger.info(f"FAISS metadata loaded for {len(self.faiss_metadata)} entries.")
            # --- End FAISS Loading ---
            
            # Define QA prompt template
            logger.debug("Setting up QA prompt template")
            qa_template = """You are a helpful assistant specialized in cryptocurrency due diligence, analyzing blockchain projects, and identifying potential risks.

CONTEXT:
{context}

USER QUERY:
{question}

Provide a detailed and well-reasoned answer based on the given context. If the context doesn't contain enough information to answer confidently, mention that clearly. Include references to specific documents or data points from the context to support your analysis.
"""
            
            qa_prompt = PromptTemplate(
                template=qa_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            logger.debug("Creating QA chain")
            self.qa_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff",# # Can be adjusted to other chain types, but chose "stuff" for simplicity
                prompt=qa_prompt
            )
            logger.info("QA chain created successfully")
            
        except FileNotFoundError as fnf_error:
             logger.error(f"Initialization failed: {fnf_error}")
             raise
        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline: {str(e)}", exc_info=True)
            raise
    

    def generate_response(self, query: str) -> str:
        """Generate response using RAG pipeline, retrieving context from FAISS."""
        try:
            logger.info("Generating response using FAISS context")
            logger.debug(f"Query: {query}")

            # 1. Retrieve context from FAISS
            logger.debug(f"Retrieving top {DEFAULT_TOP_K_FAISS} documents from FAISS")
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
            
            # Add dimension check
            query_dim = query_embedding.shape[1]
            index_dim = self.faiss_index.d
            if query_dim != index_dim:
                logger.error(f"Dimension mismatch: Query embedding dimension ({query_dim}) != FAISS index dimension ({index_dim}). Ensure the index was created with the '{EMBEDDING_MODEL_NAME}' model.")
                # Optionally raise an error or return a specific message
                # raise ValueError(f"Dimension mismatch: Query({query_dim}) != Index({index_dim})") 
                return f"Error: Embedding dimension mismatch (Query: {query_dim}, Index: {index_dim}). Cannot perform search."

            distances, indices = self.faiss_index.search(np.array(query_embedding).astype('float32'), DEFAULT_TOP_K_FAISS)

            faiss_contexts = []
            retrieved_indices = indices[0]
            valid_indices = [idx for idx in retrieved_indices if idx != -1 and idx < len(self.faiss_metadata)]

            if not valid_indices:
                 logger.warning("FAISS search returned no valid documents for the query.")
                 return "I couldn't find any relevant information in the knowledge base to answer your query."

            for i, idx in enumerate(valid_indices):
                 metadata_entry = self.faiss_metadata[idx]
                 if isinstance(metadata_entry, dict):
                     content = metadata_entry.get('chunk_text', 'Content not found')
                     source = metadata_entry.get('source_file', 'Unknown FAISS Source')
                     doc_metadata = {'source': source, 'retrieval_score': float(distances[0][i])}
                     faiss_contexts.append(Document(page_content=content, metadata=doc_metadata))
                 else:
                      # Handle case where metadata might just be text strings
                      content = str(metadata_entry) # Assume it's the text
                      source = 'Unknown FAISS Source'
                      doc_metadata = {'source': source, 'retrieval_score': float(distances[0][i])}
                      faiss_contexts.append(Document(page_content=content, metadata=doc_metadata))


            logger.info(f"Retrieved {len(faiss_contexts)} documents from FAISS.")
            logger.debug(f"Number of context documents from FAISS: {len(faiss_contexts)}")

            # 2. Generate response using the retrieved context
            logger.debug("Invoking QA chain with FAISS context")
            response_dict = self.qa_chain.invoke({
                "input_documents": faiss_contexts, # Use documents retrieved from FAISS
                "question": query
            })
            logger.info("Response generated successfully")

            # Extract the actual answer string
            answer = response_dict.get('output_text', '')
            if not answer:
                 logger.warning("QA chain returned an empty 'output_text'.")
                 answer = "I processed the relevant information, but couldn't formulate a specific answer based on the context."

            logger.debug(f"Response length: {len(answer)} characters")

            return answer # Return only the string answer

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return f"Sorry, an error occurred while processing your request: {str(e)}"

