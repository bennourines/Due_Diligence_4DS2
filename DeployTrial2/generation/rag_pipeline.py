# filepath: DeployTrial2/generation/rag_pipeline.py
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting
from typing import List, Dict, Any
import logging
from fastapi import HTTPException, status # Import HTTPException and status codes from fastapi

from retrieval.vector_store import FaissVectorStoreManager
from generation.llm_clients import get_default_llm # Import the default client factory

logger = logging.getLogger(__name__)

# Define the QA prompt template (could be loaded from config or a file)
# Enhanced prompt for better grounding and source attribution
QA_TEMPLATE = """You are an AI assistant specialized in analyzing cryptocurrency project documents for due diligence. Your task is to answer the user's query based *strictly* on the provided context documents.

**CONTEXT DOCUMENTS:**
---
{context}
---

**USER QUERY:**
{question}

**Instructions:**
1.  Carefully read the CONTEXT DOCUMENTS provided above.
2.  Answer the USER QUERY using *only* information found within the CONTEXT DOCUMENTS.
3.  If the context does not contain the answer, explicitly state "The provided documents do not contain information to answer this query." Do *not* use any external knowledge or make assumptions.
4.  Structure your answer clearly and concisely.
5.  If possible, mention the source document(s) (e.g., "[Source: whitepaper.pdf]") where the information was found, based on the metadata provided with the context.

**ANSWER:**
"""

class RAGPipeline:
    """Handles the RAG process: retrieval, context formatting, and generation."""

    def __init__(self, vector_store_manager: FaissVectorStoreManager, llm_client: BaseChatModel = None):
        """
        Initializes the RAG Pipeline.

        Args:
            vector_store_manager: An instance of FaissVectorStoreManager (or compatible).
            llm_client: An optional pre-initialized LLM client. If None, uses get_default_llm().
        """
        self.vector_store_manager = vector_store_manager
        self.llm = llm_client or get_default_llm() # Use provided or get default

        qa_prompt = PromptTemplate(
            template=QA_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Using "stuff" chain type - simplest, puts all context into one prompt.
        # Ensure your context size + query + prompt fit within the LLM's context window.
        # Consider "map_reduce" or "refine" for very large contexts.
        try:
            self.qa_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff", # Most common, but check context limits
                prompt=qa_prompt,
                verbose=False # Set to True for detailed LangChain debugging
            )
            logger.info("RAGPipeline initialized with 'stuff' QA chain.")
        except Exception as e:
             logger.error(f"Failed to load QA chain: {e}", exc_info=True)
             raise RuntimeError(f"Failed to initialize RAG QA chain: {e}") from e

    def _format_context_for_prompt(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a string for the LLM prompt."""
        if not docs:
            return "No relevant context documents found."

        formatted_context = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            content_preview = doc.page_content.replace('\n', ' ').strip()[:300]  # Limit preview length
            formatted_context.append(f"Source: {source}\nContent: {content_preview}...\n---")
            
        return "\n".join(formatted_context)


    async def generate_response(self, user_id: str, project_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generates a response based on a query and project-specific context.
        Performs retrieval, formats context, invokes LLM, and returns results.

        Args:
            user_id: ID of the user performing the query.
            project_id: ID of the project context.
            query: The user's question.
            top_k: The maximum number of document chunks to retrieve.

        Returns:
            A dictionary containing:
                - "answer": The generated response string.
                - "sources": A list of unique source filenames retrieved.

        Raises:
            HTTPException: If retrieval or generation fails significantly.
        """
        logger.info(f"Generating RAG response for project {project_id}, query: '{query[:50]}...'")

        # 1. Retrieve relevant documents from the project's vector store
        try:
            retrieved_docs: List[Document] = await self.vector_store_manager.search(
                user_id=user_id,
                project_id=project_id,
                query=query,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Error during document retrieval for project {project_id}: {e}", exc_info=True)
            # Raise HTTPException to be caught by the API layer
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve context documents from the vector store."
            )

        retrieved_sources = []
        if not retrieved_docs:
            logger.warning(f"No relevant documents found for query in project {project_id}.")
            # Pass specific message to LLM or handle directly
            # Return early with a clear message when no documents are found
            return {
                "answer": "I couldn't find any relevant information in the project documents to answer your query.",
                "sources": []
            }
        else:
            # Format the documents for the LLM prompt and store the result
            formatted_context = self._format_context_for_prompt(retrieved_docs)
            # Extract unique source filenames
            retrieved_sources = sorted(list(set(
                doc.metadata.get("source", "Unknown") for doc in retrieved_docs if doc.metadata.get("source")
            )))
            logger.info(f"Retrieved {len(retrieved_docs)} chunks from sources: {retrieved_sources}")

        # 2. Invoke the QA chain with formatted context and query
        logger.debug(f"Invoking QA chain for project {project_id}...")
        response_payload = {
            "context": formatted_context,
            "input_documents": retrieved_docs,
            "question": query
        }

        try:
            # Try async invocation first, fall back to sync if needed
            chain_result = None
            try:
                chain_result = await self.qa_chain.ainvoke(response_payload)
            except AttributeError:
                logger.warning("Async chain invocation not supported, falling back to sync")
                chain_result = self.qa_chain(response_payload)

            if not chain_result:
                raise ValueError("Chain invocation failed to produce a result")
            
            # Extract and validate the answer
            answer = chain_result.get('output_text', '').strip()
            if not answer:
                logger.warning(f"QA chain returned empty 'output_text' for project {project_id}.")
                answer = "The language model could not generate an answer based on the provided context."

            logger.info(f"Successfully generated response for project {project_id}.")
            return {"answer": answer, "sources": retrieved_sources}

        except Exception as e:
            logger.error(f"Error during LLM generation for project {project_id}: {e}", exc_info=True)
            # Handle LLM errors (e.g., API errors, rate limits, context window exceeded)
            # Check for specific error types if possible
            detail = f"Language model service error: {str(e)}"
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE  # Default to 503

            # Example: Check for context length errors (may vary by LLM/library)
            if "context length" in str(e).lower():
                detail = "The document context is too large for the language model. Try a shorter query or fewer documents."
                status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

            raise HTTPException(status_code=status_code, detail=detail)
