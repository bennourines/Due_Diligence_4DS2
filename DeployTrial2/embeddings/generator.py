# filepath: DeployTrial2/embeddings/generator.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import logging
from core.config import settings
import asyncio # For running sync code in async context

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles generation of embeddings for document chunks using HuggingFace."""

    def __init__(self):
        try:
            logger.info(f"Initializing HuggingFaceEmbeddings with model: {settings.EMBEDDING_MODEL_NAME} on device: {settings.EMBEDDING_DEVICE}")
            # Ensure device setting is appropriate ('cpu', 'cuda', 'mps', etc.)
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': settings.EMBEDDING_DEVICE},
                # Normalization is often recommended for cosine similarity search
                encode_kwargs={'normalize_embeddings': True}
            )
            # Perform a dummy embedding to ensure model loads correctly at startup
            _ = self.embedding_model.embed_query("test initialization")
            logger.info("Embedding model loaded and tested successfully.")
        except Exception as e:
            logger.error(f"Failed to load or test embedding model '{settings.EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
            # Depending on requirements, might want to raise a critical error to stop startup
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

    async def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generates embeddings for a list of LangChain Document objects asynchronously.
        Runs the synchronous embedding function in a thread pool executor.
        Returns a list of embedding vectors.
        """
        if not documents:
            logger.warning("embed_documents called with empty list.")
            return []

        texts = [doc.page_content for doc in documents]
        num_docs = len(texts)
        logger.info(f"Generating embeddings for {num_docs} document chunks...")

        try:
            # HuggingFaceEmbeddings embed_documents is sync. Run it in the default executor.
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, # Use default executor (ThreadPoolExecutor)
                self.embedding_model.embed_documents,
                texts
            )
            # embeddings = self.embedding_model.embed_documents(texts) # Sync call

            if not embeddings or len(embeddings) != num_docs:
                 logger.error(f"Embedding generation returned unexpected result: expected {num_docs}, got {len(embeddings) if embeddings else 0}")
                 raise RuntimeError("Embedding generation failed or returned incorrect number of embeddings.")

            logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Error during document embedding generation: {e}", exc_info=True)
            raise RuntimeError("Failed during document embedding") from e # Propagate error

    async def embed_query(self, query: str) -> List[float]:
        """
        Generates embedding for a single query string asynchronously.
        Runs the synchronous embedding function in a thread pool executor.
        """
        if not query:
             logger.warning("embed_query called with empty query.")
             # Return a zero vector or raise error depending on requirements
             # Example: return [0.0] * self.embedding_model.client.get_sentence_embedding_dimension()
             raise ValueError("Cannot embed empty query.")

        logger.debug(f"Generating embedding for query: '{query[:50]}...'")
        try:
            # embed_query is sync. Run it in the default executor.
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                None, # Use default executor
                self.embedding_model.embed_query,
                query
            )
            # embedding = self.embedding_model.embed_query(query) # Sync call

            if not embedding:
                 logger.error("Query embedding generation returned empty result.")
                 raise RuntimeError("Query embedding generation failed.")

            logger.debug("Query embedding generated.")
            return embedding
        except Exception as e:
            logger.error(f"Error during query embedding: {e}", exc_info=True)
            raise RuntimeError("Failed during query embedding") from e # Propagate error

    def get_embedding_dimension(self) -> int:
        """Returns the dimension of the embeddings produced by the model."""
        try:
            # Access dimension if available directly (might depend on langchain version/model)
            # Or embed a dummy query and check length
            if hasattr(self.embedding_model, 'client') and hasattr(self.embedding_model.client, 'get_sentence_embedding_dimension'):
                 return self.embedding_model.client.get_sentence_embedding_dimension()
            else:
                 # Fallback: embed dummy query
                 dummy_embedding = self.embedding_model.embed_query("dimension_check")
                 return len(dummy_embedding)
        except Exception as e:
             logger.error(f"Could not determine embedding dimension automatically: {e}")
             # Return a default or raise error - critical for FAISS index creation
             raise RuntimeError("Failed to determine embedding dimension") from e
