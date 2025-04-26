# filepath: DeployTrial2/retrieval/vector_store.py
import faiss
import numpy as np
import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
import asyncio # For running sync FAISS operations in executor

from core.config import settings
from embeddings.generator import EmbeddingGenerator # Needs an instance passed

logger = logging.getLogger(__name__)

class FaissVectorStoreManager:
    """
    Manages FAISS vector stores and metadata for different projects,
    saving index and metadata as separate files (.faiss, .json) per user/project.
    Handles asynchronous loading/saving and searching using asyncio executor.
    """
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.base_path = settings.VECTOR_STORE_BASE_PATH
        self.embedding_generator = embedding_generator
        # Determine embedding dimension dynamically at init
        try:
            # Use the method from EmbeddingGenerator
            self.embedding_dim = self.embedding_generator.get_embedding_dimension()
            logger.info(f"FAISS Manager initialized with embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.critical(f"CRITICAL: Could not determine embedding dimension for FAISS Manager: {e}", exc_info=True)
            # This is critical, application should likely not start
            raise SystemExit(f"Failed to get embedding dimension: {e}")


    def _get_project_paths(self, user_id: str, project_id: str) -> Tuple[str, str, str]:
        """Gets the directory, index path, and metadata path for a project."""
        # Ensure user_id and project_id are safe for path creation (basic example)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        safe_project_id = "".join(c for c in project_id if c.isalnum() or c in ('-', '_'))

        project_dir = os.path.join(self.base_path, safe_user_id, safe_project_id)
        index_path = os.path.join(project_dir, "index.faiss")
        metadata_path = os.path.join(project_dir, "metadata.json")
        return project_dir, index_path, metadata_path

    async def _load_index_and_metadata(self, user_id: str, project_id: str) -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
        """Loads FAISS index and metadata list from files asynchronously."""
        _, index_path, metadata_path = self._get_project_paths(user_id, project_id)
        index = None
        metadata = []
        loop = asyncio.get_running_loop()

        if os.path.exists(index_path):
            try:
                # Run synchronous faiss.read_index in executor
                index = await loop.run_in_executor(None, faiss.read_index, index_path)
                logger.info(f"Loaded existing FAISS index from {index_path} with {index.ntotal} vectors.")
                # Dimension check
                if index.d != self.embedding_dim:
                     logger.error(f"Dimension mismatch! Index ({index.d}) vs Model ({self.embedding_dim}) for {project_id}")
                     raise ValueError(f"Dimension mismatch loading index for project {project_id}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {index_path}: {e}", exc_info=True)
                raise # Propagate error

        if os.path.exists(metadata_path):
            try:
                # Reading JSON is I/O bound, can run in executor too for large files
                def read_json_sync():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                metadata = await loop.run_in_executor(None, read_json_sync)

                if not isinstance(metadata, list):
                     logger.error(f"Metadata file {metadata_path} is not a JSON list.")
                     raise ValueError(f"Invalid metadata format for project {project_id}")
                logger.info(f"Loaded {len(metadata)} metadata entries from {metadata_path}.")
            except json.JSONDecodeError as json_err:
                 logger.error(f"Error decoding JSON metadata file {metadata_path}: {json_err}")
                 raise ValueError(f"Invalid JSON format in metadata file: {metadata_path}") from json_err
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_path}: {e}", exc_info=True)
                raise # Propagate error

        # Consistency check
        if index is not None and index.ntotal != len(metadata):
            logger.warning(f"Index size ({index.ntotal}) and metadata count ({len(metadata)}) mismatch for project {project_id}. Re-indexing might be required.")
            # Decide handling: raise error, log warning, attempt repair?

        return index, metadata

    async def _save_index_and_metadata(self, user_id: str, project_id: str, index: faiss.Index, metadata: List[Dict[str, Any]]):
        """Saves FAISS index and metadata list to files asynchronously."""
        project_dir, index_path, metadata_path = self._get_project_paths(user_id, project_id)
        loop = asyncio.get_running_loop()

        # Ensure directory exists (sync os call, usually fast)
        os.makedirs(project_dir, exist_ok=True)

        try:
            # Run synchronous faiss.write_index in executor
            await loop.run_in_executor(None, faiss.write_index, index, index_path)
            logger.info(f"Saved FAISS index to {index_path} with {index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index to {index_path}: {e}", exc_info=True)
            raise

        try:
            # Run synchronous JSON writing in executor
            def write_json_sync():
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2) # Use indent=2 for smaller file size
            await loop.run_in_executor(None, write_json_sync)
            logger.info(f"Saved {len(metadata)} metadata entries to {metadata_path}.")
        except Exception as e:
            logger.error(f"Failed to save metadata to {metadata_path}: {e}", exc_info=True)
            raise

    async def add_documents(self, user_id: str, project_id: str, documents: List[Document]):
        """Adds document chunks, embeddings, and metadata to the project's store asynchronously."""
        if not documents:
            logger.warning(f"No documents provided to add for project {project_id}.")
            return

        logger.info(f"Adding {len(documents)} document chunks to project {project_id} for user {user_id}.")
        loop = asyncio.get_running_loop()

        # 1. Load existing index and metadata (async)
        index, metadata_list = await self._load_index_and_metadata(user_id, project_id)

        # 2. Generate embeddings for new documents (async)
        new_embeddings_list = await self.embedding_generator.embed_documents(documents)
        if not new_embeddings_list or len(new_embeddings_list) != len(documents):
             logger.error("Embedding generation failed or returned incorrect number of embeddings.")
             raise RuntimeError("Embedding generation failed.")
        # Ensure numpy array with correct dtype for FAISS
        new_embeddings = np.array(new_embeddings_list).astype('float32')

        # 3. Prepare metadata for new documents (sync, fast)
        new_metadata = []
        for doc in documents:
            entry = {
                # Store only essential info needed for reconstruction/display
                "chunk_text": doc.page_content,
                "source_file": doc.metadata.get("source", "Unknown"),
                "start_index": doc.metadata.get("start_index"),
                # Add other relevant metadata if needed, keep it minimal
            }
            new_metadata.append(entry)

        # 4. Add to FAISS index (sync FAISS calls in executor)
        def add_to_index_sync():
            nonlocal index # Allow modification of outer scope variable
            if index is None:
                # Create new index
                logger.info(f"Creating new FAISS index for project {project_id} with dimension {self.embedding_dim}.")
                # Use IndexFlatL2 for simplicity and exact search. Consider IVF for large scale.
                index = faiss.IndexFlatL2(self.embedding_dim)
                # If using IVF:
                # nlist = max(1, min(100, len(new_embeddings) // 40)) # Example heuristic
                # quantizer = faiss.IndexFlatL2(self.embedding_dim)
                # index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
                # if not index.is_trained and len(new_embeddings) > nlist: # Need enough data to train
                #     index.train(new_embeddings)

            # Add new vectors to the index
            index.add(new_embeddings)
            return index # Return the modified or new index

        index = await loop.run_in_executor(None, add_to_index_sync)
        logger.info(f"Added {len(new_embeddings)} vectors to FAISS index. Total vectors: {index.ntotal}")

        # 5. Append new metadata (sync, fast)
        metadata_list.extend(new_metadata)

        # 6. Save updated index and metadata (async)
        await self._save_index_and_metadata(user_id, project_id, index, metadata_list)
        logger.info(f"Successfully updated vector store for project {project_id}.")


    async def search(self, user_id: str, project_id: str, query: str, top_k: int = 5) -> List[Document]:
        """Searches the project's vector store asynchronously."""
        logger.info(f"Searching project {project_id} for query: '{query[:50]}...' (top_k={top_k})")
        loop = asyncio.get_running_loop()

        # 1. Load index and metadata (async)
        index, metadata_list = await self._load_index_and_metadata(user_id, project_id)

        if index is None or index.ntotal == 0:
            logger.warning(f"No index or empty index found for project {project_id}. Cannot perform search.")
            return []

        # 2. Embed the query (async)
        query_embedding = await self.embedding_generator.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')

        # Dimension check before search
        if index.d != query_vector.shape[1]:
             logger.error(f"Dimension mismatch during search! Index ({index.d}) vs Query ({query_vector.shape[1]}) for {project_id}")
             return [] # Or raise

        # 3. Perform FAISS search (sync FAISS call in executor)
        def search_sync():
            try:
                # Ensure k is not greater than the number of vectors in the index
                actual_k = min(top_k, index.ntotal)
                distances, indices = index.search(query_vector, k=actual_k)
                logger.debug(f"FAISS search returned indices: {indices[0]}, distances: {distances[0]}")
                return distances, indices
            except Exception as e:
                logger.error(f"FAISS search failed for project {project_id}: {e}", exc_info=True)
                return None, None # Indicate failure

        distances, indices = await loop.run_in_executor(None, search_sync)

        if distances is None or indices is None:
             return [] # Search failed

        # 4. Retrieve and format results (sync, fast)
        results = []
        retrieved_indices = indices[0] # Get indices for the first query vector

        for i, idx in enumerate(retrieved_indices):
            if idx == -1: # FAISS returns -1 if fewer than k results are found or for empty results
                continue
            # Check bounds carefully
            if idx < 0 or idx >= len(metadata_list):
                logger.warning(f"Invalid index {idx} returned by FAISS search for project {project_id}. Metadata length: {len(metadata_list)}. Skipping.")
                continue

            try:
                metadata_entry = metadata_list[idx]
                score = distances[0][i] # Corresponding distance/score

                # Reconstruct LangChain Document
                content = metadata_entry.get("chunk_text", "") # Get text from metadata
                if not content:
                     logger.warning(f"Empty 'chunk_text' found for index {idx} in project {project_id}. Skipping.")
                     continue

                doc_metadata = {
                    "source": metadata_entry.get("source_file", "Unknown"),
                    "retrieval_score": float(score), # Store the score/distance
                    "start_index": metadata_entry.get("start_index"),
                    # Add any other metadata stored
                }
                results.append(Document(page_content=content, metadata=doc_metadata))
            except Exception as e:
                 logger.error(f"Error processing metadata for index {idx} in project {project_id}: {e}", exc_info=True)
                 continue # Skip problematic entries

        logger.info(f"Retrieved {len(results)} relevant document chunks for project {project_id}.")
        return results

    async def delete_project_store(self, user_id: str, project_id: str):
        """Deletes the vector store files for a given project asynchronously."""
        project_dir, index_path, metadata_path = self._get_project_paths(user_id, project_id)
        logger.warning(f"Attempting to delete vector store for project {project_id} at {project_dir}")
        loop = asyncio.get_running_loop()

        async def delete_files_sync():
            deleted_count = 0
            try:
                if os.path.exists(index_path):
                    os.remove(index_path)
                    logger.info(f"Deleted index file: {index_path}")
                    deleted_count += 1
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    logger.info(f"Deleted metadata file: {metadata_path}")
                    deleted_count += 1
                # Attempt to remove the directory if empty
                if os.path.exists(project_dir):
                     try:
                         os.rmdir(project_dir)
                         logger.info(f"Removed empty project directory: {project_dir}")
                     except OSError:
                         logger.debug(f"Project directory not empty or could not be removed: {project_dir}")
                # If the user directory becomes empty, optionally remove it too
                user_dir = os.path.dirname(project_dir)
                if os.path.exists(user_dir):
                     try:
                         os.rmdir(user_dir)
                         logger.info(f"Removed empty user directory: {user_dir}")
                     except OSError:
                         logger.debug(f"User directory not empty or could not be removed: {user_dir}")
                return deleted_count > 0
            except OSError as e:
                logger.error(f"Error deleting vector store files for project {project_id}: {e}", exc_info=True)
                return False # Indicate failure

        deleted = await loop.run_in_executor(None, delete_files_sync)
        if not deleted:
             logger.warning(f"Vector store files might not have been fully deleted for project {project_id}")
        # Decide if this should raise an error or just log
