# filepath: DeployTrial2/processing/processor.py
from langchain_community.document_loaders import (
    UnstructuredFileLoader, PyPDFLoader, Docx2txtLoader
)
# Add other loaders as needed (e.g., UnstructuredExcelLoader requires extra deps)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
import logging

from core.config import settings # Import settings if needed for config

logger = logging.getLogger(__name__)

# Consider making chunk size/overlap configurable via settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

class DocumentProcessor:
    """Loads, splits, and prepares documents for embedding."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        # Use settings if defined, otherwise use defaults
        # self.chunk_size = getattr(settings, 'DOC_CHUNK_SIZE', chunk_size)
        # self.chunk_overlap = getattr(settings, 'DOC_CHUNK_OVERLAP', chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True, # Useful for context referencing
        )
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    async def load_and_split(self, file_path: str) -> List[Document]:
        """
        Loads a document from a file path and splits it into chunks.
        Handles basic document types (PDF, DOCX, TXT, MD).
        Returns a list of LangChain Document objects.
        """
        if not os.path.exists(file_path):
             logger.error(f"File not found for processing: {file_path}")
             raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing file: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        loader = None
        documents = []

        try:
            logger.debug(f"Selecting loader for file extension: {file_extension}")
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension in [".docx", ".doc"]:
                # Ensure docx2txt is installed: pip install docx2txt
                loader = Docx2txtLoader(file_path)
            elif file_extension in [".txt", ".md"]: # Add other text-based formats
                 # UnstructuredFileLoader can handle various text formats
                 # Specify encoding if needed: loader = UnstructuredFileLoader(file_path, encoding="utf-8")
                loader = UnstructuredFileLoader(file_path, mode="single", strategy="fast") # Use single mode for simple text
            # Add other loaders like UnstructuredExcelLoader if needed and deps installed
            # elif file_extension in [".xlsx", ".xls"]:
            #     # Requires: pip install "unstructured[xlsx]" openpyxl
            #     loader = UnstructuredExcelLoader(file_path, mode="single")
            else:
                logger.warning(f"Unsupported file extension '{file_extension}' for {file_path}. Trying UnstructuredFileLoader as fallback.")
                # Fallback for potentially supported types by Unstructured
                # Requires: pip install "unstructured[local-inference]" (or specific format deps)
                try:
                    loader = UnstructuredFileLoader(file_path, mode="single", strategy="auto", errors="ignore") # Ignore errors for unknown types
                except ImportError:
                     logger.error(f"UnstructuredFileLoader fallback failed for {file_path}. Ensure 'unstructured' dependencies are installed for this file type.")
                     return [] # Return empty if fallback fails due to missing deps

            if loader:
                # Langchain loaders often have sync 'load' and async 'aload'
                # Use sync load for simplicity here, consider async if loader supports & performance needed
                # documents = await loader.aload() # Check loader documentation for async support
                documents = loader.load() # Sync load
                logger.info(f"Loaded {len(documents)} initial document part(s) from {file_path}")

                if not documents:
                     logger.warning(f"Loader returned no documents for {file_path}")
                     return []

                # Add source filename to metadata *before* splitting
                base_filename = os.path.basename(file_path)
                for doc in documents:
                    # Ensure metadata exists
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["source"] = base_filename
                    # Clean up potential large metadata from unstructured if needed
                    # doc.metadata = {"source": base_filename} # Example: Keep only source

                # Split the loaded documents
                chunks = self.text_splitter.split_documents(documents)
                logger.info(f"Split '{base_filename}' into {len(chunks)} chunks.")

                # Optional: Log chunk details for debugging
                # for i, chunk in enumerate(chunks):
                #     logger.debug(f"Chunk {i+1} ({chunk.metadata.get('source')}): {len(chunk.page_content)} chars, start_index: {chunk.metadata.get('start_index')}")

                return chunks
            else:
                # This case might be redundant if fallback loader is always attempted
                logger.error(f"No suitable loader determined for file: {file_path}")
                return []

        except FileNotFoundError:
             # Already logged above, re-raise specific error
             raise
        except ImportError as ie:
             logger.error(f"Missing dependency for processing {file_path} (extension: {file_extension}): {ie}")
             raise RuntimeError(f"Missing dependency for file type {file_extension}: {ie}") from ie
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
            # Re-raise to be caught by the caller (e.g., background task handler)
            raise RuntimeError(f"Failed to process document {file_path}") from e
