# document_processor.py
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document types and split into chunks"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        logger.info(f"Initializing DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def load_and_split(self, file_path: str) -> List:
        """Load document and split into chunks"""
        logger.info(f"Processing file: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Select appropriate loader based on file extension
        logger.debug(f"Selecting loader for file extension: {file_extension}")
        if file_extension == ".pdf":
            logger.debug("Using PyPDFLoader")
            loader = PyPDFLoader(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            logger.debug("Using UnstructuredExcelLoader")
            loader = UnstructuredExcelLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            logger.debug("Using Docx2txtLoader")
            loader = Docx2txtLoader(file_path)
        else:
            # Default to unstructured for other file types
            logger.debug("Using UnstructuredFileLoader")
            loader = UnstructuredFileLoader(file_path)
        
        try:
            # Load documents
            logger.debug("Loading document content")
            documents = loader.load()
            logger.info(f"Successfully loaded document with {len(documents)} pages/sections")
            
            # Add source metadata
            logger.debug("Adding source metadata")
            for doc in documents:
                doc.metadata["source"] = os.path.basename(file_path)
            
            # Split documents
            logger.debug("Splitting documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Log chunk sizes for debugging
            for i, chunk in enumerate(chunks):
                logger.debug(f"Chunk {i+1}: {len(chunk.page_content)} characters")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}", exc_info=True)
            raise