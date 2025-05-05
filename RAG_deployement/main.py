# main.py - FastAPI backend
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components
logger.info("Importing components...")
from document_processor import DocumentProcessor
from search_engine import HybridSearchEngine
from optimizedRag import RAGSystem
from storage import FileStorage
from models import UserQuery, ChatMessage
logger.info("Components imported successfully")

app = FastAPI(title="Crypto Due Diligence API")
logger.info("FastAPI app initialized")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware added")

# Initialize components
logger.info("Initializing components...")
try:
    doc_processor = DocumentProcessor()
    logger.info("DocumentProcessor initialized")
    search_engine = HybridSearchEngine()
    logger.info("HybridSearchEngine initialized")
    storage = FileStorage()
    logger.info("FileStorage initialized")
    # Initialize RAG system only when needed
    rag_system = None
    logger.info("RAGSystem initialization deferred")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}", exc_info=True)
    raise

@app.post("/upload/", status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = "anonymous"
):
    """Upload and process a document for analysis"""
    try:
        logger.info(f"Receiving upload request for file: {file.filename} from user: {user_id}")
        
        # Generate unique project ID
        project_id = str(uuid.uuid4())
        logger.debug(f"Generated project ID: {project_id}")
        
        # Save file to temporary location
        file_path = f"temp/{project_id}_{file.filename}"
        os.makedirs("temp", exist_ok=True)
        logger.debug(f"Saving file to: {file_path}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            logger.debug(f"File saved successfully: {len(content)} bytes")
            
        # Process document in background
        background_tasks.add_task(
            process_document,
            file_path=file_path,
            file_name=file.filename,
            project_id=project_id,
            user_id=user_id
        )
        logger.info(f"Document processing task created for project: {project_id}")
        
        return {"message": "Document processing started", "project_id": project_id}
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

async def process_document(file_path: str, file_name: str, project_id: str, user_id: str):
    """Process document and store embeddings"""
    try:
        logger.info(f"Starting document processing for {file_name} (Project ID: {project_id})")
        
        # Process document
        logger.debug("Loading and splitting document...")
        docs = doc_processor.load_and_split(file_path)
        logger.info(f"Document split into {len(docs)} chunks")
        
        # Generate embeddings and store in vector database
        logger.debug("Generating embeddings and storing in vector database...")
        search_engine.add_documents(docs, project_id)
        logger.info("Documents successfully added to search engine")
        
        # Store document metadata
        logger.debug("Storing document metadata...")
        document_metadata = {
            "project_id": project_id,
            "user_id": user_id,
            "filename": file_name,
            "upload_time": datetime.utcnow().isoformat(),
            "status": "processed",
            "document_count": len(docs)
        }
        
        storage.store_document(project_id, document_metadata)
        logger.info("Document metadata stored successfully")
        
        # Clean up temp file
        os.remove(file_path)
        logger.debug(f"Temporary file removed: {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        # Update document status to failed
        document_metadata = {
            "project_id": project_id,
            "user_id": user_id,
            "filename": file_name,
            "upload_time": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": str(e)
        }
        storage.store_document(project_id, document_metadata)
        logger.info("Document status updated to failed")

@app.post("/query/", response_model=ChatMessage)
async def query_documents(query: UserQuery):
    """Query documents using the RAG system"""
    try:
        logger.info(f"Received query request from user {query.user_id} for project {query.project_id}")
        logger.debug(f"Query text: {query.query}")
        
        # Initialize RAG system if needed
        global rag_system
        if rag_system is None:
            logger.info("Initializing RAG system...")
            rag_system = RAGSystem()
            logger.info("RAG system initialized")
        
        # Generate response using RAG system
        logger.debug("Generating response using RAG system...")
        response, eval_result = rag_system.query(query.query)
        logger.info(f"Response generated successfully. Retrieval latency: {eval_result.retrieval_latency:.3f}s, Generation latency: {eval_result.generation_latency:.3f}s")
        
        # Store conversation
        logger.debug("Storing conversation...")
        chat_message = ChatMessage(
            project_id=query.project_id,
            user_id=query.user_id,
            query=query.query,
            response=response,
            timestamp=datetime.utcnow().isoformat()
        )
        
        storage.store_conversation(query.project_id, chat_message.dict())
        logger.info("Conversation stored successfully")
        
        return chat_message
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/history/{project_id}", response_model=List[ChatMessage])
async def get_chat_history(project_id: str):
    """Retrieve chat history for a specific project"""
    try:
        logger.info(f"Retrieving chat history for project {project_id}")
        
        chat_history = storage.get_chat_history(project_id)
        logger.debug(f"Retrieved {len(chat_history)} chat messages")
        return chat_history
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)