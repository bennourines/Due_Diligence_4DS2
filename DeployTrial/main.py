# main.py - FastAPI backend
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
import pymongo
from datetime import datetime
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
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
from document_processor import DocumentProcessor
from search_engine import HybridSearchEngine
from rag_pipeline import RAGPipeline
from OpenRouter.risk_analyzer import RiskAnalyzer
from database import get_mongodb_connection, store_conversation
from models import UserQuery, ChatMessage, RiskReport

app = FastAPI(title="Crypto Due Diligence API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
doc_processor = DocumentProcessor()
search_engine = HybridSearchEngine()
rag_pipeline = RAGPipeline()
risk_analyzer = RiskAnalyzer()

# MongoDB connection
@app.on_event("startup")
async def startup_db_client():
    logger.info("Connecting to MongoDB...")
    try:
        app.mongodb_client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
        app.mongodb = app.mongodb_client.crypto_due_diligence
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise e
    
@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Closing MongoDB connection...")
    app.mongodb_client.close()

@app.post("/upload/", status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = "anonymous",
    db = Depends(get_mongodb_connection) # Inject db connection here

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
            db=db, # Pass the db object
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

async def process_document(db,file_path: str, file_name: str, project_id: str, user_id: str):
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
        logger.debug("Storing document metadata in MongoDB...")
        document_metadata = {
            "project_id": project_id,
            "user_id": user_id,
            "filename": file_name,
            "upload_time": datetime.utcnow(),
            "status": "processed",
            "document_count": len(docs)
        }
        
        await db.documents.insert_one(document_metadata)
        logger.info("Document metadata stored successfully")
        
        # Clean up temp file
        os.remove(file_path)
        logger.debug(f"Temporary file removed: {file_path}")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        # Update document status to failed
        await app.mongodb.documents.update_one(
            {"project_id": project_id},
            {"$set": {"status": "failed", "error": str(e)}}
        )
        logger.info("Document status updated to failed")

@app.post("/query/", response_model=ChatMessage)
async def query_documents(
    query: UserQuery,
    db = Depends(get_mongodb_connection)
):
    """Query documents using the RAG pipeline"""
    try:
        logger.info(f"Received query request from user {query.user_id} for project {query.project_id}")
        logger.debug(f"Query text: {query.query}")
        
        # Get relevant contexts using hybrid search
        logger.debug("Performing hybrid search...")
        contexts = search_engine.hybrid_search(
            query.query,
            project_id=query.project_id,
            k=5
        )
        logger.debug(f"Found {len(contexts)} relevant contexts")
        
        # Generate response using RAG pipeline
        logger.debug("Generating response using RAG pipeline...")
        response = rag_pipeline.generate_response(query.query, contexts)
        logger.info("Response generated successfully")
        
        # Store conversation in MongoDB
        logger.debug("Storing conversation in MongoDB...")
        chat_message = ChatMessage(
            project_id=query.project_id,
            user_id=query.user_id,
            query=query.query,
            response=response,
            timestamp=datetime.utcnow()
        )
        
        await store_conversation(db, chat_message.dict())
        logger.info("Conversation stored successfully")
        
        return chat_message
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/analyze/", response_model=RiskReport)
async def generate_risk_report(
    query: UserQuery,
    db = Depends(get_mongodb_connection)
):
    """Generate a comprehensive risk report for the crypto asset"""
    try:
        logger.info(f"Generating risk report for project {query.project_id}")
        
        # Generate risk report
        logger.debug("Running risk analysis...")
        risk_report = risk_analyzer.generate_report(
            project_id=query.project_id,
            search_engine=search_engine,
            rag_pipeline=rag_pipeline
        )
        logger.info("Risk report generated successfully")
        
        # Store report in MongoDB
        logger.debug("Storing risk report in MongoDB...")
        report_doc = {
            "project_id": query.project_id,
            "user_id": query.user_id,
            "timestamp": datetime.utcnow(),
            "report": risk_report
        }
        
        await db.reports.insert_one(report_doc)
        logger.info("Risk report stored successfully")
        
        return risk_report
        
    except Exception as e:
        logger.error(f"Error generating risk report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating risk report: {str(e)}")

@app.get("/history/{project_id}", response_model=List[ChatMessage])
async def get_chat_history(
    project_id: str,
    db = Depends(get_mongodb_connection)
):
    """Retrieve chat history for a specific project"""
    try:
        logger.info(f"Retrieving chat history for project {project_id}")
        
        chat_history = await db.conversations.find(
            {"project_id": project_id}
        ).sort("timestamp", -1).to_list(50)  # Limit to last 50 messages
        
        logger.debug(f"Retrieved {len(chat_history)} chat messages")
        return chat_history
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)