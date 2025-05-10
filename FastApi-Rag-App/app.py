import os
import glob
import logging
import uuid
import time  
from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import shutil
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Get Serper API key from environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    logger.warning("SERPER_API_KEY not found in environment variables")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize embeddings
logger.info("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Initialize Qdrant (local storage)
COLLECTION_NAME = "document_store"
VECTOR_SIZE = 384  # For all-MiniLM-L6-v2

# Create Qdrant client
logger.info("Connecting to Qdrant vector store...")
qdrant_path = "./qdrant_data"
os.makedirs(qdrant_path, exist_ok=True)

client = QdrantClient(path=qdrant_path)

# Check if collection exists, create if it doesn't
try:
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Using existing Qdrant collection '{COLLECTION_NAME}'")
except Exception:
    logger.info(f"Creating new Qdrant collection '{COLLECTION_NAME}'")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

# Initialize Qdrant wrapper with UUID generated IDs
vectordb = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# Shared text splitter for all documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Angular frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestResponse(BaseModel):
    status: str
    chunks_added: int = 0
    message: str = ""

class QARequest(BaseModel):
    id: str
    question: str


# Define the request model
class QATestQuestion(BaseModel):
    question: str
    reference_answer: str
    doc_id: str = "all"

class QAPerformanceRequest(BaseModel):
    test_questions: List[QATestQuestion]
    threshold: float = 0.7  # Minimum acceptable F1 score

class SearchRequest(BaseModel):
    keyword: str
    num_results: int = 10  # Default number of results to return

class SearchResponse(BaseModel):
    status: str
    results: List[dict] = []
    message: str = ""


@app.on_event("startup")
def load_cleaned_store():

    # Check if the function is disabled via environment variable
    disable_startup_load = os.getenv("DISABLE_STARTUP_LOAD", "false").lower() == "true"
    
    if disable_startup_load:
        logger.info("Startup data loading disabled by DISABLE_STARTUP_LOAD environment variable")
        return

    text_dir = "cleaned_text"  # Your folder name
    logger.info(f"Loading files from '{text_dir}'...")
    
    # Check if directory exists
    if not os.path.exists(text_dir):
        logger.warning(f"Directory '{text_dir}' does not exist. Skipping.")
        return
    
    # Get all files in the directory (supporting both .txt and .pdf)
    all_files = glob.glob(os.path.join(text_dir, "*.txt")) + glob.glob(os.path.join(text_dir, "*.pdf"))
    logger.info(f"Found {len(all_files)} files.")
    
    for file_path in all_files:
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        file_ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Processing file '{file_path}' as doc_id '{doc_id}'...")
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            logger.warning(f"File '{file_path}' is empty. Skipping.")
            continue
            
        try:
            # Load based on file extension
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
                file_type = "pdf"
            else:  # Default to text loader
                loader = TextLoader(file_path)
                file_type = "text"
                
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} raw documents from '{file_path}'.")
            
            # Add document source metadata
            for doc in docs:
                doc.metadata["source"] = file_path
                doc.metadata["doc_id"] = doc_id
                doc.metadata["file_type"] = file_type
            
            chunks = splitter.split_documents(docs)
            logger.info(f"Split into {len(chunks)} chunks.")
            
            # Skip if no chunks were created
            if not chunks:
                logger.warning(f"No chunks were created from '{file_path}'. Skipping.")
                continue
            
            # Create a list of UUIDs for document IDs
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            
            # Add documents to Qdrant with UUID ids
            vectordb.add_documents(documents=chunks, ids=ids)
            logger.info(f"Added {len(chunks)} chunks to vector store from '{doc_id}'.")
            
        except Exception as e:
            logger.error(f"Error processing file '{file_path}': {str(e)}")
            continue
    
    logger.info(f"Completed loading all files from '{text_dir}' into Qdrant.")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Ingest a new file: load, split, and add to vector store.
    """
    logger.info(f"Received ingest request for file: {file.filename}")
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the uploaded file to a temporary location
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Load based on file extension
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
            file_type = "pdf"
        else:  # Default to text loader
            loader = TextLoader(file_path)
            file_type = "text"
            
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from '{file.filename}'.")
        
        # Add document source metadata
        for doc in docs:
            doc.metadata["source"] = file.filename
            doc.metadata["doc_id"] = os.path.splitext(file.filename)[0]
            doc.metadata["file_type"] = file_type
        
        chunks = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks.")
        
        # Generate UUID ids for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Add to vector store
        vectordb.add_documents(documents=chunks, ids=ids)
        logger.info(f"Added {len(chunks)} chunks to vector store from '{file.filename}'.")
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return IngestResponse(status="ok", chunks_added=len(chunks))
    except Exception as e:
        logger.error(f"Error during ingest: {str(e)}")
        # Clean up the temporary file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        return IngestResponse(status="error", message=str(e))
    finally:
        # Ensure file is closed
        await file.close()

@app.post("/qa")
def qa(req: QARequest):
    """
    Retrieve and answer questions using all embedded knowledge.
    """
    logger.info(f"Received QA request: id={req.id}, question='{req.question}'")
    try:
        # Create a retriever with optional filtering
        search_filter = None
        if req.id != "all" and req.id != "":  # If specific document is requested
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.doc_id",
                        match=models.MatchValue(value=req.id)
                    )
                ]
            )
            
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "filter": search_filter
            }
        )
        logger.info(f"Retriever initialized with filter for doc_id: {req.id if req.id != 'all' and req.id != '' else 'No filter'}")
        
        # Initialize LLM with error handling
        try:
            llm = Ollama(model="llama3", temperature=0.1)
            logger.info("Initialized Ollama LLM with llama3 model.")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {str(e)}")
            return {"answer": "Error: Could not initialize language model. Please check if Ollama is running."}
        
        # Create and run the chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        logger.info("Running retrieval QA chain...")
        result = chain({"query": req.question})
        answer = result["result"]
        
        # Extract sources for transparency
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "doc_id": doc.metadata.get("doc_id", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "file_type": doc.metadata.get("file_type", "Unknown")
            }
            sources.append(source_info)
        
        logger.info(f"Answer generated with {len(sources)} source documents.")
        logger.info(f"Answer: {answer}")
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in QA endpoint: {str(e)}")
        return {"answer": f"Error: {str(e)}"}

@app.get("/status")
def get_status():
    """
    Get information about the vector store status.
    """
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
        
        return {
            "status": "ok",
            "collection_name": COLLECTION_NAME,
            "documents_count": points_count,
            "vector_size": VECTOR_SIZE
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.delete("/clear")
def clear_collection():
    """
    Clear all documents from the collection.
    """
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' deleted.")
        
        # Recreate the collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' recreated.")
        
        return {"status": "ok", "message": f"Collection '{COLLECTION_NAME}' cleared and recreated."}
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Perform a web search using Serper API for the given keyword.
    """
    logger.info(f"Received search request for keyword: {req.keyword}")
    
    if not SERPER_API_KEY:
        return SearchResponse(
            status="error",
            message="Serper API key not configured"
        )
    
    try:
        # Prepare the request to Serper API
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {
            'q': req.keyword,
            'num': req.num_results
        }
        
        # Make the request to Serper API
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        # Process the response
        search_results = response.json()
        
        # Extract organic search results
        organic_results = search_results.get('organic', [])
        processed_results = [{
            'title': result.get('title', ''),
            'link': result.get('link', ''),
            'snippet': result.get('snippet', '')
        } for result in organic_results[:req.num_results]]
        
        logger.info(f"Successfully retrieved {len(processed_results)} search results")
        
        return SearchResponse(
            status="ok",
            results=processed_results
        )
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error making request to Serper API: {str(e)}"
        logger.error(error_msg)
        return SearchResponse(
            status="error",
            message=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error during search: {str(e)}"
        logger.error(error_msg)
        return SearchResponse(
            status="error",
            message=error_msg
        )


@app.post("/qa/performance")
def qa_performance_metrics(req: QAPerformanceRequest):
    """
    Test performance metrics and scores for the QA system against reference answers.
    Returns precision, recall, F1 score, latency, and confidence metrics.
    """
    logger.info(f"Running performance test on QA system with {len(req.test_questions)} questions")
    
    try:
        # Initialize metrics
        results = {
            "overall": {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "latency_ms": 0.0,
                "confidence": 0.0
            },
            "questions": []
        }
        
        # Initialize LLM for evaluation
        try:
            eval_llm = Ollama(model="llama3", temperature=0.0)
            logger.info("Initialized evaluation LLM")
        except Exception as e:
            logger.error(f"Error initializing evaluation LLM: {str(e)}")
            return {"error": "Could not initialize evaluation model"}
        
        total_latency = 0
        
        # Test each question
        for idx, test_item in enumerate(req.test_questions):
            question = test_item.question
            reference_answer = test_item.reference_answer
            doc_id = test_item.doc_id if hasattr(test_item, "doc_id") else "all"
            
            logger.info(f"Testing question {idx+1}/{len(req.test_questions)}: '{question}'")
            
            # Measure latency
            start_time = time.time()
            
            # Create a retriever with optional filtering
            search_filter = None
            if doc_id != "all" and doc_id != "":
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
                
            retriever = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4,
                    "filter": search_filter
                }
            )
            
            # Create and run the chain
            chain = RetrievalQA.from_chain_type(
                llm=Ollama(model="llama3", temperature=0.1),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            result = chain({"query": question})
            answer = result["result"]
            
            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            total_latency += latency_ms
            
            # Use evaluation LLM to calculate metrics
            eval_prompt = f"""
            You are an expert at evaluating question answering systems. 
            Compare the system's answer to the reference answer and score on a scale of 0 to 1:
            
            Question: {question}
            
            Reference Answer: {reference_answer}
            
            System Answer: {answer}
            
            Please provide numerical scores for:
            1. Precision (How accurate is the provided information? 0-1)
            2. Recall (How complete is the answer compared to reference? 0-1)
            3. Confidence (How certain can we be about this answer? 0-1)
            
            Also calculate F1 score as the harmonic mean of precision and recall.
            Return your evaluation as a JSON object with these metrics as float values.
            """
            
            try:
                eval_result = eval_llm.invoke(eval_prompt)
                
                # Parse the JSON metrics from the response
                # Find JSON content between ```json and ``` if present
                import re
                import json
                
                json_pattern = r'```json\s*([\s\S]*?)\s*```'
                json_match = re.search(json_pattern, eval_result)
                
                if json_match:
                    metrics_str = json_match.group(1)
                else:
                    # Try to find any JSON-like content
                    json_pattern = r'\{[\s\S]*?\}'
                    json_match = re.search(json_pattern, eval_result)
                    metrics_str = json_match.group(0) if json_match else "{}"
                
                try:
                    metrics = json.loads(metrics_str)
                except:
                    # Fallback if JSON parsing fails
                    logger.warning("JSON parsing failed, using default metrics")
                    metrics = {
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1_score": 0.5,
                        "confidence": 0.5
                    }
                    
                # Calculate F1 if not provided
                if "f1_score" not in metrics and "precision" in metrics and "recall" in metrics:
                    p = metrics["precision"]
                    r = metrics["recall"]
                    metrics["f1_score"] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            except Exception as e:
                logger.error(f"Error in evaluation: {str(e)}")
                metrics = {
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1_score": 0.5,
                    "confidence": 0.5
                }
            
            # Extract sources
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "doc_id": doc.metadata.get("doc_id", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                    "file_type": doc.metadata.get("file_type", "Unknown")
                }
                sources.append(source_info)
            
            # Add question results
            question_result = {
                "question": question,
                "reference_answer": reference_answer,
                "system_answer": answer,
                "metrics": metrics,
                "latency_ms": latency_ms,
                "sources": sources
            }
            
            results["questions"].append(question_result)
            
            logger.info(f"Question {idx+1} evaluated - F1: {metrics.get('f1_score', 0):.2f}, Latency: {latency_ms:.2f}ms")
        
        # Calculate overall metrics
        if req.test_questions:
            avg_precision = sum(q["metrics"].get("precision", 0) for q in results["questions"]) / len(req.test_questions)
            avg_recall = sum(q["metrics"].get("recall", 0) for q in results["questions"]) / len(req.test_questions)
            avg_f1 = sum(q["metrics"].get("f1_score", 0) for q in results["questions"]) / len(req.test_questions)
            avg_confidence = sum(q["metrics"].get("confidence", 0) for q in results["questions"]) / len(req.test_questions)
            avg_latency = total_latency / len(req.test_questions)
            
            results["overall"] = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1,
                "latency_ms": avg_latency,
                "confidence": avg_confidence,
                "questions_evaluated": len(req.test_questions)
            }
        
        logger.info(f"Performance test completed. Overall F1: {results['overall']['f1_score']:.2f}")
        return results
    
    except Exception as e:
        logger.error(f"Error in QA performance endpoint: {str(e)}")
        return {"error": f"Error in performance testing: {str(e)}"}