# test_pipeline.py
import os
import logging
from fastapi.testclient import TestClient
from main import app
import uvicorn

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete pipeline from document upload to query response"""
    logger.info("Starting complete pipeline test")
    
    client = TestClient(app)
    
    try:
        # 1. Test document upload
        logger.info("Testing document upload")
        test_file_path = "project-requirements-doc.md"
        
        if not os.path.exists(test_file_path):
            logger.error(f"Test file not found: {test_file_path}")
            return
        
        with open(test_file_path, "rb") as f:
            response = client.post(
                "/upload/",
                files={"file": ("test_doc.md", f, "text/markdown")},
                params={"user_id": "test_user"}
            )
        
        if response.status_code != 201:
            logger.error(f"Document upload failed: {response.text}")
            return
        
        project_id = response.json()["project_id"]
        logger.info(f"Document uploaded successfully. Project ID: {project_id}")
        
        # 2. Test document querying
        logger.info("Testing document querying")
        test_query = "What are the main components of the system?"
        
        response = client.post(
            "/query/",
            json={
                "project_id": project_id,
                "user_id": "test_user",
                "query": test_query
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Query failed: {response.text}")
            return
            
        logger.info("Query successful")
        logger.debug(f"Response: {response.json()}")
        
        # 3. Test risk report generation
        logger.info("Testing risk report generation")
        response = client.post(
            "/analyze/",
            json={
                "project_id": project_id,
                "user_id": "test_user",
                "query": "Generate a risk report"
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Risk report generation failed: {response.text}")
            return
            
        logger.info("Risk report generated successfully")
        logger.debug(f"Risk Report: {response.json()}")
        
        # 4. Test chat history retrieval
        logger.info("Testing chat history retrieval")
        response = client.get(f"/history/{project_id}")
        
        if response.status_code != 200:
            logger.error(f"Chat history retrieval failed: {response.text}")
            return
            
        logger.info("Chat history retrieved successfully")
        logger.debug(f"Chat History: {response.json()}")
        
        logger.info("All pipeline tests completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Run the test
    test_complete_pipeline()