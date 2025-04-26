# filepath: DeployTrial2/api/routers/query.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
from typing import List
import logging

from models.api_models import AnalysisHistoryContent, QueryRequest, QueryResponse, HistoryEntry, QueryHistoryContent # Import history models
from models.db_models import UserInDB, ConversationHistoryInDB, QueryHistoryContentInDB, AnalysisHistoryContentInDB # Import DB history models
from database.connection import get_database
from database import crud
from api.dependencies import get_current_active_user, get_project_for_user # Use project dependency
from generation.rag_pipeline import RAGPipeline
from retrieval.vector_store import FaissVectorStoreManager # Need instance
from embeddings.generator import EmbeddingGenerator # Need instance
from generation.llm_clients import get_default_llm # Import LLM client factory
from models.db_models import ProjectInDB # Import ProjectInDB for type hint

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection Setup (Example - Centralize this later) ---
# These instances should ideally be created once at startup and managed/injected.
# Using simple global instances here for demonstration.
try:
    embedding_gen_query = EmbeddingGenerator()
    vector_store_mgr_query = FaissVectorStoreManager(embedding_generator=embedding_gen_query)
    llm_client_query = get_default_llm()
    rag_pipeline_instance = RAGPipeline(
        vector_store_manager=vector_store_mgr_query,
        llm_client=llm_client_query
    )
    logger.info("Query endpoint dependencies initialized.")
except Exception as e:
     logger.critical(f"Failed to initialize dependencies for query endpoint: {e}", exc_info=True)
     # Application might not function correctly, consider raising SystemExit
     # raise SystemExit("Failed to initialize query dependencies")
     # For now, allow startup but log critical error
     rag_pipeline_instance = None # Ensure it's None if init fails


# --- Helper for Background Task ---
async def save_query_history(db: AsyncIOMotorDatabase, entry_data: dict):
    """Async function to save history entry, suitable for background tasks."""
    try:
        # Reconstruct Pydantic model before saving
        entry = ConversationHistoryInDB(**entry_data)
        await crud.add_history_entry(db, entry)
        logger.info(f"Query history saved successfully for project {entry.project_id}")
    except Exception as e:
        logger.error(f"Background task failed to save query history: {e}", exc_info=True)


# --- Endpoint ---

@router.post("/{project_id}/query", response_model=QueryResponse)
async def handle_project_query(
    project_id: str, # Get project_id from path
    request: QueryRequest,
    background_tasks: BackgroundTasks, # For saving history
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_active_user),
    project: ProjectInDB = Depends(get_project_for_user), # Use dependency to get & verify project
    # Inject dependencies if using FastAPI's DI properly
    # rag_pipeline: RAGPipeline = Depends(get_rag_pipeline_instance) # Example
):
    """
    Accepts a user query for a specific project, performs RAG,
    returns the generated response, and stores the interaction in history.

    - **project_id**: The ID of the project to query within.
    - **query**: The user's question.
    - **top_k**: (Optional) Number of document chunks to retrieve (default: 5).
    """
    if rag_pipeline_instance is None:
         logger.critical("RAG pipeline instance is not available. Query endpoint cannot function.")
         raise HTTPException(
              status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
              detail="Query processing service is currently unavailable."
         )

    logger.info(f"Received query for project {project_id} by user {current_user.email}")
    # Project existence and ownership verified by get_project_for_user dependency

    # 2. Perform RAG using the pipeline instance
    try:
        rag_result = await rag_pipeline_instance.generate_response(
            user_id=current_user.id,
            project_id=project.id, # Use project ID from the verified project object
            query=request.query,
            top_k=request.top_k or 5 # Use default if not provided
        )
    except HTTPException as http_exc:
         # Re-raise HTTP exceptions from the pipeline (like 500, 503, 413)
         logger.warning(f"RAG pipeline raised HTTPException: {http_exc.status_code} - {http_exc.detail}")
         raise http_exc
    except Exception as e:
         # Catch unexpected errors during RAG
         logger.error(f"Unexpected error during RAG for project {project_id}: {e}", exc_info=True)
         raise HTTPException(
              status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
              detail="An internal error occurred during query processing."
         )

    # 3. Prepare response object
    response_time = datetime.utcnow()
    response = QueryResponse(
        project_id=project.id,
        query=request.query,
        response=rag_result["answer"],
        retrieved_sources=rag_result["sources"],
        timestamp=response_time
    )

    # 4. Prepare data for history saving (use DB model structure)
    history_content = QueryHistoryContentInDB(
        query=request.query,
        response=rag_result["answer"],
        retrieved_sources=rag_result["sources"]
    )
    history_entry_data = ConversationHistoryInDB(
        project_id=project.id,
        user_id=current_user.id,
        type="query",
        content=history_content, # Embed the content object
        timestamp=response_time
    ).model_dump(by_alias=True) # Convert to dict for background task

    # Use background task to save history without blocking response
    background_tasks.add_task(save_query_history, db, history_entry_data)
    logger.debug(f"Added history saving task for project {project.id}")

    return response

# --- Endpoint for History ---
@router.get("/{project_id}/history", response_model=List[HistoryEntry])
async def get_project_history(
    project_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    project: ProjectInDB = Depends(get_project_for_user), # Verify project access
    current_user: UserInDB = Depends(get_current_active_user) # Ensure user is active
):
    """
    Retrieve the conversation history (queries and analyses) for a specific project.
    """
    logger.info(f"Fetching history for project {project_id} for user {current_user.email}")
    history_db = await crud.get_history_by_project(db=db, project_id=project.id, user_id=current_user.id)

    # Convert DB models to API response models
    history_api = []
    for entry_db in history_db:
        try:
            # Map DB content model to API content model before creating HistoryEntry
            api_content = None
            if entry_db.type == "query" and isinstance(entry_db.content, QueryHistoryContentInDB):
                api_content = QueryHistoryContent(
                    project_id=entry_db.project_id, # Add fields required by API model if different
                    timestamp=entry_db.timestamp,
                    query=entry_db.content.query,
                    response=entry_db.content.response,
                    retrieved_sources=entry_db.content.retrieved_sources
                )
            elif entry_db.type == "analysis" and isinstance(entry_db.content, AnalysisHistoryContentInDB):
                 # Assuming AnalysisHistoryContent API model exists and matches
                 api_content = AnalysisHistoryContent(
                      project_id=entry_db.project_id,
                      timestamp=entry_db.timestamp,
                      report=entry_db.content.report
                 )

            if api_content:
                history_api.append(HistoryEntry(
                    id=entry_db.id,
                    project_id=entry_db.project_id,
                    user_id=entry_db.user_id,
                    type=entry_db.type,
                    content=api_content,
                    timestamp=entry_db.timestamp
                ))
            else:
                 logger.warning(f"Skipping history entry {entry_db.id} due to unknown type or content mismatch.")

        except Exception as e:
             logger.error(f"Error converting history entry {entry_db.id} to API model: {e}")
             continue # Skip entries that fail validation/conversion

    logger.info(f"Returning {len(history_api)} history entries for project {project_id}")
    return history_api
