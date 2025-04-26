# filepath: DeployTrial2/api/routers/projects.py
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List
import logging

from models.api_models import ProjectCreate, ProjectResponse
from models.db_models import ProjectInDB, UserInDB
from database.connection import get_database
from database import crud
from api.dependencies import get_current_active_user, get_project_for_user # Import dependencies
# Import vector store manager for deletion
from retrieval.vector_store import FaissVectorStoreManager
from embeddings.generator import EmbeddingGenerator # Need instance for manager

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection Setup (Example - Reuse or manage globally) ---
# This should ideally be managed centrally
embedding_gen_proj = EmbeddingGenerator()
vector_store_mgr_proj = FaissVectorStoreManager(embedding_generator=embedding_gen_proj)


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_new_project(
    project_in: ProjectCreate,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Create a new project for the authenticated user.

    - **name**: Name of the project.
    - **description**: Optional description for the project.
    """
    logger.info(f"User {current_user.email} creating project: {project_in.name}")
    project_db = ProjectInDB(
        user_id=current_user.id,
        name=project_in.name,
        description=project_in.description
    )
    try:
        created_project = await crud.create_project(db=db, project_in=project_db)
        logger.info(f"Project '{created_project.name}' created with ID: {created_project.id}")
        # Use model_dump for Pydantic v2 compatibility if needed, ensure alias works
        return ProjectResponse(**created_project.model_dump(by_alias=True))
    except Exception as e:
        logger.error(f"Error creating project '{project_in.name}' for user {current_user.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the project."
        )

@router.get("/", response_model=List[ProjectResponse])
async def get_user_projects(
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    List all projects belonging to the authenticated user.
    """
    logger.debug(f"Fetching projects for user: {current_user.email}")
    projects_db = await crud.get_projects_by_user(db=db, user_id=current_user.id)
    logger.info(f"Found {len(projects_db)} projects for user: {current_user.email}")
    return [ProjectResponse(**p.model_dump(by_alias=True)) for p in projects_db]

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project_details(
    project_id: str, # Get project_id from path
    project: ProjectInDB = Depends(get_project_for_user) # Use dependency to get & verify project
    # db: AsyncIOMotorDatabase = Depends(get_database), # No longer needed directly
    # current_user: UserInDB = Depends(get_current_active_user) # No longer needed directly
):
    """
    Get details of a specific project belonging to the authenticated user.
    """
    logger.debug(f"Fetching details for project: {project_id}")
    # The dependency 'get_project_for_user' handles fetching and 404/403 errors
    return ProjectResponse(**project.model_dump(by_alias=True))

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_project(
    project_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_active_user),
    # Inject vector store manager if needed for deletion logic here
    vector_store_manager: FaissVectorStoreManager = Depends(lambda: vector_store_mgr_proj) # Example injection
):
    """
    Delete a specific project and its associated data (documents metadata, history, vector store files).
    """
    logger.warning(f"User {current_user.email} attempting to delete project: {project_id}")

    # 1. Delete DB records (project, docs, history) - handled by crud.delete_project
    deleted_db = await crud.delete_project(db=db, project_id=project_id, user_id=current_user.id)

    if not deleted_db:
        # crud.delete_project logs warning if not found
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # 2. Delete associated vector store files
    try:
        logger.info(f"Attempting to delete vector store files for project: {project_id}")
        await vector_store_manager.delete_project_store(user_id=current_user.id, project_id=project_id)
        logger.info(f"Successfully requested deletion of vector store files for project: {project_id}")
    except Exception as e:
        # Log error but don't fail the request if DB deletion succeeded
        logger.error(f"Error deleting vector store files for project {project_id}: {e}", exc_info=True)
        # Optionally, mark the project for cleanup later

    logger.info(f"Project {project_id} and associated data deleted successfully.")
    return None # Return No Content (FastAPI handles the 204 status)
