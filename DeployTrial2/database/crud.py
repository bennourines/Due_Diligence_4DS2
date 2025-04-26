# filepath: DeployTrial2/database/crud.py
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Optional

from pymongo import ASCENDING, DESCENDING
from models.db_models import (
    UserInDB, ProjectInDB, DocumentMetadataInDB, ConversationHistoryInDB,
    QueryHistoryContentInDB, AnalysisHistoryContentInDB # Import content types
)
from models.api_models import UserCreate, HistoryEntry # For user creation structure and history response
from auth.security import get_password_hash # Import hashing function
from bson import ObjectId # If using ObjectId directly, though UUID strings are used here
import logging
from datetime import datetime # Import datetime

logger = logging.getLogger(__name__)

# --- User CRUD ---
async def get_user_by_email(db: AsyncIOMotorDatabase, email: str) -> Optional[UserInDB]:
    logger.debug(f"Fetching user by email: {email}")
    user_dict = await db["users"].find_one({"email": email})
    if user_dict:
        logger.debug(f"User found: {email}")
        return UserInDB(**user_dict)
    logger.debug(f"User not found: {email}")
    return None

async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> Optional[UserInDB]:
    logger.debug(f"Fetching user by ID: {user_id}")
    user_dict = await db["users"].find_one({"_id": user_id})
    if user_dict:
        logger.debug(f"User found: {user_id}")
        return UserInDB(**user_dict)
    logger.debug(f"User not found: {user_id}")
    return None


async def create_user(db: AsyncIOMotorDatabase, user_in: UserCreate) -> UserInDB:
    logger.info(f"Creating new user: {user_in.email}")
    hashed_password = get_password_hash(user_in.password)
    user_db = UserInDB(email=user_in.email, hashed_password=hashed_password)
    try:
        insert_result = await db["users"].insert_one(user_db.model_dump(by_alias=True))
        logger.info(f"User created with ID: {insert_result.inserted_id}")
        # Fetch the created user to return the full object with ID
        # Use the generated ID directly if possible, otherwise fetch by email
        created_user = await get_user_by_id(db, user_db.id) # Fetch using the generated UUID string
        if not created_user: # Should not happen in normal flow
             logger.error(f"Failed to retrieve created user immediately after insertion: {user_in.email}")
             raise Exception("Failed to retrieve created user.")
        return created_user
    except Exception as e:
        logger.error(f"Error creating user {user_in.email}: {e}", exc_info=True)
        raise

# --- Project CRUD ---
async def create_project(db: AsyncIOMotorDatabase, project_in: ProjectInDB) -> ProjectInDB:
    logger.info(f"Creating new project '{project_in.name}' for user {project_in.user_id}")
    try:
        await db["projects"].insert_one(project_in.model_dump(by_alias=True))
        # Fetch created project
        created_project = await db["projects"].find_one({"_id": project_in.id})
        if not created_project:
            logger.error(f"Failed to retrieve created project immediately: {project_in.id}")
            raise Exception("Failed to retrieve created project.")
        logger.info(f"Project created successfully: {project_in.id}")
        return ProjectInDB(**created_project)
    except Exception as e:
        logger.error(f"Error creating project {project_in.name}: {e}", exc_info=True)
        raise

async def get_project_by_id(db: AsyncIOMotorDatabase, project_id: str, user_id: str) -> Optional[ProjectInDB]:
    logger.debug(f"Fetching project by ID: {project_id} for user: {user_id}")
    project_dict = await db["projects"].find_one({"_id": project_id, "user_id": user_id})
    if project_dict:
        logger.debug(f"Project found: {project_id}")
        return ProjectInDB(**project_dict)
    logger.debug(f"Project not found or access denied: {project_id}")
    return None

async def get_projects_by_user(db: AsyncIOMotorDatabase, user_id: str) -> List[ProjectInDB]:
    logger.debug(f"Fetching all projects for user: {user_id}")
    projects = []
    cursor = db["projects"].find({"user_id": user_id}).sort("created_at", DESCENDING)
    async for project_dict in cursor:
        projects.append(ProjectInDB(**project_dict))
    logger.debug(f"Found {len(projects)} projects for user: {user_id}")
    return projects

async def delete_project(db: AsyncIOMotorDatabase, project_id: str, user_id: str) -> bool:
    logger.warning(f"Attempting to delete project: {project_id} for user: {user_id}")
    # Note: Also consider deleting associated documents, history, and vector store files
    delete_result = await db["projects"].delete_one({"_id": project_id, "user_id": user_id})

    if delete_result.deleted_count > 0:
        logger.info(f"Deleted project record: {project_id}")
        # Also delete related data...
        doc_delete_result = await db["documents"].delete_many({"project_id": project_id, "user_id": user_id})
        logger.info(f"Deleted {doc_delete_result.deleted_count} associated document records for project {project_id}")
        hist_delete_result = await db["history"].delete_many({"project_id": project_id, "user_id": user_id})
        logger.info(f"Deleted {hist_delete_result.deleted_count} associated history records for project {project_id}")
        # Need logic to delete vector store files for this project_id/user_id (handled elsewhere)
        return True
    else:
        logger.warning(f"Project not found or not deleted: {project_id}")
        return False

# --- Document Metadata CRUD ---
async def create_document_metadata(db: AsyncIOMotorDatabase, doc_meta: DocumentMetadataInDB) -> DocumentMetadataInDB:
    logger.info(f"Creating document metadata for file '{doc_meta.filename}' in project {doc_meta.project_id}")
    try:
        await db["documents"].insert_one(doc_meta.model_dump(by_alias=True))
        created_doc = await db["documents"].find_one({"_id": doc_meta.id})
        if not created_doc:
            logger.error(f"Failed to retrieve created document metadata immediately: {doc_meta.id}")
            raise Exception("Failed to retrieve created document metadata.")
        logger.info(f"Document metadata created successfully: {doc_meta.id}")
        return DocumentMetadataInDB(**created_doc)
    except Exception as e:
        logger.error(f"Error creating document metadata for {doc_meta.filename}: {e}", exc_info=True)
        raise

async def get_document_metadata(db: AsyncIOMotorDatabase, doc_id: str, project_id: str, user_id: str) -> Optional[DocumentMetadataInDB]:
     logger.debug(f"Fetching document metadata by ID: {doc_id} for project: {project_id}")
     doc_dict = await db["documents"].find_one({"_id": doc_id, "project_id": project_id, "user_id": user_id})
     if doc_dict:
         logger.debug(f"Document metadata found: {doc_id}")
         return DocumentMetadataInDB(**doc_dict)
     logger.debug(f"Document metadata not found or access denied: {doc_id}")
     return None

async def update_document_status(db: AsyncIOMotorDatabase, doc_id: str, status: str, **kwargs) -> bool:
    logger.info(f"Updating status for document {doc_id} to '{status}'")
    update_data = {"status": status, **kwargs}
    # Ensure processed_at is set only once on completion/failure
    if status in ["completed", "failed"]:
        update_data["processed_at"] = datetime.utcnow()

    try:
        result = await db["documents"].update_one(
            {"_id": doc_id},
            {"$set": update_data}
        )
        if result.modified_count > 0:
            logger.info(f"Successfully updated status for document {doc_id}")
            return True
        elif result.matched_count == 1:
             logger.warning(f"Document {doc_id} found but status was not modified (already '{status}'?).")
             return True # Or False depending on desired behavior
        else:
             logger.warning(f"Document {doc_id} not found for status update.")
             return False
    except Exception as e:
        logger.error(f"Error updating status for document {doc_id}: {e}", exc_info=True)
        return False


async def get_documents_by_project(db: AsyncIOMotorDatabase, project_id: str, user_id: str) -> List[DocumentMetadataInDB]:
    logger.debug(f"Fetching all document metadata for project: {project_id}")
    docs = []
    cursor = db["documents"].find({"project_id": project_id, "user_id": user_id}).sort("uploaded_at", ASCENDING)
    async for doc_dict in cursor:
        docs.append(DocumentMetadataInDB(**doc_dict))
    logger.debug(f"Found {len(docs)} document metadata entries for project: {project_id}")
    return docs

# --- Conversation History CRUD ---
async def add_history_entry(db: AsyncIOMotorDatabase, entry: ConversationHistoryInDB) -> ConversationHistoryInDB:
    logger.info(f"Adding history entry of type '{entry.type}' for project {entry.project_id}")
    try:
        # Ensure content matches the type
        if entry.type == "query" and not isinstance(entry.content, QueryHistoryContentInDB):
            raise ValueError("History entry type 'query' requires QueryHistoryContentInDB")
        if entry.type == "analysis" and not isinstance(entry.content, AnalysisHistoryContentInDB):
             raise ValueError("History entry type 'analysis' requires AnalysisHistoryContentInDB")

        await db["history"].insert_one(entry.model_dump(by_alias=True))
        # Fetch the created entry to return the full object
        created_entry = await db["history"].find_one({"_id": entry.id})
        if not created_entry:
            logger.error(f"Failed to retrieve created history entry immediately: {entry.id}")
            raise Exception("Failed to retrieve created history entry.")
        logger.info(f"History entry added successfully: {entry.id}")
        return ConversationHistoryInDB(**created_entry)
    except Exception as e:
        logger.error(f"Error adding history entry for project {entry.project_id}: {e}", exc_info=True)
        raise

async def get_history_by_project(db: AsyncIOMotorDatabase, project_id: str, user_id: str) -> List[ConversationHistoryInDB]:
    logger.debug(f"Fetching history for project: {project_id}")
    history = []
    # Sort by timestamp ascending to get chronological order
    cursor = db["history"].find(
        {"project_id": project_id, "user_id": user_id}
    ).sort("timestamp", ASCENDING)
    async for entry_dict in cursor:
        try:
            # Need to handle potential validation errors if DB schema drifts
            history.append(ConversationHistoryInDB(**entry_dict))
        except Exception as e:
             logger.error(f"Error parsing history entry {entry_dict.get('_id')} from DB: {e}")
             # Skip corrupted entries or handle differently
             continue
    logger.debug(f"Found {len(history)} history entries for project: {project_id}")
    return history
