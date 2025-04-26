# filepath: DeployTrial2/api/routers/documents.py
from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
)
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List
import os
import shutil
import logging
import uuid
from datetime import datetime # Import datetime

from models.api_models import DocumentMetadataResponse
from models.db_models import UserInDB, DocumentMetadataInDB, ProjectInDB # Import ProjectInDB
from database.connection import get_database
from database import crud
from api.dependencies import get_current_active_user, get_project_for_user # Use project dependency
from core.config import settings
# Need to import processing components and vector store manager
from processing.processor import DocumentProcessor
from embeddings.generator import EmbeddingGenerator
from retrieval.vector_store import FaissVectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection Setup (Example - Centralize this later) ---
# These instances should ideally be created once at startup and managed/injected.
try:
    doc_processor_instance = DocumentProcessor()
    # Ensure embedding generator and vector store manager are initialized
    # These might be shared with other routers, ensure consistency
    embedding_gen_doc = EmbeddingGenerator()
    vector_store_mgr_doc = FaissVectorStoreManager(embedding_generator=embedding_gen_doc)
    logger.info("Document endpoint dependencies initialized.")
except Exception as e:
     logger.critical(f"Failed to initialize dependencies for document endpoint: {e}", exc_info=True)
     # Mark as None to prevent endpoint usage if init fails
     doc_processor_instance = None
     embedding_gen_doc = None
     vector_store_mgr_doc = None


# --- Background Task Function ---
# IMPORTANT: This function runs *after* the response is sent.
# For production, use Celery/RQ for reliability, scalability, and error handling.
async def process_document_background(
    doc_metadata_id: str,
    temp_file_path: str,
    project_id: str,
    user_id: str,
    filename: str,
    # Pass instances explicitly - avoid relying on globals in background tasks
    db: AsyncIOMotorDatabase,
    doc_processor: DocumentProcessor,
    embedding_gen: EmbeddingGenerator,
    vector_store_mgr: FaissVectorStoreManager
):
    """Background task to process a single uploaded document."""
    logger.info(f"BG Task Started: Processing doc {doc_metadata_id} ('{filename}') for project {project_id}")

    # Check if dependencies were initialized correctly
    if not all([db, doc_processor, embedding_gen, vector_store_mgr]):
         logger.critical(f"BG Task Failed: Dependencies not available for doc {doc_metadata_id}.")
         # Attempt to update status to failed, but DB might also be unavailable
         try:
              await crud.update_document_status(db, doc_id=doc_metadata_id, status="failed", error_message="Internal server error: Processing components unavailable.")
         except Exception as db_err:
              logger.error(f"BG Task Failed: Could not even update status for doc {doc_metadata_id} due to DB error: {db_err}")
         return # Stop processing

    processing_successful = False
    try:
        # 1. Update status to processing
        await crud.update_document_status(db, doc_id=doc_metadata_id, status="processing")
        logger.info(f"BG Task: Status updated to 'processing' for doc {doc_metadata_id}")

        # 2. Process document (load, split)
        logger.info(f"BG Task: Loading and splitting {filename}...")
        chunks = await doc_processor.load_and_split(temp_file_path)
        chunk_count = len(chunks)
        if not chunks:
            # Handle case where document is empty or unsupported after loading
            logger.warning(f"BG Task: Document processing yielded no chunks for {filename} (doc {doc_metadata_id}). Marking as completed with 0 chunks.")
            await crud.update_document_status(db, doc_id=doc_metadata_id, status="completed", chunk_count=0, processed_at=datetime.utcnow())
            processing_successful = True # Consider this 'success' as processing finished
            return # Exit task early

        logger.info(f"BG Task: Processed {filename} into {chunk_count} chunks.")

        # 3. Add documents to vector store (handles embedding)
        logger.info(f"BG Task: Adding {chunk_count} chunks from {filename} to vector store for project {project_id}...")
        await vector_store_mgr.add_documents(
            user_id=user_id,
            project_id=project_id,
            documents=chunks
        )
        logger.info(f"BG Task: Added chunks from {filename} to vector store successfully.")

        # 4. Update status to completed
        await crud.update_document_status(db, doc_id=doc_metadata_id, status="completed", chunk_count=chunk_count)
        logger.info(f"BG Task: Successfully processed document: {doc_metadata_id} ({filename})")
        processing_successful = True

    except FileNotFoundError as fnf_err:
         error_msg = f"BG Task Failed: Temporary file not found for doc {doc_metadata_id}: {fnf_err}"
         logger.error(error_msg)
         await crud.update_document_status(db, doc_id=doc_metadata_id, status="failed", error_message=error_msg)
    except RuntimeError as rt_err: # Catch errors raised by processor/embedder/vector store
         error_msg = f"BG Task Failed: Runtime error processing doc {doc_metadata_id} ('{filename}'): {rt_err}"
         logger.error(error_msg, exc_info=True)
         await crud.update_document_status(db, doc_id=doc_metadata_id, status="failed", error_message=str(rt_err))
    except Exception as e:
        error_msg = f"BG Task Failed: Unexpected error processing doc {doc_metadata_id} ('{filename}'): {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Update status to failed
        try:
            await crud.update_document_status(db, doc_id=doc_metadata_id, status="failed", error_message=error_msg)
        except Exception as db_update_err:
             logger.error(f"BG Task Failed: Could not update status to failed for doc {doc_metadata_id} after error: {db_update_err}")
    finally:
        # 5. Clean up temporary file regardless of success/failure
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"BG Task: Cleaned up temporary file: {temp_file_path}")
        except OSError as e:
            # Log error but don't fail the task just for cleanup failure
            logger.error(f"BG Task: Error cleaning up temporary file {temp_file_path}: {e}")
        logger.info(f"BG Task Finished for doc: {doc_metadata_id} ({filename}). Success: {processing_successful}")


# --- Endpoints ---

@router.post("/{project_id}/upload", response_model=List[DocumentMetadataResponse], status_code=status.HTTP_202_ACCEPTED)
async def upload_project_documents(
    project_id: str, # Get project_id from path
    files: List[UploadFile] = File(..., description="One or more files to upload for the project."),
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_active_user),
    project: ProjectInDB = Depends(get_project_for_user), # Use dependency to get & verify project
    background_tasks: BackgroundTasks = None # BackgroundTasks at the end with default None
):
    """
    Uploads one or more documents (PDF, DOCX, TXT, MD) to a specific project.

    Initiates background processing for each file and returns immediately
    with the initial metadata record(s) marked as 'pending'. Check the status
    endpoint (`/projects/{project_id}/documents`) to monitor progress.
    """
    # Check if processing components are available
    if not all([doc_processor_instance, embedding_gen_doc, vector_store_mgr_doc]):
         logger.critical("Document processing components are not available. Upload endpoint cannot function.")
         raise HTTPException(
              status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
              detail="Document processing service is currently unavailable."
         )

    logger.info(f"Received {len(files)} file(s) for upload to project {project_id} by user {current_user.email}")
    # Project existence/ownership verified by get_project_for_user dependency

    created_metadata_list = []
    processed_files_count = 0
    for file in files:
        if not file.filename:
             logger.warning("Received file upload without filename. Skipping.")
             continue

        # Basic check for allowed extensions (optional, but good practice)
        allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"} # Add .xlsx etc. if supported
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
             logger.warning(f"Skipping file '{file.filename}' due to unsupported extension '{file_ext}'.")
             # Optionally raise 400 Bad Request or just skip
             continue

        # Sanitize filename and create unique temporary path
        # Using UUID ensures uniqueness even if filenames are the same
        safe_filename_base = "".join(c for c in os.path.splitext(file.filename)[0] if c.isalnum() or c in ('-', '_'))
        unique_suffix = str(uuid.uuid4())[:8]
        temp_filename = f"{safe_filename_base}_{unique_suffix}{file_ext}"
        temp_file_path = os.path.join(settings.TEMP_UPLOAD_DIR, temp_filename)

        # 2. Save uploaded file temporarily (asynchronously)
        try:
            logger.debug(f"Saving uploaded file '{file.filename}' to '{temp_file_path}'")
            # Use async file writing if possible, or run sync in executor
            # For simplicity, using sync shutil within a standard file open context
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer) # file.file is a SpooledTemporaryFile (sync)
            logger.info(f"Temporarily saved uploaded file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to save uploaded file {file.filename}: {e}", exc_info=True)
            # Decide how to handle partial failures if multiple files are uploaded
            # Option 1: Stop and raise error for the first failure
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not save file: {file.filename}")
            # Option 2: Log error, skip this file, and continue with others (more complex response needed)
            # continue
        finally:
            # Ensure the file handle provided by FastAPI is closed
            await file.close() # Use await as file.close() might be async

        # 3. Create initial metadata entry in DB
        doc_meta = DocumentMetadataInDB(
            project_id=project.id, # Use verified project ID
            user_id=current_user.id,
            filename=file.filename, # Store original filename for display
            original_filepath=temp_file_path, # Store temp path for background task
            status="pending",
            uploaded_at=datetime.utcnow() # Set upload time explicitly
        )
        try:
            created_meta_db = await crud.create_document_metadata(db, doc_meta)
            created_metadata_list.append(DocumentMetadataResponse(**created_meta_db.model_dump(by_alias=True)))
            processed_files_count += 1
        except Exception as e:
             logger.error(f"Failed to create database metadata for file {file.filename}: {e}", exc_info=True)
             # Clean up the saved temp file if DB entry fails
             if os.path.exists(temp_file_path): os.remove(temp_file_path)
             # Decide how to handle: stop all, or skip this file
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not create metadata for file: {file.filename}")
             # continue

        # 4. Add background task for processing this specific file
        # Pass necessary dependencies to the background task function
        background_tasks.add_task(
            process_document_background,
            doc_metadata_id=created_meta_db.id, # Pass the ID of the DB record
            temp_file_path=temp_file_path,
            project_id=project.id,
            user_id=current_user.id,
            filename=file.filename,
            # Pass instances needed by the task
            db=db,
            doc_processor=doc_processor_instance,
            embedding_gen=embedding_gen_doc,
            vector_store_mgr=vector_store_mgr_doc
        )
        logger.info(f"Scheduled background processing for: {file.filename} (Doc ID: {created_meta_db.id})")

    if processed_files_count == 0:
         # This happens if all files were skipped (e.g., wrong extension, no filename)
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid files were provided or accepted for upload.")

    # Return 202 Accepted with the initial metadata for successfully scheduled files
    logger.info(f"Upload request completed for {processed_files_count} file(s) for project {project_id}. Processing initiated.")
    return created_metadata_list


@router.get("/{project_id}/documents", response_model=List[DocumentMetadataResponse])
async def get_project_document_status(
    project_id: str, # Get project_id from path
    db: AsyncIOMotorDatabase = Depends(get_database),
    project: ProjectInDB = Depends(get_project_for_user), # Use dependency to get & verify project
    current_user: UserInDB = Depends(get_current_active_user) # Included via project dependency
):
    """
    Get the status and metadata of all documents uploaded to a specific project.
    """
    logger.info(f"Fetching document status for project {project_id} for user {current_user.email}")
    # Project existence/ownership verified by get_project_for_user dependency

    # 2. Fetch document metadata from DB
    docs_db = await crud.get_documents_by_project(db=db, project_id=project.id, user_id=current_user.id)
    logger.info(f"Found {len(docs_db)} document metadata entries for project {project_id}")
    return [DocumentMetadataResponse(**d.model_dump(by_alias=True)) for d in docs_db]

# Optional: Endpoint to get status of a single document
@router.get("/{project_id}/documents/{document_id}", response_model=DocumentMetadataResponse)
async def get_single_document_status(
    project_id: str,
    document_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    project: ProjectInDB = Depends(get_project_for_user), # Verify project access
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Get the status and metadata of a single document within a project.
    """
    logger.info(f"Fetching status for document {document_id} in project {project_id}")
    doc_meta = await crud.get_document_metadata(db, doc_id=document_id, project_id=project.id, user_id=current_user.id)
    if not doc_meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found in this project")
    return DocumentMetadataResponse(**doc_meta.model_dump(by_alias=True))
