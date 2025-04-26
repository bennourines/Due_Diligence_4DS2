# filepath: DeployTrial2/models/db_models.py
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Union # Added Union
from datetime import datetime
import uuid

# Using Pydantic models directly with Motor is often convenient
# You might add custom methods or inherit if needed

class UserInDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    email: EmailStr = Field(..., unique=True) # Motor doesn't enforce unique directly, handle in code/index
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "user-uuid-123",
                "email": "user@example.com",
                "hashed_password": "bcrypt_hash_string",
                "created_at": "2025-04-23T10:00:00Z"
            }
        }


class ProjectInDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    user_id: str # Reference to UserInDB.id
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "project-uuid-456",
                "user_id": "user-uuid-123",
                "name": "Project Alpha",
                "description": "Due diligence on AlphaCoin",
                "created_at": "2025-04-23T10:05:00Z"
            }
        }

class DocumentMetadataInDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    project_id: str # Reference to ProjectInDB.id
    user_id: str # Reference to UserInDB.id
    filename: str
    original_filepath: Optional[str] = None # Path where it was initially saved
    status: str = "pending" # "pending", "processing", "completed", "failed"
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "doc-uuid-789",
                "project_id": "project-uuid-456",
                "user_id": "user-uuid-123",
                "filename": "whitepaper.pdf",
                "original_filepath": "temp_uploads/unique_whitepaper.pdf",
                "status": "completed",
                "uploaded_at": "2025-04-23T10:10:00Z",
                "processed_at": "2025-04-23T10:12:00Z",
                "chunk_count": 55,
                "error_message": None
            }
        }

# Define content models matching api_models for consistency within DB history
class QueryHistoryContentInDB(BaseModel):
    query: str
    response: str
    retrieved_sources: Optional[List[str]] = None

class AnalysisHistoryContentInDB(BaseModel):
     report: str # Or structured report

class ConversationHistoryInDB(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    project_id: str
    user_id: str
    type: str # "query" or "analysis"
    # Use Union for content based on type
    content: Union[QueryHistoryContentInDB, AnalysisHistoryContentInDB]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "hist-uuid-012",
                "project_id": "project-uuid-456",
                "user_id": "user-uuid-123",
                "type": "query",
                "content": {
                    "query": "What is the tokenomics model?",
                    "response": "The tokenomics model involves...",
                    "retrieved_sources": ["whitepaper.pdf"]
                },
                "timestamp": "2025-04-23T10:15:00Z"
            }
        }
