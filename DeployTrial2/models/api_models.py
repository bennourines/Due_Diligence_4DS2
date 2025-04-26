# filepath: DeployTrial2/models/api_models.py
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Union # Added Union
from datetime import datetime

# --- Auth ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserInDBBase(BaseModel):
    id: str = Field(alias="_id")
    email: EmailStr

    class Config:
        populate_by_name = True
        from_attributes = True # For ORM mode if needed later

# --- Projects ---
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True

# --- Documents ---
class DocumentMetadataResponse(BaseModel):
    id: str = Field(alias="_id")
    project_id: str
    user_id: str
    filename: str
    status: str # e.g., "pending", "processing", "completed", "failed"
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True
        from_attributes = True

# --- Query/Analysis ---
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5 # Number of chunks to retrieve

class QueryResponse(BaseModel):
    project_id: str
    query: str
    response: str
    retrieved_sources: Optional[List[str]] = None # List of source filenames
    timestamp: datetime

class AnalysisResponse(BaseModel):
    project_id: str
    report: str # Or could be a structured Pydantic model
    generated_at: datetime

# --- History ---
# Define a base class or individual models if structure differs significantly
class BaseHistoryContent(BaseModel):
    project_id: str
    timestamp: datetime

class QueryHistoryContent(BaseHistoryContent):
    query: str
    response: str
    retrieved_sources: Optional[List[str]] = None

class AnalysisHistoryContent(BaseHistoryContent):
     report: str # Or structured report

class HistoryEntry(BaseModel):
    id: str = Field(alias="_id") # Add id field
    project_id: str
    user_id: str
    type: str # "query" or "analysis"
    # Use Union for content based on type
    content: Union[QueryHistoryContent, AnalysisHistoryContent]
    timestamp: datetime

    class Config:
        populate_by_name = True
        from_attributes = True
