from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class UserQuery(BaseModel):
    """User query model"""
    project_id: str
    user_id: str = "anonymous"
    query: str

class ChatMessage(BaseModel):
    """Chat message model"""
    project_id: str
    user_id: str
    query: str
    response: str
    timestamp: datetime

class RiskReport(BaseModel):
    """Risk report model"""
    summary: str
    overall_score: float
    categories: Dict[str, Any]
    recommendation: str
    key_findings: List[str]