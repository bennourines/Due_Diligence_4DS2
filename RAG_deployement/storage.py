import json
import os
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FileStorage:
    """Simple file-based storage for documents and conversations"""
    
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = storage_dir
        self.documents_file = os.path.join(storage_dir, "documents.json")
        self.conversations_file = os.path.join(storage_dir, "conversations.json")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        if not os.path.exists(self.documents_file):
            self._save_documents({})
        if not os.path.exists(self.conversations_file):
            self._save_conversations({})
    
    def _save_documents(self, documents: Dict):
        """Save documents to file"""
        with open(self.documents_file, 'w') as f:
            json.dump(documents, f, default=str)
    
    def _save_conversations(self, conversations: Dict):
        """Save conversations to file"""
        with open(self.conversations_file, 'w') as f:
            json.dump(conversations, f, default=str)
    
    def _load_documents(self) -> Dict:
        """Load documents from file"""
        try:
            with open(self.documents_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _load_conversations(self) -> Dict:
        """Load conversations from file"""
        try:
            with open(self.conversations_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def store_document(self, project_id: str, document_data: Dict):
        """Store document metadata"""
        documents = self._load_documents()
        documents[project_id] = document_data
        self._save_documents(documents)
    
    def store_conversation(self, project_id: str, message: Dict):
        """Store conversation message"""
        conversations = self._load_conversations()
        if project_id not in conversations:
            conversations[project_id] = []
        conversations[project_id].append(message)
        self._save_conversations(conversations)
    
    def get_chat_history(self, project_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a project"""
        conversations = self._load_conversations()
        project_conversations = conversations.get(project_id, [])
        return sorted(project_conversations, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_document(self, project_id: str) -> Dict:
        """Get document metadata"""
        documents = self._load_documents()
        return documents.get(project_id, {}) 