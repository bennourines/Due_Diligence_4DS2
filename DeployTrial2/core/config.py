# filepath: DeployTrial2/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file variables
load_dotenv()

class Settings(BaseSettings):
    # Application Settings
    PROJECT_NAME: str = "Crypto Due Diligence API"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # Database Settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "crypto_diligence_db")

    # JWT Settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "supersecretkey") # CHANGE THIS!
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # 1 day

    # LLM & Embeddings Settings
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "meta-llama/llama-4-maverick:free")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu") # or 'cuda' if available

    # File Storage Settings
    TEMP_UPLOAD_DIR: str = "temp_uploads"
    VECTOR_STORE_BASE_PATH: str = "vector_stores" # Base path for FAISS indices

    # Task Queue (Example - if using Celery)
    # CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    # CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    class Config:
        case_sensitive = True
        # Optional: Load from .env file if pydantic-settings supports it directly
        # env_file = ".env"

settings = Settings()

# Ensure necessary directories exist
os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_BASE_PATH, exist_ok=True)
