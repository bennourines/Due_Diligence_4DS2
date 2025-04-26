#frontend/config.py
import os

# --- Backend API Configuration ---
# Use environment variable if set, otherwise default to localhost
# Ensure your FastAPI backend is running at this address
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
API_V1_PREFIX = "/api/v1"

# --- UI Configuration ---
APP_TITLE = "Crypto Due Diligence Assistant"
PAGE_ICON = "🪙"
LAYOUT = "wide" # "centered" or "wide"

# --- Styling (Optional) ---
# Example: Load custom CSS
# CUSTOM_CSS_PATH = "style.css"

# --- Session State Keys ---
# Keys used to store data in Streamlit's session state
SESSION_STATE_KEY_AUTH_TOKEN = "auth_token"
SESSION_STATE_KEY_USER_EMAIL = "user_email"
SESSION_STATE_KEY_LOGGED_IN = "logged_in"
SESSION_STATE_KEY_PROJECT_LIST = "project_list"
SESSION_STATE_KEY_SELECTED_PROJECT_ID = "selected_project_id"
SESSION_STATE_KEY_SELECTED_PROJECT_NAME = "selected_project_name"
SESSION_STATE_KEY_DOCUMENTS = "documents"
SESSION_STATE_KEY_CHAT_HISTORY = "chat_history" # For the current project

# --- Other Settings ---
DOCUMENT_STATUS_REFRESH_INTERVAL = 10 # Seconds to wait before auto-refreshing doc status
MAX_UPLOAD_FILES = 10 # Max number of files allowed in a single upload
MAX_UPLOAD_SIZE_MB = 200 # Max total upload size in MB (Streamlit has its own limits too)
# explanation='Create frontend configuration file.'

