# # frontend/utils/api_client.py
import requests
import streamlit as st
import logging
from typing import List, Dict, Any, Optional

from frontend.config import (
    BACKEND_API_URL, API_V1_PREFIX,
    SESSION_STATE_KEY_AUTH_TOKEN
)

logger = logging.getLogger(__name__)

# --- Base API Request Function ---

def make_api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    is_form_data: bool = False
) -> requests.Response:
    """
    Makes a request to the backend API.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE).
        endpoint: API endpoint path (e.g., "/auth/login").
        data: JSON payload for POST/PUT requests.
        params: URL parameters for GET requests.
        files: Dictionary of files for multipart/form-data uploads.
        token: JWT authentication token.
        is_form_data: Set True if 'data' should be sent as form data instead of JSON.

    Returns:
        requests.Response object.
    """
    base_url = f"{BACKEND_API_URL}{API_V1_PREFIX}"
    url = f"{base_url}{endpoint}"
    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=60)
        elif method.upper() == "POST":
            if files:
                # Let requests handle Content-Type for multipart/form-data
                response = requests.post(url, headers=headers, files=files, data=data, timeout=300) # Longer timeout for uploads
            elif is_form_data:
                # Send data as x-www-form-urlencoded
                response = requests.post(url, headers=headers, data=data, timeout=60)
            else:
                # Default to JSON
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, json=data, timeout=120) # Longer timeout for potential LLM calls
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeout=60)
        elif method.upper() == "PUT": # Add PUT if needed later
             headers["Content-Type"] = "application/json"
             response = requests.put(url, headers=headers, json=data, timeout=60)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            logger.error(f"Unsupported HTTP method: {method}")
            return None # Or raise error

        # Log request/response details (optional, can be verbose)
        # logger.debug(f"API Request: {method} {url} -> Status: {response.status_code}")
        # if response.status_code >= 400:
        #      logger.warning(f"API Error Response Body: {response.text[:500]}") # Log first 500 chars of error

        return response

    except requests.exceptions.ConnectionError as e:
        st.error(f"Connection Error: Could not connect to the backend at {base_url}. Please ensure it's running.")
        logger.error(f"API ConnectionError to {url}: {e}")
        return None
    except requests.exceptions.Timeout as e:
         st.error(f"Request Timed Out: The backend took too long to respond.")
         logger.error(f"API Timeout for {method} {url}: {e}")
         return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during the API request: {e}")
        logger.error(f"API RequestException for {method} {url}: {e}", exc_info=True)
        return None

# --- Specific API Call Functions ---

# Authentication
def api_register(email: str, password: str) -> requests.Response:
    return make_api_request("POST", "/auth/register", data={"email": email, "password": password})

def api_login(email: str, password: str) -> requests.Response:
    # Login uses form data
    return make_api_request("POST", "/auth/login", data={"username": email, "password": password}, is_form_data=True)

def api_get_me(token: str) -> requests.Response:
    return make_api_request("GET", "/auth/me", token=token)

# Projects
def api_create_project(name: str, description: Optional[str], token: str) -> requests.Response:
    return make_api_request("POST", "/projects/", data={"name": name, "description": description}, token=token)

def api_get_projects(token: str) -> requests.Response:
    return make_api_request("GET", "/projects/", token=token)

def api_get_project_details(project_id: str, token: str) -> requests.Response:
     return make_api_request("GET", f"/projects/{project_id}", token=token)

def api_delete_project(project_id: str, token: str) -> requests.Response:
    return make_api_request("DELETE", f"/projects/{project_id}", token=token)

# Documents
def api_upload_documents(project_id: str, files: List[Any], token: str) -> requests.Response:
    # Prepare files for requests library format: list of ('files', (filename, file_obj, content_type)) tuples
    upload_files = [("files", (file.name, file, file.type)) for file in files]
    return make_api_request("POST", f"/projects/{project_id}/upload", files=upload_files, token=token)

def api_get_documents_status(project_id: str, token: str) -> requests.Response:
    return make_api_request("GET", f"/projects/{project_id}/documents", token=token)

# Query & History
def api_post_query(project_id: str, query: str, top_k: int, token: str) -> requests.Response:
    return make_api_request("POST", f"/projects/{project_id}/query", data={"query": query, "top_k": top_k}, token=token)

def api_get_history(project_id: str, token: str) -> requests.Response:
    return make_api_request("GET", f"/projects/{project_id}/history", token=token)

# --- Helper to get token from session state ---
def get_auth_token():
     return st.session_state.get(SESSION_STATE_KEY_AUTH_TOKEN)

#explanation='Create API client utility for frontend-backend communication.

