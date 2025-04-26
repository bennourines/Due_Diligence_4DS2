# frontend/pages/2_🚀_Project_Workspace.py
import streamlit as st
import pandas as pd
import time
import logging
from datetime import datetime

from frontend.utils.api_client import (
    api_create_project, api_get_projects, api_delete_project,
    api_upload_documents, api_get_documents_status,
    api_post_query, api_get_history, get_auth_token
)
from frontend.utils.auth import handle_logout, verify_token_and_get_user
from frontend.utils.state import reset_project_state
from frontend.config import (
    APP_TITLE, PAGE_ICON, LAYOUT,
    SESSION_STATE_KEY_LOGGED_IN, SESSION_STATE_KEY_AUTH_TOKEN,
    SESSION_STATE_KEY_USER_EMAIL, SESSION_STATE_KEY_PROJECT_LIST,
    SESSION_STATE_KEY_SELECTED_PROJECT_ID, SESSION_STATE_KEY_SELECTED_PROJECT_NAME,
    SESSION_STATE_KEY_DOCUMENTS, SESSION_STATE_KEY_CHAT_HISTORY,
    DOCUMENT_STATUS_REFRESH_INTERVAL, MAX_UPLOAD_FILES, MAX_UPLOAD_SIZE_MB
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# --- Authentication Check ---
if not st.session_state.get(SESSION_STATE_KEY_LOGGED_IN, False):
    st.warning("Please log in to access the workspace.")
    st.switch_page("pages/1_🔐_Login_Signup.py")
    st.stop()

# Verify token on page load (handles expired tokens)
auth_token = st.session_state.get(SESSION_STATE_KEY_AUTH_TOKEN)
if not auth_token or not verify_token_and_get_user(auth_token):
     st.error("Your session has expired or is invalid. Please log in again.")
     # Clear potentially stale state before redirecting
     reset_project_state()
     st.session_state[SESSION_STATE_KEY_LOGGED_IN] = False
     st.session_state.pop(SESSION_STATE_KEY_AUTH_TOKEN, None)
     st.session_state.pop(SESSION_STATE_KEY_USER_EMAIL, None)
     time.sleep(2) # Give user time to see message
     st.switch_page("pages/1_🔐_Login_Signup.py")
     st.stop()

# --- Helper Functions ---
def load_projects():
    """Fetches projects from backend and updates session state."""
    token = get_auth_token()
    if not token: return
    response = api_get_projects(token)
    if response and response.status_code == 200:
        st.session_state[SESSION_STATE_KEY_PROJECT_LIST] = response.json()
    elif response:
        st.error(f"Failed to load projects: {response.json().get('detail', 'Unknown error')}")
        st.session_state[SESSION_STATE_KEY_PROJECT_LIST] = []
    else:
        st.session_state[SESSION_STATE_KEY_PROJECT_LIST] = [] # API client handles connection errors

def load_documents(project_id):
    """Fetches document status for the selected project."""
    token = get_auth_token()
    if not token or not project_id: return
    response = api_get_documents_status(project_id, token)
    if response and response.status_code == 200:
        st.session_state[SESSION_STATE_KEY_DOCUMENTS] = response.json()
    elif response:
        st.error(f"Failed to load documents: {response.json().get('detail', 'Unknown error')}")
        st.session_state[SESSION_STATE_KEY_DOCUMENTS] = []
    else:
        st.session_state[SESSION_STATE_KEY_DOCUMENTS] = []

def load_chat_history(project_id):
     """Fetches chat history for the selected project."""
     token = get_auth_token()
     if not token or not project_id: return
     response = api_get_history(project_id, token)
     if response and response.status_code == 200:
          history_data = response.json()
          # Convert backend history format to Streamlit chat format
          chat_messages = []
          for entry in history_data:
               if entry.get("type") == "query" and entry.get("content"):
                    content = entry["content"]
                    if content.get("query"):
                         chat_messages.append({"role": "user", "content": content["query"]})
                    if content.get("response"):
                         # Add source info to assistant message if available
                         sources_text = ""
                         if content.get("retrieved_sources"):
                              sources_text = "\\n\\n*Sources: " + ", ".join(content["retrieved_sources"]) + "*"
                         chat_messages.append({"role": "assistant", "content": content["response"] + sources_text})
               # Add handling for 'analysis' type if implemented later
          st.session_state[SESSION_STATE_KEY_CHAT_HISTORY] = chat_messages
     elif response:
          st.error(f"Failed to load chat history: {response.json().get('detail', 'Unknown error')}")
          st.session_state[SESSION_STATE_KEY_CHAT_HISTORY] = []
     else:
          st.session_state[SESSION_STATE_KEY_CHAT_HISTORY] = []


# --- Initial Data Loading ---
if SESSION_STATE_KEY_PROJECT_LIST not in st.session_state or not st.session_state[SESSION_STATE_KEY_PROJECT_LIST]:
    load_projects()

# --- Sidebar ---
with st.sidebar:
    st.title(APP_TITLE)
    st.write(f"Welcome, {st.session_state.get(SESSION_STATE_KEY_USER_EMAIL, 'User')}!")
    st.divider()

    # Project Selection / Creation
    st.subheader("Projects")
    projects = st.session_state.get(SESSION_STATE_KEY_PROJECT_LIST, [])
    project_names = {p['id']: p['name'] for p in projects}

    # Use index to manage selectbox state properly after creation/deletion
    project_ids = list(project_names.keys())
    current_selection_id = st.session_state.get(SESSION_STATE_KEY_SELECTED_PROJECT_ID)
    try:
        current_index = project_ids.index(current_selection_id) if current_selection_id in project_ids else 0
    except ValueError:
        current_index = 0 # Default to first project if previous selection is gone

    if project_ids:
         selected_id = st.selectbox(
              "Select Project",
              project_ids,
              format_func=lambda pid: project_names.get(pid, "Unknown Project"),
              index=current_index,
              key="project_selector" # Give it a key
         )
         # Update session state if selection changes
         if selected_id != st.session_state.get(SESSION_STATE_KEY_SELECTED_PROJECT_ID):
              st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_ID] = selected_id
              st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_NAME] = project_names.get(selected_id)
              # Reset documents and chat history for the new project
              st.session_state[SESSION_STATE_KEY_DOCUMENTS] = []
              st.session_state[SESSION_STATE_KEY_CHAT_HISTORY] = []
              st.rerun() # Rerun to load data for the new project

         # Load data if project selected but data not loaded yet
         if selected_id and not st.session_state.get(SESSION_STATE_KEY_DOCUMENTS):
              load_documents(selected_id)
         if selected_id and not st.session_state.get(SESSION_STATE_KEY_CHAT_HISTORY):
              load_chat_history(selected_id)

    else:
         st.caption("No projects found. Create one below.")
         st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_ID] = None # Ensure reset if list becomes empty
         st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_NAME] = None

    with st.expander("Create New Project"):
        with st.form("new_project_form"):
            new_project_name = st.text_input("Project Name")
            new_project_desc = st.text_area("Description (Optional)")
            create_button = st.form_submit_button("Create Project")
            if create_button and new_project_name:
                token = get_auth_token()
                if token:
                    response = api_create_project(new_project_name, new_project_desc, token)
                    if response and response.status_code == 201:
                        st.success(f"Project '{new_project_name}' created!")
                        load_projects() # Refresh project list
                        # Automatically select the new project
                        new_project_id = response.json().get("id")
                        st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_ID] = new_project_id
                        st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_NAME] = new_project_name
                        st.session_state[SESSION_STATE_KEY_DOCUMENTS] = []
                        st.session_state[SESSION_STATE_KEY_CHAT_HISTORY] = []
                        st.rerun()
                    elif response:
                         st.error(f"Failed to create project: {response.json().get('detail', 'Unknown error')}")
            elif create_button:
                 st.warning("Project name cannot be empty.")

    st.divider()
    if st.button("Logout", type="primary"):
        handle_logout()

# --- Main Content Area ---
selected_project_id = st.session_state.get(SESSION_STATE_KEY_SELECTED_PROJECT_ID)
selected_project_name = st.session_state.get(SESSION_STATE_KEY_SELECTED_PROJECT_NAME, "No Project Selected")

st.header(f"Workspace: {selected_project_name}")

if not selected_project_id:
    st.info("Select a project from the sidebar or create a new one to get started.")
    st.stop()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "🗑️ Project Settings"])

with tab1:
    st.subheader("Chat with your Documents")

    # Display existing chat messages
    for message in st.session_state.get(SESSION_STATE_KEY_CHAT_HISTORY, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the project documents..."):
        token = get_auth_token()
        if not token:
             st.error("Authentication token not found. Please log in again.")
             st.stop()

        # Add user message to chat history and display it
        st.session_state[SESSION_STATE_KEY_CHAT_HISTORY].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send query to backend and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            response = api_post_query(selected_project_id, prompt, top_k=5, token=token) # Make top_k configurable?

            if response and response.status_code == 200:
                query_response = response.json()
                answer = query_response.get("response", "No answer received.")
                sources = query_response.get("retrieved_sources", [])
                full_response = answer
                if sources:
                    sources_text = ", ".join(sources)
                    full_response += f"\\n\\n*Sources: {sources_text}*"

                message_placeholder.markdown(full_response)
                # Add assistant response to history
                st.session_state[SESSION_STATE_KEY_CHAT_HISTORY].append({"role": "assistant", "content": full_response})
            elif response:
                 error_detail = response.json().get("detail", "Unknown error")
                 message_placeholder.error(f"Error getting response: {error_detail}")
                 st.session_state[SESSION_STATE_KEY_CHAT_HISTORY].append({"role": "assistant", "content": f"Error: {error_detail}"})
            else:
                 # Network or connection error handled by api_client
                 message_placeholder.error("Failed to connect to the backend.")
                 st.session_state[SESSION_STATE_KEY_CHAT_HISTORY].append({"role": "assistant", "content": "Error: Could not reach backend."})


with tab2:
    st.subheader("Manage Documents")

    # File Upload
    with st.expander("Upload New Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Choose files (.pdf, .docx, .txt, .md)",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md"], # Add more types if backend supports them
            key="file_uploader"
        )

        if st.button("Upload and Process"):
            token = get_auth_token()
            if not uploaded_files:
                st.warning("Please select files to upload.")
            elif not token:
                 st.error("Authentication token not found. Please log in again.")
            else:
                 # Basic size check (Streamlit also has server.maxUploadSize config)
                 total_size = sum(f.size for f in uploaded_files)
                 if len(uploaded_files) > MAX_UPLOAD_FILES:
                      st.error(f"You can upload a maximum of {MAX_UPLOAD_FILES} files at a time.")
                 elif total_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                      st.error(f"Total upload size exceeds {MAX_UPLOAD_SIZE_MB} MB limit.")
                 else:
                      with st.spinner(f"Uploading {len(uploaded_files)} file(s)..."):
                           response = api_upload_documents(selected_project_id, uploaded_files, token)
                           if response and response.status_code == 202: # 202 Accepted
                                st.success(f"{len(uploaded_files)} file(s) uploaded successfully! Processing started in the background.")
                                # Clear the uploader and refresh document list after a short delay
                                st.session_state["file_uploader"] = [] # Clear uploaded files visually
                                time.sleep(1) # Short delay before refresh
                                load_documents(selected_project_id)
                                st.rerun()
                           elif response:
                                st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                           # else: API client handles connection error

    st.divider()

    # Document Status Table
    st.subheader("Uploaded Documents Status")
    if st.button("Refresh Status"):
        load_documents(selected_project_id)

    documents = st.session_state.get(SESSION_STATE_KEY_DOCUMENTS, [])

    if not documents:
        st.info("No documents uploaded for this project yet.")
    else:
        # Prepare data for display
        doc_data = []
        processing_ongoing = False
        for doc in documents:
            status = doc.get('status', 'Unknown')
            if status in ["pending", "processing"]:
                 processing_ongoing = True
            # Format timestamps nicely
            uploaded_at = datetime.fromisoformat(doc.get('uploaded_at')).strftime('%Y-%m-%d %H:%M:%S') if doc.get('uploaded_at') else '-'
            processed_at = datetime.fromisoformat(doc.get('processed_at')).strftime('%Y-%m-%d %H:%M:%S') if doc.get('processed_at') else '-'

            doc_data.append({
                "Filename": doc.get('filename', 'N/A'),
                "Status": status.capitalize(),
                "Chunks": doc.get('chunk_count', '-'),
                "Uploaded At": uploaded_at,
                "Processed At": processed_at,
                "Error": doc.get('error_message', '')
            })

        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)

        # Auto-refresh if processing is ongoing
        if processing_ongoing:
             st.caption(f"Status will auto-refresh in {DOCUMENT_STATUS_REFRESH_INTERVAL} seconds...")
             time.sleep(DOCUMENT_STATUS_REFRESH_INTERVAL)
             load_documents(selected_project_id)
             st.rerun()


with tab3:
    st.subheader("Project Settings")
    st.warning("🚨 Deleting a project is irreversible and will remove all associated documents and chat history.", icon="⚠️")

    if st.button(f"Delete Project '{selected_project_name}'", type="primary"):
        token = get_auth_token()
        if token:
            st.markdown(f"**Are you sure you want to delete project '{selected_project_name}'?**")
            col1, col2, _ = st.columns([1,1,5])
            with col1:
                if st.button("Yes, Delete Permanently", key="confirm_delete"):
                     with st.spinner("Deleting project..."):
                          response = api_delete_project(selected_project_id, token)
                          if response and response.status_code == 204:
                               st.success(f"Project '{selected_project_name}' deleted successfully.")
                               reset_project_state()
                               load_projects() # Refresh list
                               time.sleep(2)
                               st.rerun() # Rerun to reflect changes in sidebar/main area
                          elif response:
                               st.error(f"Failed to delete project: {response.json().get('detail', 'Unknown error')}")
                          # else: API client handles connection error
            with col2:
                 if st.button("Cancel", key="cancel_delete"):
                      st.info("Deletion cancelled.")
                      time.sleep(1)
                      st.rerun() # Rerun to clear the confirmation buttons