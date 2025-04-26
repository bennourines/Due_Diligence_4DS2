# frontend/utils/state.py
import streamlit as st
from frontend.config import * # Import all config keys

def initialize_session_state():
    """Initializes required keys in Streamlit's session state if they don't exist."""
    defaults = {
        SESSION_STATE_KEY_LOGGED_IN: False,
        SESSION_STATE_KEY_AUTH_TOKEN: None,
        SESSION_STATE_KEY_USER_EMAIL: None,
        SESSION_STATE_KEY_PROJECT_LIST: [],
        SESSION_STATE_KEY_SELECTED_PROJECT_ID: None,
        SESSION_STATE_KEY_SELECTED_PROJECT_NAME: None,
        SESSION_STATE_KEY_DOCUMENTS: [],
        SESSION_STATE_KEY_CHAT_HISTORY: [] # List of {"role": "user/assistant", "content": "..."}
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_project_state():
     """Resets state related to a specific project when switching or logging out."""
     st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_ID] = None
     st.session_state[SESSION_STATE_KEY_SELECTED_PROJECT_NAME] = None
     st.session_state[SESSION_STATE_KEY_DOCUMENTS] = []
     st.session_state[SESSION_STATE_KEY_CHAT_HISTORY] = []