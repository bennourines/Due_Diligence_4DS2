#\frontend\\utils\\auth.py
import streamlit as st
import logging
from frontend.utils.api_client import api_login, api_register, api_get_me
from frontend.config import (
    SESSION_STATE_KEY_AUTH_TOKEN, SESSION_STATE_KEY_CHAT_HISTORY, SESSION_STATE_KEY_DOCUMENTS, SESSION_STATE_KEY_PROJECT_LIST, SESSION_STATE_KEY_SELECTED_PROJECT_ID, SESSION_STATE_KEY_SELECTED_PROJECT_NAME, SESSION_STATE_KEY_USER_EMAIL,
    SESSION_STATE_KEY_LOGGED_IN
)

logger = logging.getLogger(__name__)

def handle_login(email, password):
    """Attempts to log in the user and updates session state."""
    response = api_login(email, password)
    if response and response.status_code == 200:
        token_data = response.json()
        st.session_state[SESSION_STATE_KEY_AUTH_TOKEN] = token_data.get("access_token")
        st.session_state[SESSION_STATE_KEY_USER_EMAIL] = email # Store email used for login
        st.session_state[SESSION_STATE_KEY_LOGGED_IN] = True
        logger.info(f"User '{email}' logged in successfully.")
        st.success("Login successful!")
        st.rerun() # Rerun to navigate to the main app page
    elif response:
        error_detail = response.json().get("detail", "Unknown login error")
        st.error(f"Login Failed: {error_detail}")
        logger.warning(f"Login failed for user '{email}': {error_detail} (Status: {response.status_code})")
    else:
        # Error handled by make_api_request (e.g., connection error)
        pass # Error message already shown by api_client

def handle_signup(email, password):
    """Attempts to register a new user."""
    response = api_register(email, password)
    if response and response.status_code == 201:
        st.success("Registration successful! You can now log in.")
        logger.info(f"User '{email}' registered successfully.")
        return True # Indicate success
    elif response:
        error_detail = response.json().get("detail", "Unknown registration error")
        st.error(f"Registration Failed: {error_detail}")
        logger.warning(f"Registration failed for user '{email}': {error_detail} (Status: {response.status_code})")
    else:
        # Error handled by make_api_request
        pass
    return False # Indicate failure

def handle_logout():
    """Logs out the user by clearing relevant session state."""
    logged_out_user = st.session_state.get(SESSION_STATE_KEY_USER_EMAIL, "Unknown user")
    keys_to_clear = [
        SESSION_STATE_KEY_AUTH_TOKEN, SESSION_STATE_KEY_USER_EMAIL,
        SESSION_STATE_KEY_LOGGED_IN, SESSION_STATE_KEY_PROJECT_LIST,
        SESSION_STATE_KEY_SELECTED_PROJECT_ID, SESSION_STATE_KEY_SELECTED_PROJECT_NAME,
        SESSION_STATE_KEY_DOCUMENTS, SESSION_STATE_KEY_CHAT_HISTORY
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    logger.info(f"User '{logged_out_user}' logged out.")
    st.success("You have been logged out.")
    st.rerun() # Rerun to go back to login page

def verify_token_and_get_user(token):
     """Verifies token with /auth/me endpoint."""
     response = api_get_me(token)
     if response and response.status_code == 200:
          user_data = response.json()
          # Store/update email just in case it changed somehow (unlikely with JWT sub)
          st.session_state[SESSION_STATE_KEY_USER_EMAIL] = user_data.get("email")
          st.session_state[SESSION_STATE_KEY_LOGGED_IN] = True
          return True
     else:
          # Token is invalid or expired, or backend error
          st.session_state[SESSION_STATE_KEY_LOGGED_IN] = False
          st.session_state.pop(SESSION_STATE_KEY_AUTH_TOKEN, None)
          st.session_state.pop(SESSION_STATE_KEY_USER_EMAIL, None)
          logger.warning("Token verification failed or token expired.")
          return False
