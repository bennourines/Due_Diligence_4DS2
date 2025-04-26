# frontend/app.py
import streamlit as st
import logging
import sys
import os

# Add project root to sys.path to allow imports from frontend package
# This assumes app.py is in the 'frontend' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.utils.state import initialize_session_state
from frontend.config import APP_TITLE, PAGE_ICON, LAYOUT, SESSION_STATE_KEY_LOGGED_IN

# --- Basic Logging Setup ---
# Configure logging level and format
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose frontend logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout # Log to standard output
)
logger = logging.getLogger(__name__)
logger.info("Frontend application starting...")

# --- Initialize Session State ---
# This should run very first upon script execution
initialize_session_state()
logger.debug(f"Initial session state keys: {list(st.session_state.keys())}")


# --- Page Configuration (Optional - can be set in individual pages too) ---
# st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# --- Main App Logic (Routing) ---
# Streamlit's multi-page app feature handles routing based on files in 'pages' dir.
# This main app.py can be used for global setup or as a landing/redirect page.

st.title(f"Welcome to {APP_TITLE}")

if st.session_state.get(SESSION_STATE_KEY_LOGGED_IN):
    logger.info("User is logged in, redirecting to workspace.")
    st.info("Redirecting to your workspace...")
    # Use st.page_link (newer Streamlit versions) or st.switch_page
    # st.page_link("pages/2_🚀_Project_Workspace.py", label="Go to Workspace") # Example using page_link
    st.switch_page("pages/2_🚀_Project_Workspace.py")
else:
    logger.info("User is not logged in, redirecting to login.")
    st.info("Please log in or sign up to continue.")
    # st.page_link("pages/1_🔐_Login_Signup.py", label="Login / Sign Up")
    st.switch_page("pages/1_🔐_Login_Signup.py")

# You could add global elements here if needed, but usually, pages handle their own content.
st.caption("Loading...")

logger.info("Frontend app.py execution finished.")