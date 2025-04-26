# frontend/pages/1_🔐_Login_Signup.py
import streamlit as st
import logging
from frontend.utils.auth import handle_login, handle_signup
from frontend.config import APP_TITLE, SESSION_STATE_KEY_LOGGED_IN

logger = logging.getLogger(__name__)

st.set_page_config(page_title=f"Login - {APP_TITLE}", layout="centered")

# Redirect if already logged in
if st.session_state.get(SESSION_STATE_KEY_LOGGED_IN, False):
    st.switch_page("pages/2_🚀_Project_Workspace.py")

st.title(f"Welcome to {APP_TITLE} 🪙")
st.caption("Please log in or sign up to continue.")

choice = st.radio("Choose Action", ["Login", "Sign Up"], horizontal=True)

if choice == "Login":
    st.subheader("Login")
    with st.form("login_form"):
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not login_email or not login_password:
                st.warning("Please enter both email and password.")
            else:
                handle_login(login_email, login_password)

elif choice == "Sign Up":
    st.subheader("Sign Up")
    with st.form("signup_form"):
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        signup_button = st.form_submit_button("Sign Up")

        if signup_button:
            if not signup_email or not signup_password or not confirm_password:
                st.warning("Please fill in all fields.")
            elif signup_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                # Attempt signup, handle_signup shows success/error messages
                handle_signup(signup_email, signup_password)

# Add footer or links if needed
st.markdown("---")
st.caption("Ensure your FastAPI backend server is running.")