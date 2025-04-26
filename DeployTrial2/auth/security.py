# filepath: DeployTrial2/auth/security.py
from passlib.context import CryptContext
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from models.db_models import UserInDB
from database import crud
import logging

logger = logging.getLogger(__name__)

# Configure password hashing context
# Using bcrypt as the default scheme
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a stored hash."""
    try:
        is_valid = pwd_context.verify(plain_password, hashed_password)
        if not is_valid:
            logger.debug("Password verification failed.")
        return is_valid
    except Exception as e:
        # Log potential errors during verification (e.g., invalid hash format)
        logger.error(f"Error verifying password: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    try:
        hashed_password = pwd_context.hash(password)
        logger.debug("Password hashed successfully.")
        return hashed_password
    except Exception as e:
        logger.error(f"Error hashing password: {e}", exc_info=True)
        raise # Re-raise error as hashing failure is critical

async def authenticate_user(db: AsyncIOMotorDatabase, email: str, password: str) -> Optional[UserInDB]:
    """
    Authenticates a user by email and password.

    Args:
        db: AsyncIOMotorDatabase instance.
        email: User's email.
        password: User's plain password.

    Returns:
        The UserInDB object if authentication is successful, otherwise None.
    """
    logger.debug(f"Attempting to authenticate user: {email}")
    user = await crud.get_user_by_email(db, email=email)
    if not user:
        logger.warning(f"Authentication failed: User '{email}' not found.")
        return None
    if not verify_password(password, user.hashed_password):
        logger.warning(f"Authentication failed: Incorrect password for user '{email}'.")
        return None

    logger.info(f"User authenticated successfully: {email}")
    return user
