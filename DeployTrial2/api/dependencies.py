# filepath: DeployTrial2/api/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
# from jose import JWTError, jwt # Using pyjwt from jwt_handler instead
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

from core.config import settings
from models.api_models import TokenData, UserInDBBase
from models.db_models import ProjectInDB, UserInDB
from database.connection import get_database
from database import crud
from auth.jwt_handler import decode_access_token # Use our custom decoder

logger = logging.getLogger(__name__)

# Define the OAuth2 scheme, pointing to the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

async def get_current_user(
    db: AsyncIOMotorDatabase = Depends(get_database),
    token: str = Depends(oauth2_scheme)
) -> UserInDB:
    """
    Dependency to get the current user from the JWT token.
    Verifies the token, decodes it, and fetches the user from the database.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        logger.warning("Token decoding failed or token expired.")
        raise credentials_exception

    email: Optional[str] = payload.get("sub") # Assuming email is stored in 'sub' claim
    if email is None:
        logger.warning("Token payload missing 'sub' (subject/email).")
        raise credentials_exception

    # We have the email, now fetch the user from DB
    user = await crud.get_user_by_email(db, email=email)
    if user is None:
        logger.warning(f"User '{email}' from token not found in database.")
        raise credentials_exception

    logger.debug(f"Authenticated user retrieved: {user.email}")
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """
    Dependency to get the current *active* user.
    Placeholder for future logic to check if a user is active/enabled.
    """
    # Example: Add logic here if users can be deactivated
    # if not current_user.is_active:
    #     logger.warning(f"Attempt to access by inactive user: {current_user.email}")
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    logger.debug(f"Confirmed active user: {current_user.email}")
    return current_user

# Optional: Dependency to get the project and verify ownership
async def get_project_for_user(
    project_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    current_user: UserInDB = Depends(get_current_active_user)
) -> ProjectInDB:
    """
    Dependency to fetch a project by ID and verify it belongs to the current user.
    """
    project = await crud.get_project_by_id(db=db, project_id=project_id, user_id=current_user.id)
    if not project:
        logger.warning(f"Project {project_id} not found or not owned by user {current_user.email}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    logger.debug(f"Project {project_id} verified for user {current_user.email}")
    return project
