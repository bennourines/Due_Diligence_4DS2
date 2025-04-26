# filepath: DeployTrial2/api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import timedelta
import logging

from models.api_models import UserCreate, Token, UserInDBBase
from models.db_models import UserInDB
from database.connection import get_database
from database import crud
from auth.security import authenticate_user
from auth.jwt_handler import create_access_token
from core.config import settings
from api.dependencies import get_current_active_user # Import dependency

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/register", response_model=UserInDBBase, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_in: UserCreate,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Register a new user.

    - **email**: User's email address.
    - **password**: User's desired password.
    """
    logger.info(f"Registration attempt for email: {user_in.email}")
    existing_user = await crud.get_user_by_email(db, email=user_in.email)
    if existing_user:
        logger.warning(f"Registration failed: Email '{user_in.email}' already registered.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    try:
        user = await crud.create_user(db=db, user_in=user_in)
        logger.info(f"User '{user.email}' registered successfully with ID: {user.id}")
        # Return basic user info, not the full DB model with password hash
        return UserInDBBase(id=user.id, email=user.email)
    except Exception as e:
        # Catch potential errors during user creation (e.g., DB write error)
        logger.error(f"Error during registration for {user_in.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during user registration."
        )


@router.post("/login", response_model=Token)
async def login_for_access_token(
    db: AsyncIOMotorDatabase = Depends(get_database),
    form_data: OAuth2PasswordRequestForm = Depends() # Uses form data: 'username' and 'password'
):
    """
    Authenticate user via email and password (sent as form data) and return JWT token.
    """
    logger.info(f"Login attempt for user: {form_data.username}")
    user = await authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        # Error logged within authenticate_user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, # Use email as the subject claim
        expires_delta=access_token_expires
    )
    logger.info(f"Login successful, token generated for: {user.email}")
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserInDBBase)
async def read_users_me(
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Get the profile information of the currently authenticated user.
    """
    logger.debug(f"Fetching profile for current user: {current_user.email}")
    # Return basic user info based on the dependency result
    return UserInDBBase(id=current_user.id, email=current_user.email)
