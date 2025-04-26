# filepath: DeployTrial2/auth/jwt_handler.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Union # Added Union
import jwt # Using pyjwt library
from core.config import settings
import logging

logger = logging.getLogger(__name__)

SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

def create_access_token(data: dict, expires_delta: Optional[Union[timedelta, int]] = None) -> str:
    """
    Creates a JWT access token.

    Args:
        data: Dictionary to be encoded in the token payload (e.g., {"sub": user_email}).
        expires_delta: Optional timedelta or minutes until token expiration.
                       Defaults to ACCESS_TOKEN_EXPIRE_MINUTES from settings.

    Returns:
        The encoded JWT token string.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)

    if expires_delta:
        if isinstance(expires_delta, int):
            expire = now + timedelta(minutes=expires_delta)
        elif isinstance(expires_delta, timedelta):
             expire = now + expires_delta
        else:
             # Default if type is wrong, though type hints should help
             expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": now}) # Add issued-at time
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.debug(f"Created JWT token for subject: {data.get('sub')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error encoding JWT token: {e}", exc_info=True)
        raise # Re-raise the error for handling upstream


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes a JWT access token.

    Args:
        token: The JWT token string.

    Returns:
        The decoded payload dictionary if the token is valid and not expired,
        otherwise None.
    """
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]} # Ensure essential claims are present
        )
        logger.debug(f"Successfully decoded JWT token for subject: {payload.get('sub')}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired.")
        return None # Or raise specific exception
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None # Or raise specific exception
    except Exception as e:
        logger.error(f"Unexpected error decoding JWT token: {e}", exc_info=True)
        return None # Or raise
