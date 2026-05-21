"""Authentication routes: register, login, logout, me.

JWT-based auth using python-jose for token signing and passlib[bcrypt] for
password hashing.  All routes are stateless — logout is a client-side
operation (drop the token).

Token claims: ``sub`` (user_id as string), ``role``, ``exp``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt
import structlog
from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    DbSessionDep,
    SettingsDep,
    get_db_session,
    get_settings,
)
from app.config import Settings
from app.domain.errors import PermissionDenied, ValidationError
from app.repositories.models import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 h


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """New user registration payload."""

    email: EmailStr = Field(..., description="Email address (unique)")
    password: str = Field(..., min_length=8, description="Password (min 8 chars)")


class UserResponse(BaseModel):
    """Public user profile (no hashed_password)."""

    id: uuid.UUID
    email: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """JWT bearer token."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_password(password: str) -> str:
    hashed = _bcrypt.hashpw(password[:72].encode(), _bcrypt.gensalt())
    return hashed.decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return bool(_bcrypt.checkpw(plain[:72].encode(), hashed.encode()))


def _create_access_token(user_id: uuid.UUID, role: str, signing_key: str) -> str:
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
    }
    return str(jwt.encode(payload, signing_key, algorithm=_JWT_ALGORITHM))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    settings: SettingsDep,
    db: DbSessionDep,
) -> UserResponse:
    """Register a new user account.

    Args:
        body: Email + password.
        settings: Application settings (signing key, etc.).
        db: Database session.

    Returns:
        UserResponse with the newly created user profile.

    Raises:
        ValidationError: Email already registered.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    existing: User | None = result.scalar_one_or_none()
    if existing is not None:
        raise ValidationError("Email already registered")

    user = User(
        id=uuid.uuid4(),
        email=body.email,
        hashed_password=_hash_password(body.password),
        role="user",
        is_active=True,
        is_superuser=False,
        is_verified=False,
    )
    db.add(user)
    await db.flush()  # Populate server defaults (created_at, etc.)

    logger.info("auth.register", user_id=str(user.id), email=body.email)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """Authenticate and return a JWT bearer token.

    Accepts ``application/x-www-form-urlencoded`` (OAuth2 password flow).

    Args:
        form: ``username`` (email) + ``password``.
        settings: Application settings.
        db: Database session.

    Returns:
        TokenResponse with JWT and user profile.

    Raises:
        PermissionDenied: Invalid credentials or inactive account.
    """
    result = await db.execute(select(User).where(User.email == form.username))
    user: User | None = result.scalar_one_or_none()

    if user is None or not _verify_password(form.password, user.hashed_password):
        raise PermissionDenied("Invalid email or password")

    if not user.is_active:
        raise PermissionDenied("Account is disabled")

    token = _create_access_token(user.id, user.role, settings.jwt_signing_key)
    logger.info("auth.login", user_id=str(user.id))
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout() -> None:
    """Invalidate the current session (stateless — client drops the token).

    Returns:
        204 No Content.
    """
    # JWT is stateless; invalidation is the client's responsibility.
    return None


@router.get("/me", response_model=UserResponse)
async def me(
    request: Request,
    settings: SettingsDep,
    db: DbSessionDep,
    token: str = Depends(_oauth2_scheme),
) -> UserResponse:
    """Return the current user's profile.

    Args:
        request: Incoming request (used for request_id logging).
        settings: Application settings.
        db: Database session.
        token: Bearer token from Authorization header.

    Returns:
        UserResponse with the caller's profile.

    Raises:
        PermissionDenied: Token invalid, expired, or user not found.
    """
    user = await _get_current_user(token=token, settings=settings, db=db)
    return UserResponse.model_validate(user)


# ---------------------------------------------------------------------------
# Shared auth dependency used by other routes via Depends
# ---------------------------------------------------------------------------


async def get_current_user(
    token: str = Depends(_oauth2_scheme),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """FastAPI dependency: validate JWT and return the User ORM object.

    Args:
        token: Bearer token extracted by OAuth2PasswordBearer.
        settings: Application settings for JWT signing key.
        db: Database session.

    Returns:
        Authenticated User ORM instance.

    Raises:
        PermissionDenied: Token invalid, expired, or user not found / inactive.
    """
    return await _get_current_user(token=token, settings=settings, db=db)


async def _get_current_user(
    token: str,
    settings: Settings,
    db: AsyncSession,
) -> User:
    """Shared implementation for JWT validation → User lookup.

    Args:
        token: Raw JWT string.
        settings: Application settings (provides jwt_signing_key).
        db: SQLAlchemy async session for user lookup.

    Returns:
        Authenticated User ORM instance.

    Raises:
        PermissionDenied: Token is invalid, expired, or user is inactive.
    """
    try:
        payload = jwt.decode(token, settings.jwt_signing_key, algorithms=[_JWT_ALGORITHM])
        user_id_raw: str | None = payload.get("sub")
        if user_id_raw is None:
            raise PermissionDenied("Invalid token: missing sub claim")
        user_id = uuid.UUID(user_id_raw)
    except JWTError as exc:
        raise PermissionDenied(f"Invalid or expired token: {exc}") from exc
    except ValueError as exc:
        raise PermissionDenied(f"Invalid token payload: {exc}") from exc

    result = await db.execute(select(User).where(User.id == user_id))
    user: User | None = result.scalar_one_or_none()
    if user is None:
        raise PermissionDenied("User not found")
    if not user.is_active:
        raise PermissionDenied("Account is disabled")

    return user
