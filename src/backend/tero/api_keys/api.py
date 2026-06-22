from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Annotated, Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.api import BASE_PATH
from ..core.auth import get_current_user, to_utc_aware
from ..core.env import env
from ..core.repos import get_db
from ..users.domain import User
from .domain import (
    ApiKey,
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyTokenRequest,
    ApiKeyTokenResponse,
)
from .repos import ApiKeyRepository

router = APIRouter()
API_KEYS_PATH = f"{BASE_PATH}/api-keys"
TOKEN_PATH = f"{API_KEYS_PATH}/token"

API_KEY_TOKEN_EXPIRY_HOURS = 1


@router.post(API_KEYS_PATH, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyCreate,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiKeyCreated:
    api_key_value = _generate_api_key()
    key_id = _compute_key_id(api_key_value)
    expires_at = None
    if body.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
    api_key = ApiKey(
        user_id=user.id,
        name=body.name,
        key_id=key_id,
        hashed_secret=_hash_secret(api_key_value),
        expires_at=expires_at,
    )
    api_key = await ApiKeyRepository(db).create(api_key)
    return ApiKeyCreated(
        id=api_key.id,
        name=api_key.name,
        api_key=api_key_value,
    )


@router.post(TOKEN_PATH)
async def exchange_token(
    body: ApiKeyTokenRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiKeyTokenResponse:
    repo = ApiKeyRepository(db)
    key_id = _api_key_id(body.api_key)
    if not key_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key credentials")
    api_key = await repo.find_by_key_id(key_id)
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key credentials")
    if not _verify_secret(body.api_key, api_key.hashed_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key credentials")
    if api_key.expires_at and to_utc_aware(api_key.expires_at) < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has expired")
    await repo.update_last_used(api_key)
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(api_key.user_id),
        "source": "api_key",
        "api_key_id": api_key.id,
        "iat": now,
        "exp": now + timedelta(hours=API_KEY_TOKEN_EXPIRY_HOURS),
    }
    token = jwt.encode(payload, env.secret_encryption_key.get_secret_value(), algorithm="HS256")
    return ApiKeyTokenResponse(
        access_token=token,
        expires_in=API_KEY_TOKEN_EXPIRY_HOURS * 3600,
    )

API_KEY_PREFIX = "tero_sk_"

def _generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(27)

def _hash_secret(secret: str) -> str:
    return bcrypt.hashpw(secret.encode(), bcrypt.gensalt()).decode()

def _verify_secret(secret: str, hashed: str) -> bool:
    return bcrypt.checkpw(secret.encode(), hashed.encode())

def _compute_key_id(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:32]

def _api_key_id(secret: str) -> Optional[str]:
    if not secret.startswith(API_KEY_PREFIX):
        return None
    return _compute_key_id(secret)
