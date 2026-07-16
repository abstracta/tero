from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest_asyncio
from freezegun import freeze_time
from httpx import ASGITransport, AsyncClient
from sqlmodel.ext.asyncio.session import AsyncSession

from tero.api import app
from tero.api_keys.api import API_KEYS_PATH, TOKEN_PATH, API_KEY_TOKEN_EXPIRY_HOURS, _api_key_id, _hash_secret
from tero.api_keys.domain import ApiKey
from tero.api_keys.repos import ApiKeyRepository
from tero.core.repos import get_db
from tero.users.api import CURRENT_USER_PATH

from .common import *


@pytest_asyncio.fixture
async def unauthenticated_client(session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def get_db_override() -> AsyncGenerator[AsyncSession, None]:
        yield session

    app.dependency_overrides[get_db] = get_db_override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()


async def test_create_api_key(client: AsyncClient):
    resp = await client.post(API_KEYS_PATH, json={"name": "ci-key"})
    resp.raise_for_status()
    data = resp.json()
    assert data["name"] == "ci-key"
    assert data["apiKey"].startswith("tero_sk_")


async def test_exchange_api_key(client: AsyncClient, unauthenticated_client: AsyncClient):
    create_resp = await client.post(API_KEYS_PATH, json={"name": "ci-key"})
    create_resp.raise_for_status()
    api_key = create_resp.json()["apiKey"]

    token_resp = await unauthenticated_client.post(TOKEN_PATH, json={"api_key": api_key})
    token_resp.raise_for_status()
    data = token_resp.json()
    assert data["accessToken"]
    assert data["expiresIn"] == API_KEY_TOKEN_EXPIRY_HOURS * 3600


async def test_api_key_token_authenticates(client: AsyncClient, unauthenticated_client: AsyncClient):
    create_resp = await client.post(API_KEYS_PATH, json={"name": "ci-key"})
    create_resp.raise_for_status()
    api_key = create_resp.json()["apiKey"]

    token_resp = await unauthenticated_client.post(TOKEN_PATH, json={"api_key": api_key})
    token_resp.raise_for_status()
    access_token = token_resp.json()["accessToken"]

    profile_resp = await unauthenticated_client.get(
        CURRENT_USER_PATH,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    profile_resp.raise_for_status()
    assert "teams" in profile_resp.json()


async def test_exchange_invalid_api_key(unauthenticated_client: AsyncClient):
    resp = await unauthenticated_client.post(TOKEN_PATH, json={"api_key": "tero_sk_invalid"})
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED


@freeze_time(CURRENT_TIME)
async def test_exchange_expired_api_key(session: AsyncSession, unauthenticated_client: AsyncClient):
    expired_value = "tero_sk_expiredtestkey000000000000000000"
    key_id = _api_key_id(expired_value)
    assert key_id is not None
    await ApiKeyRepository(session).create(ApiKey(
        user_id=USER_ID,
        name="expired",
        key_id=key_id,
        hashed_secret=_hash_secret(expired_value),
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    ))

    resp = await unauthenticated_client.post(TOKEN_PATH, json={"api_key": expired_value})
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
