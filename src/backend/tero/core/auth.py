import datetime
import json
import logging
from typing import Optional, Annotated, Any, AsyncGenerator

import aiohttp
from fastapi import Depends, HTTPException, status
from fastapi.security import OpenIdConnect
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from jose.exceptions import JWKError
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.requests import Request

from .env import env
from .repos import get_db
from ..teams.repos import TeamRepository
from ..teams.domain import TeamRole, Role, TeamRoleStatus, GLOBAL_TEAM_ID
from ..api_keys.repos import ApiKeyRepository
from ..users.domain import User
from ..users.repos import UserRepository


class BearerOpenIdConnect(OpenIdConnect):

    async def __call__(self, request: Request) -> Optional[str]:
        authorization = request.headers.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            raise _build_auth_exception()
        return param


def _build_auth_exception() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})


def _get_config_url() -> Optional[str]:
    return env.openid_url + "/.well-known/openid-configuration" if env.openid_url else None


logger = logging.getLogger(__name__)
config_url = _get_config_url()
if config_url:
    auth_scheme = BearerOpenIdConnect(openIdConnectUrl=config_url)
else:
    auth_scheme = lambda: None


class OpenIdConfig:

    def __init__(self, url: str, http_cli: aiohttp.ClientSession):
        self.url = url
        self._http_cli = http_cli
        self._last_update = None
        self._keys = None

    async def get_updated_keys(self, period: datetime.timedelta) -> Any:
        if self._last_update is None or datetime.datetime.now(datetime.UTC) - self._last_update > period:
            await self._update_keys()
        return self._keys

    async def _update_keys(self):
        async with self._http_cli.get(self.url) as config_resp:
            config_resp.raise_for_status()
            jwks_uri = (await config_resp.json())['jwks_uri']
        async with self._http_cli.get(jwks_uri) as ret_resp:
            ret_resp.raise_for_status()
            self._keys = await ret_resp.json()
        self._last_update = datetime.datetime.now(datetime.UTC)


async def get_http_cli() -> AsyncGenerator[aiohttp.ClientSession, None]:
    async with aiohttp.ClientSession() as cli:
        yield cli


def _get_openid_config(http_cli: Annotated[aiohttp.ClientSession, Depends(get_http_cli)]) -> Optional[OpenIdConfig]:
    if not config_url:
        return None
    return OpenIdConfig(config_url, http_cli)


async def _decode_token(token: str, openid_config: Annotated[OpenIdConfig, Depends(_get_openid_config)]) -> dict:
    # openid keys rolled according to recommendation from
    # https://learn.microsoft.com/en-us/entra/identity-platform/signing-key-rollover
    openid_keys = await openid_config.get_updated_keys(datetime.timedelta(days=1))
    options = {"verify_aud": False}
    try:
        return jwt.decode(token, openid_keys, options=options)
    except JWTError as e:
        new_keys = await openid_config.get_updated_keys(datetime.timedelta(minutes=5))
        if new_keys == openid_keys:
            raise e
        return jwt.decode(token, new_keys, options=options)


def to_utc_aware(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc)


def _decode_api_key_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, env.secret_encryption_key.get_secret_value(), algorithms=["HS256"])
    except JWTError:
        return None


async def _try_api_key_auth(token: Optional[str], db: AsyncSession) -> User:
    if not token:
        raise _build_auth_exception()
    payload = _decode_api_key_token(token)
    if not payload or payload.get("source") != "api_key":
        raise _build_auth_exception()
    try:
        user_id = int(payload["sub"])
        api_key_id = int(payload["api_key_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _build_auth_exception() from exc
    api_key_repo = ApiKeyRepository(db)
    api_key = await api_key_repo.find_by_id(api_key_id)
    if not api_key:
        raise _build_auth_exception()
    if api_key.expires_at and to_utc_aware(api_key.expires_at) < datetime.datetime.now(datetime.timezone.utc):
        raise _build_auth_exception()
    ret = await UserRepository(db).find_by_id(user_id)
    if not ret or not ret.is_active():
        raise _build_auth_exception()
    if env.allowed_users and ret.username not in env.allowed_users:
        raise _build_auth_exception()
    return ret


async def get_current_user(token: Annotated[Optional[str], Depends(auth_scheme)],
        open_id_config: Annotated[Optional[OpenIdConfig], Depends(_get_openid_config)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> User:
    try:
        if not token or not open_id_config:
            logger.debug("No token or open_id_config found, trying API key auth")
            return await _try_api_key_auth(token, db)
        payload = await _decode_token(token, open_id_config)
        username = payload.get("email")
        if username is None:
            logger.warning("No username could be found in %s", json.dumps(payload))
            raise _build_auth_exception()
        name = payload.get("name")
        if name is None:
            logger.warning("No name could be found in %s", json.dumps(payload))
            raise _build_auth_exception()
        if env.allowed_users and username not in env.allowed_users:
            raise _build_auth_exception()
        user_repo = UserRepository(db)
        ret = await user_repo.find_by_username(username)
        if ret and not ret.is_active():
            raise _build_auth_exception()
        if not ret:
            deleted_user = await user_repo.find_by_username_include_deleted(username)
            if deleted_user:
                raise _build_auth_exception()
            ret = await user_repo.create_user(User(username=username, name=name, monthly_usd_limit=env.monthly_usd_limit_default))
            if ret.id == 1:
                await TeamRepository(db).save_team_role(TeamRole(user_id=ret.id, team_id=GLOBAL_TEAM_ID, role=Role.TEAM_OWNER, status=TeamRoleStatus.ACCEPTED))
        elif not ret.name and name:
            ret.name = name
            ret = await user_repo.update_user(ret)
        return ret
    except (JWTError, JWKError):
        logger.debug("OIDC token validation failed, trying API key auth", exc_info=True)
        return await _try_api_key_auth(token, db)
