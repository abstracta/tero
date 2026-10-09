from abc import ABC
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
import secrets
import time
from typing import Any, Callable, Literal, Optional, cast
from urllib.parse import urlencode, urljoin

from fastapi import HTTPException, status
import httpx
from mcp.client.auth import OAuthClientProvider, TokenStorage, PKCEParameters, OAuthFlowError, OAuthRegistrationError, OAuthTokenError
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_client_registration_request,
    create_oauth_metadata_request,
    get_client_metadata_scopes,
    handle_auth_metadata_response,
    handle_protected_resource_response,
    handle_registration_response,
    should_use_client_metadata_url,
    create_client_info_from_metadata_url
)
from mcp.shared.auth import OAuthClientMetadata, OAuthToken, OAuthClientInformationFull, OAuthMetadata
from pydantic import AnyHttpUrl, BaseModel
from sqlmodel import SQLModel, Field, col, select, delete, and_
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.env import env
from ..core.repos import scalar, EncryptedField
from ..core.domain import CamelCaseModel


logger = logging.getLogger(__name__)


class ToolAuthTokenType(str, Enum):
    BEARER = "bearer"


class ToolAuthToken(SQLModel, table=True):
    __tablename__ : Any = "tool_auth_token"
    user_id: int = Field(primary_key=True)
    agent_id: int = Field(primary_key=True)
    tool_id: str = Field(primary_key=True)
    access_token: str = EncryptedField()
    token_type: ToolAuthTokenType = ToolAuthTokenType.BEARER
    expires_in: Optional[int] = Field(exclude=True, default=None)
    scope: Optional[str] = None
    refresh_token: Optional[str] = EncryptedField(nullable=True, default=None)
    expires_at: Optional[float] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)


class ToolOAuthState(SQLModel, table=True):
    __tablename__ : Any = "tool_oauth_state"
    user_id: int = Field(primary_key=True)
    agent_id: int = Field(index=True)
    tool_id: str = Field(primary_key=True)
    state: str = Field(primary_key=True)
    code_verifier: str = EncryptedField()
    token_endpoint: Optional[str]
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)


class ToolOAuthClientInfo(SQLModel, table=True):
    __tablename__ : Any = "tool_oauth_client_info"
    agent_id: int = Field(primary_key=True)
    tool_id: str = Field(primary_key=True)
    client_id: str
    client_secret: Optional[str] = EncryptedField(nullable=True)
    token_endpoint_auth_method: Optional[str] = None
    scope: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)


class ToolAuthRequest(CamelCaseModel):
    request_type: str
    tool_id: str
    agent_id: int


class ToolAuthRequestException(Exception):
    def __init__(self, request: ToolAuthRequest):
        self.request = request


class OwnerConfiguredToolAuthError(Exception):
    pass


class ToolOAuthRequest(ToolAuthRequest):
    request_type: str = "oauth"
    oauth_url: str
    oauth_state: str


class ToolAuthTokenRequest(ToolAuthRequest):
    request_type: str = "auth_token"


def build_tool_auth_request_http_exception(request: ToolAuthRequest) -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=request.model_dump(by_alias=True))


class ToolAuthCallback(BaseModel, ABC):
    pass


class ToolOAuthCallback(ToolAuthCallback):
    state: str
    code: str


class ToolAuthTokenCallback(ToolAuthCallback):
    auth_token: str


class ToolAuthCallbackError(Exception):
    pass


class ToolAuthRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_token(self, user_id: int, agent_id: int, tool_id: str) -> Optional[ToolAuthToken]:
        stmt = (select(ToolAuthToken).
            where(ToolAuthToken.user_id == user_id, ToolAuthToken.agent_id == agent_id, ToolAuthToken.tool_id == tool_id))
        result = await self._db.exec(stmt)
        return result.one_or_none()

    async def save_token(self, token: ToolAuthToken):
        token.updated_at = datetime.now(timezone.utc)
        await self._db.merge(token)
        await self._db.commit()

    async def delete_token(self, user_id: int, agent_id: int, tool_id: str):
        stmt = scalar(delete(ToolAuthToken).
            where(and_(ToolAuthToken.user_id == user_id, ToolAuthToken.agent_id == agent_id, ToolAuthToken.tool_id == tool_id)))
        await self._db.exec(stmt)
        await self._db.commit()
        await self.delete_state(user_id, agent_id, tool_id)

    async def find_state(self, user_id: int, tool_id: str, state: str) -> Optional[ToolOAuthState]:
        stmt = (select(ToolOAuthState).
            where(ToolOAuthState.user_id == user_id, ToolOAuthState.tool_id == tool_id, ToolOAuthState.state == state))
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def save_state(self, state: ToolOAuthState):
        state.updated_at = datetime.now(timezone.utc)
        await self._db.merge(state)
        await self._db.commit()

    async def delete_state(self, user_id: int, agent_id: int, tool_id: str):
        stmt = scalar(delete(ToolOAuthState).
            where(and_(ToolOAuthState.user_id == user_id, ToolOAuthState.tool_id == tool_id, ToolOAuthState.agent_id == agent_id)))
        await self._db.exec(stmt)
        await self._db.commit()

    async def cleanup(self):
        token_cutoff = datetime.now(timezone.utc) - timedelta(minutes=env.tool_oauth_token_ttl_minutes)
        token_stmt = scalar(delete(ToolAuthToken).where(and_(ToolAuthToken.updated_at < token_cutoff)))
        await self._db.exec(token_stmt)

        state_cutoff = datetime.now(timezone.utc) - timedelta(minutes=env.tool_oauth_state_ttl_minutes)
        state_stmt = scalar(delete(ToolOAuthState).where(and_(ToolOAuthState.updated_at < state_cutoff)))
        await self._db.exec(state_stmt)

        await self._db.commit()


class ToolOAuthClientInfoRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def save(self, info: ToolOAuthClientInfo):
        await self._db.merge(info)
        await self._db.commit()

    async def find_by_ids(self, agent_id: int, tool_id: str) -> Optional[ToolOAuthClientInfo]:
        stmt = (select(ToolOAuthClientInfo).
            where(ToolOAuthClientInfo.agent_id == agent_id, ToolOAuthClientInfo.tool_id == tool_id))
        result = await self._db.exec(stmt)
        return result.one_or_none()

    async def delete(self, agent_id: int, tool_id: str):
        stmt = scalar(delete(ToolOAuthClientInfo).
            where(and_(ToolOAuthClientInfo.agent_id == agent_id, ToolOAuthClientInfo.tool_id == tool_id)))
        await self._db.exec(stmt)
        await self._db.commit()

    async def cleanup(self, tool_id: str, ttl_minutes: int):
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=ttl_minutes)
        tool_id_parts = tool_id.split("-", 2)
        # we store empty client id when client doesn't support authentication. There is no need to clean it since it doesn't contain any sensitive data
        stmt = scalar(
            delete(ToolOAuthClientInfo)
            .where(and_(
                ToolOAuthClientInfo.tool_id == tool_id if len(tool_id_parts) == 1 else col(ToolOAuthClientInfo.tool_id).like(f"{tool_id_parts[0]}-%"),
                ToolOAuthClientInfo.updated_at < cutoff,
                ToolOAuthClientInfo.client_id != "")))
        await self._db.exec(stmt)
        await self._db.commit()


class AgentToolOAuthStorage(TokenStorage):

    def __init__(
        self,
        user_id: int,
        agent_id: int,
        tool_id: str,
        db: AsyncSession,
        oauth: "AgentToolOauth",
        redirect_uri: str,
        client_info_provider: Optional[Callable[[], OAuthClientInformationFull]] = None,
    ):
        self._user_id = user_id
        self._agent_id = agent_id
        self._tool_id = tool_id
        self._redirect_uri = redirect_uri
        self._oauth_repo = ToolAuthRepository(db)
        self._client_info_repo = ToolOAuthClientInfoRepository(db)
        self._oauth = oauth
        self._client_info_provider = client_info_provider

    async def get_tokens(self) -> Optional[OAuthToken]:
        ret = await self._oauth_repo.find_token(self._user_id, self._agent_id, self._tool_id)
        if ret:
            self._oauth.context.token_expiry_time = ret.expires_at
        return OAuthToken(
            access_token=ret.access_token,
            token_type="Bearer",
            expires_in=ret.expires_in,
            scope=ret.scope,
            refresh_token=ret.refresh_token
        ) if ret else None

    async def set_tokens(self, tokens: OAuthToken):
        await self._oauth_repo.save_token(ToolAuthToken(
            user_id=self._user_id,
            agent_id=self._agent_id,
            tool_id=self._tool_id,
            access_token=tokens.access_token,
            token_type=ToolAuthTokenType(tokens.token_type.lower()),
            expires_in=tokens.expires_in,
            scope=tokens.scope,
            refresh_token=tokens.refresh_token,
            expires_at=self._oauth.context.token_expiry_time
        ))

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        if self._client_info_provider:
            return self._client_info_provider()
        ret = await self._client_info_repo.find_by_ids(self._agent_id, self._tool_id)
        return OAuthClientInformationFull(
            client_id=ret.client_id,
            client_secret=ret.client_secret,
            token_endpoint_auth_method=cast(Literal["none", "client_secret_post", "client_secret_basic", "private_key_jwt"], ret.token_endpoint_auth_method),
            redirect_uris=[AnyHttpUrl(self._redirect_uri)]) if ret else None

    async def set_client_info(self, client_info: OAuthClientInformationFull):
        if self._client_info_provider:
            return
        info = ToolOAuthClientInfo(
            agent_id=self._agent_id,
            tool_id=self._tool_id,
            client_id=cast(str, client_info.client_id),
            client_secret=cast(str, client_info.client_secret),
            token_endpoint_auth_method=cast(str, client_info.token_endpoint_auth_method),
            scope=client_info.scope,
            updated_at=datetime.now(timezone.utc)
        )
        await self._client_info_repo.save(info)


def _build_redirect_uri(tool_id: str) -> str:
    return f"{env.frontend_url}/tools/{tool_id}/oauth-callback"


class UnsupportedClientRegistrationException(Exception):
    pass


class AgentToolOauth(OAuthClientProvider):

    _DUMMY_URL = AnyHttpUrl("http://localhost")

    def __init__(
        self,
        server_url: str,
        metadata: Optional[OAuthMetadata],
        scope: Optional[str],
        agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
        # tools supporting several configs per agent (eg: sql) share a single redirect uri across all of
        # them, so a single redirect uri needs to be registered in the identity provider
        redirect_uri: Optional[str] = None,
        client_info_provider: Optional[Callable[[], OAuthClientInformationFull]] = None,
    ):
        self._agent_id = agent_id
        self._tool_id = tool_id
        self._user_id = user_id
        self._oauth_repo = ToolAuthRepository(db)
        self._http_client = httpx.AsyncClient()
        redirect_uri = redirect_uri or _build_redirect_uri(tool_id)
        client_metadata = OAuthClientMetadata(redirect_uris=[AnyHttpUrl(redirect_uri)], scope=scope)
        super().__init__(
            server_url,
            client_metadata,
            AgentToolOAuthStorage(user_id, agent_id, tool_id, db, self, redirect_uri, client_info_provider),
            redirect_handler=self._redirect_handler,
            callback_handler=self._callback_handler
        )
        self.context.oauth_metadata = metadata
        self.state = ""
        self.code_verifier = ""

        # MCP SDK leaves client_id in the body under client_secret_basic; strict servers reject that.
        # https://github.com/modelcontextprotocol/python-sdk/issues/3138
        original_prepare_token_auth = self.context.prepare_token_auth

        def prepare_token_auth(data: dict[str, str], headers: dict[str, str] | None = None) -> tuple[dict[str, str], dict[str, str]]:
            data, headers = original_prepare_token_auth(data, headers)
            if headers.get("Authorization", "").startswith("Basic "):
                data = {k: v for k, v in data.items() if k not in ("client_id", "client_secret")}
            return data, headers

        self.context.prepare_token_auth = prepare_token_auth

    @property
    def server_url(self) -> str:
        return self.context.server_url

    # custom redirect handler that saves the state (to restore it in OAuth callback) and requests OAuth authentication flow
    async def _redirect_handler(self, auth_url: str):
        tool_state = ToolOAuthState(
            user_id=self._user_id,
            agent_id=self._agent_id,
            tool_id=self._tool_id,
            state=self.state,
            code_verifier=self.code_verifier,
            token_endpoint=self.context.oauth_metadata.token_endpoint.unicode_string() if self.context.oauth_metadata else None)
        await self._oauth_repo.save_state(tool_state)
        raise ToolAuthRequestException(ToolOAuthRequest(tool_id=self._tool_id, oauth_url=auth_url, oauth_state=self.state, agent_id=self._agent_id))

    # this is just to satisfy the callback_handler. It should never be called due to the redirect_handler
    async def _callback_handler(self) -> tuple[str, str | None]:
        return "", None

    # part of this logic is the same as async_auth_flow but instead of adding header to a request and doing the complete OAuth flow,
    # a ToolOAuthRequest is raised when needed
    async def solve_tokens(self) -> Optional[OAuthToken]:
        async with self.context.lock:
            if not self._initialized:
                await self._initialize()

            # if client_id is empty then it means that the client doesn't support authentication
            if not self.context.client_info or self.context.client_info.client_id:
                try:
                    await self.ensure_token()
                except UnsupportedClientRegistrationException:
                    return None

            return self.context.current_tokens

    async def ensure_token(self) -> None:
        if self.is_token_valid():
            return

        if self.context.can_refresh_token():
            refresh_request = await self._refresh_token()
            refresh_response = await self._http_request(refresh_request)

            if await self._handle_refresh_response(refresh_response):
                return
            self._initialized = False

        await self._discover_oauth_metadata()

        if not self.context.client_info:
            if should_use_client_metadata_url(
                self.context.oauth_metadata, self.context.client_metadata_url
            ):
                logger.debug(f"Using URL-based client ID (CIMD): {self.context.client_metadata_url}")
                client_information = create_client_info_from_metadata_url(
                    self.context.client_metadata_url,  # type: ignore[arg-type]
                    redirect_uris=self.context.client_metadata.redirect_uris,
                )
                self.context.client_info = client_information
                await self.context.storage.set_client_info(client_information)
            else:
                registration_request = create_client_registration_request(
                    self.context.oauth_metadata,
                    self.context.client_metadata,
                    self.context.get_authorization_base_url(self.context.server_url),
                )
                registration_response = await self._http_request(registration_request)
                client_information = await self._handle_registration_response(registration_response)
                self.context.client_info = client_information
                await self.context.storage.set_client_info(client_information)

        await self._perform_authorization()

    # override this method to add a 1 minute buffer to the token expiry time to avoid 401 errors
    def is_token_valid(self) -> bool:
        return bool(
            self.context.current_tokens
            and self.context.current_tokens.access_token
            and (not self.context.token_expiry_time or time.time() + 60 <= self.context.token_expiry_time)
        )

    async def _http_request(self, request: httpx.Request) -> httpx.Response:
        # This is a custom fix since some mcp servers (like playwright) require the client to accept both application/json and text/event-stream
        request.headers["Accept"] = "application/json, text/event-stream"
        return await self._http_client.send(request)

    async def _discover_oauth_metadata(self) -> None:
        prm_discovery_urls = build_protected_resource_metadata_discovery_urls(None, self.context.server_url)

        for url in prm_discovery_urls:
            discovery_request = create_oauth_metadata_request(url)
            discovery_response = await self._http_request(discovery_request)

            prm = await handle_protected_resource_response(discovery_response)
            if prm:
                self.context.protected_resource_metadata = prm
                self.context.auth_server_url = str(prm.authorization_servers[0])
                break
            else:
                logger.debug(f"Protected resource metadata discovery failed: {url}")

        asm_discovery_urls = build_oauth_authorization_server_metadata_discovery_urls(
            self.context.auth_server_url, self.context.server_url
        )

        for url in asm_discovery_urls:
            oauth_metadata_request = create_oauth_metadata_request(url)
            oauth_metadata_response = await self._http_request(oauth_metadata_request)

            ok, asm = await handle_auth_metadata_response(oauth_metadata_response)
            if not ok:
                break
            if ok and asm:
                self.context.oauth_metadata = asm
                break
            else:
                logger.debug(f"OAuth metadata discovery failed: {url}")

        # Add this custom logic so if scope was already provided, do not override it
        if not self.context.client_metadata.scope:
            self.context.client_metadata.scope = get_client_metadata_scopes(
                None,
                self.context.protected_resource_metadata,
                self.context.oauth_metadata,
            )

    async def _handle_registration_response(self, registration_response: httpx.Response) -> OAuthClientInformationFull:
        try:
            ret = await handle_registration_response(registration_response)
            if not ret.token_endpoint_auth_method and ret.client_secret:
                # force client_secret_post since otherwise we get "Client secret is required" with some mcp servers (like https://mcp.exa.ai)
                ret.token_endpoint_auth_method = "client_secret_post"
        except OAuthRegistrationError as e:
            # some mcp servers return 404 others may fail with 400 (eg: mcp playwright) when registration is not supported
            if e.args and e.args[0].startswith("Registration failed: 4"):
                if not e.args[0].startswith("Registration failed: 404"):
                    # we log it in case the registration actually fails and is not expected so later on admins can review it
                    logger.warning("Client registration failed for %s", self.context.server_url, exc_info=e)
                self.context.client_info = OAuthClientInformationFull(client_id="", client_secret="", redirect_uris=[self._DUMMY_URL], token_endpoint_auth_method="none", grant_types=[], response_types=[])
                await self.context.storage.set_client_info(self.context.client_info)
                raise UnsupportedClientRegistrationException(e)
            raise
        return ret

    async def _perform_authorization(self) -> httpx.Request:
        # same as the one in oauth2.py from mcp library but stores code verifier and state in the class so when redirect is invoked it can store them to later resume the flow
        if self.context.client_metadata.redirect_uris is None:
            raise OAuthFlowError("No redirect URIs provided for authorization code grant")
        if not self.context.redirect_handler:
            raise OAuthFlowError("No redirect handler provided for authorization code grant")
        if not self.context.callback_handler:
            raise OAuthFlowError("No callback handler provided for authorization code grant")

        if self.context.oauth_metadata and self.context.oauth_metadata.authorization_endpoint:
            auth_endpoint = str(self.context.oauth_metadata.authorization_endpoint)
        else:
            auth_base_url = self.context.get_authorization_base_url(self.context.server_url)
            auth_endpoint = urljoin(auth_base_url, "/authorize")

        if not self.context.client_info:
            raise OAuthFlowError("No client info available for authorization")

        pkce_params = PKCEParameters.generate()
        self.code_verifier = pkce_params.code_verifier
        self.state = secrets.token_urlsafe(32)

        auth_params = {
            "response_type": "code",
            "client_id": self.context.client_info.client_id,
            "redirect_uri": str(self.context.client_metadata.redirect_uris[0]),
            "state": self.state,
            "code_challenge": pkce_params.code_challenge,
            "code_challenge_method": "S256",
        }

        if self.context.should_include_resource_param(self.context.protocol_version):
            auth_params["resource"] = self.context.get_resource_url()

        if self.context.client_metadata.scope:
            auth_params["scope"] = self.context.client_metadata.scope

        authorization_url = f"{auth_endpoint}?{urlencode(auth_params)}"
        await self.context.redirect_handler(authorization_url)
        # returning dummy request just to satisfy the return type
        return httpx.Request("GET", "https://dummy")

    # part of this logic is the same as async_auth_flow after the callback is invoked
    async def callback(self, auth_callback: ToolOAuthCallback, state: ToolOAuthState):
        try:
            if not self._initialized:
                await self._initialize()
                await self._discover_oauth_metadata()
            token_request = await self._exchange_token_authorization_code(auth_callback.code, state.code_verifier)
            token_response = await self._http_request(token_request)
            await self._handle_token_response(token_response)
        except OAuthTokenError:
            logger.warning(f"Authentication failed for agent {self._agent_id} tool {self._tool_id} user {self._user_id}", exc_info=True)
            raise ToolAuthCallbackError("Authentication failed")
