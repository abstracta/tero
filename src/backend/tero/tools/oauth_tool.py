import abc
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import ClassVar, Optional, cast

from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyHttpUrl
from sqlmodel.ext.asyncio.session import AsyncSession

from .auth import (
    AgentToolOauth,
    OAuthMetadata,
    ToolAuthCallback,
    ToolAuthCallbackError,
    ToolAuthRepository,
    ToolAuthRequestException,
    ToolAuthTokenRequest,
    ToolOAuthCallback,
    ToolOAuthClientInfo,
    ToolOAuthClientInfoRepository,
    _build_redirect_uri,
)
from .core import AgentTool
from ..agents.domain import AgentToolConfig


@dataclass(frozen=True)
class OAuthToolConfig:
    authority_base_url: str
    authorize_path: str
    token_path: str
    scope: str

    @property
    def metadata(self) -> OAuthMetadata:
        return OAuthMetadata(
            issuer=AnyHttpUrl(self.authority_base_url),
            authorization_endpoint=AnyHttpUrl(f"{self.authority_base_url}{self.authorize_path}"),
            token_endpoint=AnyHttpUrl(f"{self.authority_base_url}{self.token_path}"),
            code_challenge_methods_supported=["S256"],
        )


_AZURE_AD_AUTHORITY_BASE = "https://login.microsoftonline.com"


def build_azure_oauth_config(tenant_id: str, scope: str) -> OAuthToolConfig:
    return OAuthToolConfig(
        authority_base_url=f"{_AZURE_AD_AUTHORITY_BASE}/{tenant_id}",
        authorize_path="/oauth2/v2.0/authorize",
        token_path="/oauth2/v2.0/token",
        scope=scope,
    )


class OAuthAgentTool(AgentTool, abc.ABC):
    _oauth: Optional[AgentToolOauth] = None
    _client_secret_config_key: ClassVar[str] = "clientSecret"
    _client_id_config_key: ClassVar[str] = "clientId"

    @abc.abstractmethod
    def _oauth_config(self) -> OAuthToolConfig:
        raise NotImplementedError

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        client_info_repo = ToolOAuthClientInfoRepository(self.db)
        # the client info is stored with the tool id the previous config was saved with, which may differ from
        # the current one for tools resolving their id from the config (eg: the alias of a sql connection)
        prev_client_info = await client_info_repo.find_by_ids(
            self.agent.id, prev_config.tool_id if prev_config else self.id
        )
        client_id = self._env_client_id() or self.config[self._client_id_config_key]
        client_secret = self._env_client_secret() or self._get_secret(self._client_secret_config_key)
        if (
            prev_config and prev_config.config != self.config
            or prev_client_info and client_secret and prev_client_info.client_secret != client_secret
        ):
            if not client_secret and prev_client_info and prev_client_info.client_id == client_id:
                client_secret = prev_client_info.client_secret
            await self.teardown()
        if client_secret:
            await self._save_client_info(client_info_repo, client_id, client_secret)
        async with self.load():
            pass

    async def _save_client_info(
        self,
        client_info_repo: ToolOAuthClientInfoRepository,
        client_id: str,
        client_secret: str,
    ) -> None:
        if self._env_oauth_configured():
            return
        await client_info_repo.save(
            ToolOAuthClientInfo(
                agent_id=self.agent.id,
                tool_id=self.id,
                client_id=client_id,
                client_secret=client_secret,
                token_endpoint_auth_method="client_secret_post",
                scope=self._oauth_config().scope,
            )
        )

    def _env_oauth_configured(self) -> bool:
        return bool(self._env_client_id() and self._env_client_secret())

    def _env_client_id(self) -> Optional[str]:
        return None

    def _env_client_secret(self) -> Optional[str]:
        return None

    # override this method in tools supporting several configs per agent to share a single oauth redirect
    # uri across all of them, instead of one redirect uri per config
    def _oauth_redirect_id(self) -> str:
        return self.id

    @asynccontextmanager
    async def load(self) -> AsyncIterator["OAuthAgentTool"]:
        self._oauth = await self._load_oauth()
        await self._oauth.solve_tokens()
        yield self

    async def auth(self, auth_callback: ToolAuthCallback):
        state = await ToolAuthRepository(self.db).find_state(
            self.user_id, self.id, cast(ToolOAuthCallback, auth_callback).state
        )
        if not state:
            raise ToolAuthCallbackError("OAuth state not found")
        oauth = await self._load_oauth()
        await oauth.callback(cast(ToolOAuthCallback, auth_callback), state)

    async def _load_oauth(self) -> AgentToolOauth:
        if not self._env_oauth_configured():
            client_info = await ToolOAuthClientInfoRepository(self.db).find_by_ids(self.agent.id, self.id)
            if not client_info or not client_info.scope:
                raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
        return self._build_oauth_client()

    def _build_oauth_client(self) -> AgentToolOauth:
        oauth_config = self._oauth_config()
        return AgentToolOauth(
            oauth_config.authority_base_url,
            oauth_config.metadata,
            oauth_config.scope + " offline_access",
            self.agent.id,
            self.id,
            self.user_id,
            self.db,
            redirect_uri=_build_redirect_uri(self._oauth_redirect_id()),
            client_info_provider=self._env_oauth_client_info if self._env_oauth_configured() else None,
        )

    def _env_oauth_client_info(self) -> OAuthClientInformationFull:
        return OAuthClientInformationFull(
            client_id=self._env_client_id(),
            client_secret=self._env_client_secret(),
            token_endpoint_auth_method="client_secret_post",
            redirect_uris=[AnyHttpUrl(_build_redirect_uri(self._oauth_redirect_id()))],
        )

    async def teardown(self):
        await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)
        await ToolOAuthClientInfoRepository(self.db).delete(self.agent.id, self.id)

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass
