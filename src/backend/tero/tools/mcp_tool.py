import abc
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from enum import Enum
import logging
from typing import Any, Mapping, Optional, cast

from httpx import HTTPStatusError
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.shared.session import RequestResponder
from mcp.types import ServerRequest, ClientResult, ServerNotification
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.domain import AgentToolConfig
from ..tools.auth import (
    AgentToolOauth,
    ToolAuthCallback,
    ToolAuthCallbackError,
    ToolAuthRepository,
    ToolAuthRequestException,
    ToolAuthToken,
    ToolAuthTokenType,
    ToolAuthTokenCallback,
    ToolAuthTokenRequest,
    ToolOAuthCallback,
    ToolOAuthClientInfoRepository,
)
from .core import AgentTool, StatusUpdateCallbackHandler


class AuthType(str, Enum):
    OAUTH = "oauth"
    BEARER_TOKEN = "bearerToken"

    @classmethod
    def from_string(cls, value: str) -> "AuthType":
        if value == "oauth":
            return cls.OAUTH
        if value == "bearerToken":
            return cls.BEARER_TOKEN
        raise ValueError(f"Invalid auth type: {value}")

logger = logging.getLogger(__name__)


class BaseMcpTool(AgentTool, abc.ABC):
    _tools: Optional[list[BaseTool]] = None
    _oauth: Optional[AgentToolOauth] = None
    _pending_bearer_token: Optional[str] = None
    _bearer_token: Optional[str] = None
    _bearer_secret_key: str = "token"

    @abc.abstractmethod
    def _get_mcp_server_url(self) -> str:
        pass

    @abc.abstractmethod
    def _get_auth_type(self) -> "AuthType":
        pass

    async def _setup_tool(self, prev_config: Optional["AgentToolConfig"]):
        self._oauth = None
        self._bearer_token = self._get_secret(self._bearer_secret_key)
        prev_token = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        if (
            prev_config
            and self._auth_config(prev_config.config) != self._auth_config(self.config)
            or self._get_auth_type() == AuthType.BEARER_TOKEN
            and self._bearer_token
            and prev_token
            and self._bearer_token != prev_token.access_token
        ):
            await self.teardown()
        if self._get_auth_type() == AuthType.BEARER_TOKEN and self._bearer_token:
            await self._save_bearer_token(self._bearer_token, self.db)
        await self._check_auth()

    async def teardown(self):
        await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)
        if self._get_auth_type() == AuthType.OAUTH:
            await ToolOAuthClientInfoRepository(self.db).delete(self.agent.id, self.id)

    async def _check_auth(self):
        async with self.load():
            pass

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass

    @asynccontextmanager
    async def load(self) -> AsyncIterator["BaseMcpTool"]:
        async def error_raising_message_handler(
            message: RequestResponder[ServerRequest, ClientResult] | ServerNotification | Exception,
        ) -> None:
            if isinstance(message, Exception):
                raise message

        server_url = self._get_mcp_server_url()
        auth_token = await self._get_auth_token()
        connection_id = self._get_connection_id()
        base_config: dict[str, Any] = {
            "transport": self._get_transport(),
            "url": server_url,
            "headers": self._build_headers(await self._get_auth_headers(auth_token)),
            "session_kwargs": {
                "message_handler": error_raising_message_handler,
                "read_timeout_seconds": timedelta(seconds=30),
            },
        }
        connections: Mapping[str, Any] = {connection_id: base_config}
        try:
            async with MultiServerMCPClient(connections).session(connection_id) as mcp_session:
                tools = await self._build_tools(mcp_session)
                self._tools = self._fix_tools_schemas(tools)
                for tool in self._tools:
                    tool.callbacks = [StatusUpdateCallbackHandler(tool.name, description=tool.description)]
                yield self
        except ExceptionGroup as exception:
            if any(
                isinstance(e, HTTPStatusError) and e.response.status_code == 401
                for e in exception.exceptions
            ):
                raise ToolAuthRequestException(
                    ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id)
                ) from exception
            raise

    async def _get_auth_token(self) -> Optional[str]:
        auth_type = self._get_auth_type()
        if auth_type == AuthType.BEARER_TOKEN:
            bearer = await self._get_bearer_token()
            return f"Bearer {bearer}"
        if auth_type == AuthType.OAUTH:
            oauth = self._get_oauth()
            tokens = await oauth.solve_tokens()
            return f"Bearer {tokens.access_token}" if tokens else None
        raise ValueError("Invalid auth type")

    async def _get_bearer_token(self) -> str:
        if self._pending_bearer_token:
            return self._pending_bearer_token
        if self._bearer_token:
            return self._bearer_token
        token = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        if not token:
            raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
        return token.access_token

    def _get_oauth(self) -> AgentToolOauth:
        if not self._oauth:
            server_url = self._get_mcp_server_url()
            self._oauth = AgentToolOauth(server_url, None, None, self.agent.id, self.id, self.user_id, self.db)
        return self._oauth

    def _get_transport(self) -> str:
        url = self._get_mcp_server_url()
        return "sse" if url.endswith("/sse") else "streamable_http"

    def _get_connection_id(self) -> str:
        return self.id

    async def _get_auth_headers(self, auth_token: Optional[str] = None) -> dict[str, str]:
        if auth_token:
            return {"Authorization": auth_token}
        return {}

    def _build_headers(self, auth_headers: dict[str, str]) -> dict[str, str]:
        headers = dict(auth_headers)
        for header in self.config.get("customHeaders", []):
            headers[header["name"]] = header["value"]
        return headers

    def _auth_config(self, config: dict) -> dict:
        non_auth_keys = ["customHeaders"]
        return {k: v for k, v in config.items() if k not in non_auth_keys}

    async def _build_tools(self, mcp_session: ClientSession) -> list[BaseTool]:
        return await load_mcp_tools(mcp_session)

    def _fix_tools_schemas(self, tools: list[BaseTool]) -> list[BaseTool]:
        ret = []
        for tool in tools:
            tool.args_schema = self._fix_schema(tool.args_schema)
            ret.append(tool)
        return ret

    def _fix_schema(self, schema: Any) -> Any:
        if isinstance(schema, dict):
            items = schema.get("items", {})
            if schema.get("type") == "array" and not items.get("type"):
                if not items:
                    schema["items"] = {"type": "string"}
                else:
                    schema["items"]["type"] = "string"
                return schema
            if schema.get("type") == "object" and not schema.get("properties"):
                schema["properties"] = {}
                return schema
            return {key: self._fix_schema(value) for key, value in schema.items()}
        if isinstance(schema, list):
            return [self._fix_schema(item) for item in schema]
        return schema

    async def _save_bearer_token(self, token: Optional[str], db: AsyncSession):
        if not token:
            return
        auth_token = ToolAuthToken(
            user_id=self.user_id,
            agent_id=self.agent.id,
            tool_id=self.id,
            access_token=token,
            token_type=ToolAuthTokenType.BEARER,
            expires_in=None,
            scope=None,
            refresh_token=None,
            expires_at=None,
        )
        await ToolAuthRepository(db).save_token(auth_token)

    async def build_langchain_tools(self) -> list[BaseTool]:
        if not self._tools:
            raise RuntimeError(f"{self.name} tool has not been set up properly")
        return self._tools

    async def auth(self, auth_callback: ToolAuthCallback):
        auth_type = self._get_auth_type()
        if auth_type == AuthType.OAUTH:
            state = await ToolAuthRepository(self.db).find_state(
                self.user_id,
                self.id,
                cast(ToolOAuthCallback, auth_callback).state,
            )
            if not state:
                raise ToolAuthCallbackError("OAuth state not found")
            await self._get_oauth().callback(cast(ToolOAuthCallback, auth_callback), state)
            return

        if auth_type == AuthType.BEARER_TOKEN:
            self._pending_bearer_token = cast(ToolAuthTokenCallback, auth_callback).auth_token
            try:
                await self._check_auth()
            except Exception:
                logger.warning(
                    f"Invalid token for agent {self.agent.id} tool {self.id} user {self.user_id}",
                    exc_info=True,
                )
                self._pending_bearer_token = None
                raise ToolAuthCallbackError("Invalid token")
            await self._save_bearer_token(self._pending_bearer_token, self.db)
            self._pending_bearer_token = None
            return

        raise ValueError("Invalid auth type")

