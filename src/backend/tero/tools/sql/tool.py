from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import re
from typing import Dict, List, Optional, cast

from langchain.tools import BaseTool
from pydantic import model_validator
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import Agent
from ...core.env import env
from ..auth import ToolAuthRequestException, ToolAuthTokenRequest
from ..core import AgentTool, load_schema
from ..oauth_tool import OAuthAgentTool, OAuthToolConfig, build_azure_oauth_config
from .azure import AzureSqlBackend
from .sql_backend import SqlBackend
from .langchain_tools import ListDatabasesTool, ListSchemasTool, ListTablesTool, QueryTool, SchemaTool

SQL_TOOL_ID = "sql"
_AZURE_SQL_SCOPE = "https://database.windows.net/user_impersonation"
# keep in sync with the alias pattern in tool-schema.json, and with the alembic revision migrating legacy
# sql tool ids. the limit keeps "sql-{alias}" within the tool id column length
MAX_ALIAS_LENGTH = 30
_AZURE_SERVER_SUFFIX = ".database.windows.net"


def solve_sql_alias(config: dict) -> str:
    return cast(str, config.get("alias") or "").strip() or build_alias_from_server(cast(str, config.get("server", "")))


def build_alias_from_server(server: str) -> str:
    ret = server.strip().lower().split(",")[0]
    if ret.endswith(_AZURE_SERVER_SUFFIX):
        ret = ret[: -len(_AZURE_SERVER_SUFFIX)]
    ret = re.sub(r"[^a-z0-9_-]", "-", ret).lstrip("-_")
    if len(ret) > MAX_ALIAS_LENGTH:
        ret = ret[: MAX_ALIAS_LENGTH - 7] + "-" + hashlib.sha256(server.encode()).hexdigest()[:6]
    return ret or SQL_TOOL_ID


class SqlTool(OAuthAgentTool):
    id: str = SQL_TOOL_ID + "-*"
    name: str = "SQL"
    description: str = "Query a SQL database using natural language"
    config_schema: dict = load_schema(__file__)
    _client_secret_config_key = "azureClientSecret"
    _client_id_config_key = "azureClientId"
    _backend: Optional[SqlBackend] = None

    @model_validator(mode="after")
    def hide_oauth_config_when_env_set(self):
        if self._env_oauth_configured():
            for key in ("azureTenantId", "azureClientId", "azureClientSecret"):
                self.config_schema["properties"].pop(key, None)
                if key in self.config_schema.get("required", []):
                    self.config_schema["required"].remove(key)
        return self

    def _env_oauth_configured(self) -> bool:
        return bool(self._env_tenant_id() and super()._env_oauth_configured())

    def _env_tenant_id(self) -> Optional[str]:
        return env.sql_tool_azure_tenant_id

    def _env_client_id(self) -> Optional[str]:
        return env.sql_tool_azure_client_id

    def _env_client_secret(self) -> Optional[str]:
        return env.sql_tool_azure_client_secret.get_secret_value() if env.sql_tool_azure_client_secret else None

    def configure(self, agent: Agent, user_id: int, config: dict, db: AsyncSession, thread_id: Optional[int] = None):
        super().configure(agent, user_id, config, db, thread_id)
        self.id = f"{SQL_TOOL_ID}-{self.alias}"

    # the alias is not persisted when it is derived from the server, so the user keeps seeing the alias field
    # empty until an explicit one is set. this means the tool id follows the server, and since the tool id is
    # part of the key of the data associated to the config, such data is renamed when saving the config
    @property
    def alias(self) -> str:
        return solve_sql_alias(self.config)

    # all the sql connections of an agent share the same oauth redirect uri, so a single redirect uri needs
    # to be registered in the azure app registration no matter how many servers are configured
    def _oauth_redirect_id(self) -> str:
        return SQL_TOOL_ID

    def _oauth_config(self) -> OAuthToolConfig:
        tenant_id = self._env_tenant_id() or self.config["azureTenantId"]
        return build_azure_oauth_config(tenant_id, _AZURE_SQL_SCOPE)

    @asynccontextmanager
    async def load(self) -> AsyncIterator["SqlTool"]:
        self._oauth = await self._load_oauth()
        tokens = await self._oauth.solve_tokens()
        if not tokens:
            raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
        backend = AzureSqlBackend(self.config["server"], tokens.access_token)
        self._backend = backend
        try:
            yield self
        finally:
            self._backend = None
            backend.close()

    async def build_langchain_tools(self) -> List[BaseTool]:
        return await SqlTool.build_shared_langchain_tools([self])

    @classmethod
    async def build_shared_langchain_tools(cls, instances: List[AgentTool]) -> List[BaseTool]:
        backends: Dict[str, SqlBackend] = {}
        for instance in instances:
            sql_tool = cast(SqlTool, instance)
            if not sql_tool._backend:
                raise RuntimeError("SQL tool not loaded")
            backends[sql_tool.alias] = sql_tool._backend
        return [
            ListDatabasesTool(backends=backends),
            ListSchemasTool(backends=backends),
            ListTablesTool(backends=backends),
            SchemaTool(backends=backends),
            QueryTool(backends=backends),
        ]

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass
