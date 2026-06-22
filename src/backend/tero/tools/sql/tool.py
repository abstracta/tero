from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import List, Optional

from langchain.tools import BaseTool
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import AgentToolConfig
from ..auth import ToolAuthCallback
from ..core import AgentTool, load_schema
from .azure import AzureAuth
from .sql_backend import SqlBackend
from .sql_auth import SqlAuth
from .langchain_tools import ListDatabasesTool, ListSchemasTool, ListTablesTool, QueryTool, SchemaTool

SQL_TOOL_ID = "sql"

_AUTHS: dict[str, type[SqlAuth]] = {
    "azure": AzureAuth,
}


class SqlTool(AgentTool):
    id: str = SQL_TOOL_ID
    name: str = "SQL"
    description: str = "Query a SQL database using natural language"
    config_schema: dict = load_schema(__file__)
    _backend: Optional[SqlBackend] = None

    def _make_auth(self) -> SqlAuth:
        return _AUTHS[self.config["dbType"]](self)

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        await self._make_auth().setup(prev_config)

    async def teardown(self):
        await self._make_auth().teardown()

    @asynccontextmanager
    async def load(self) -> AsyncIterator["SqlTool"]:
        async with self._make_auth().get_backend() as backend:
            self._backend = backend
            try:
                yield self
            finally:
                self._backend = None

    async def auth(self, auth_callback: ToolAuthCallback):
        await self._make_auth().auth(auth_callback)

    async def build_langchain_tools(self) -> List[BaseTool]:
        if not self._backend:
            raise RuntimeError("SQL tool not loaded")
        backend = self._backend
        return [
            ListDatabasesTool(backend=backend),
            ListSchemasTool(backend=backend),
            ListTablesTool(backend=backend),
            SchemaTool(backend=backend),
            QueryTool(backend=backend),
        ]

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass
