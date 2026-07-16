import logging
import struct
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional, cast

from langchain_community.utilities import SQLDatabase
from pydantic import AnyHttpUrl
from sqlalchemy import Engine, create_engine, text

from ..auth import (
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
)
from .sql_backend import SqlBackend
from .sql_auth import SqlAuth

_AZURE_SQL_SCOPE = "https://database.windows.net/user_impersonation offline_access"
_SQL_COPT_SS_ACCESS_TOKEN = 1256
_CONNECT_RETRIES = 3
_CONNECT_RETRY_BACKOFF_S = 20

logger = logging.getLogger(__name__)


class AzureSqlBackend(SqlBackend):
    def __init__(self, server: str, access_token: str):
        self._server = server
        self._access_token = access_token
        self._engines: dict[str, Engine] = {}
        self._databases: dict[tuple[str, Optional[str]], SQLDatabase] = {}
        self._known_databases: Optional[list[str]] = None

    def _make_connection(self, database: str):
        import pyodbc

        token_bytes = self._access_token.encode("utf-16-le")
        token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
        conn_str = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={self._escape_odbc_value(self._server)};"
            f"DATABASE={self._escape_odbc_value(database)};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=60;"
        )
        # Serverless Azure SQL databases auto-pause when idle and take a while to
        # resume, so the first connection attempts time out (HYT00) until the
        # database is ready; retry with a backoff to ride out the resume.
        attempt = 0
        while True:
            try:
                return pyodbc.connect(conn_str, attrs_before={_SQL_COPT_SS_ACCESS_TOKEN: token_struct})
            except pyodbc.OperationalError as e:
                sqlstate = e.args[0] if e.args else ""
                if sqlstate != "HYT00" or attempt >= _CONNECT_RETRIES:
                    raise
                attempt += 1
                logger.warning("Azure SQL login timeout (HYT00) on %s/%s, retry %d/%d", self._server, database, attempt, _CONNECT_RETRIES)
                time.sleep(_CONNECT_RETRY_BACKOFF_S)

    def _escape_odbc_value(self, value: str) -> str:
        # ODBC brace-quoting: a value wrapped in {} is taken literally (";" loses its
        # separator meaning), with "}" escaped by doubling.
        return "{" + value.replace("}", "}}") + "}"

    def _get_engine(self, name: str) -> Engine:
        if name not in self._engines:
            self._engines[name] = create_engine(
                "mssql+pyodbc://",
                creator=lambda: self._make_connection(name),
                pool_pre_ping=True,
                pool_recycle=300,
            )
        return self._engines[name]

    def _validate_database(self, database: str):
        if self._known_databases is None:
            self._known_databases = self.list_databases()
        if database not in self._known_databases:
            raise ValueError(
                f"Unknown database '{database}'. "
                f"Available databases: {', '.join(self._known_databases)}"
            )

    def get_sql_database(self, name: str, schema: Optional[str] = None) -> SQLDatabase:
        self._validate_database(name)
        key = (name, schema)
        if key not in self._databases:
            self._databases[key] = SQLDatabase(self._get_engine(name), schema=schema)
        return self._databases[key]

    def list_databases(self) -> list[str]:
        engine = self._get_engine("master")
        query = text(
            "SELECT name FROM sys.databases WHERE database_id > 4 ORDER BY name"
        )
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(query)]

    def list_schemas(self, database: str) -> list[str]:
        self._validate_database(database)
        engine = self._get_engine(database)
        query = text(
            "SELECT DISTINCT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA', 'guest') "
            "ORDER BY TABLE_SCHEMA"
        )
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(query)]

    def close(self):
        for engine in self._engines.values():
            engine.dispose()
        self._engines.clear()
        self._databases.clear()


class AzureAuth(SqlAuth):
    def _auth_config(self, config: dict) -> dict:
        return {k: config[k] for k in ("server", "azureTenantId", "azureClientId") if k in config}

    async def setup(self, prev_config):
        tool = self._tool
        client_info_repo = ToolOAuthClientInfoRepository(tool.db)
        client_secret = tool._get_secret("azureClientSecret")
        if client_secret is None:
            existing = await client_info_repo.find_by_ids(tool.agent.id, tool.id)
            client_secret = existing.client_secret if existing else None
        if prev_config and self._auth_config(prev_config.config) != self._auth_config(tool.config):
            await self.teardown()
        await client_info_repo.save(ToolOAuthClientInfo(
            agent_id=tool.agent.id,
            tool_id=tool.id,
            client_id=tool.config["azureClientId"],
            client_secret=client_secret,
            token_endpoint_auth_method="client_secret_post",
            scope=_AZURE_SQL_SCOPE,
        ))
        async with tool.load():
            pass

    async def teardown(self):
        tool = self._tool
        await ToolAuthRepository(tool.db).delete_token(tool.user_id, tool.agent.id, tool.id)
        await ToolOAuthClientInfoRepository(tool.db).delete(tool.agent.id, tool.id)

    def _build_oauth(self) -> AgentToolOauth:
        tool = self._tool
        tenant_id = tool.config["azureTenantId"]
        base = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0"
        metadata = OAuthMetadata(
            issuer=AnyHttpUrl(f"https://login.microsoftonline.com/{tenant_id}/v2.0"),
            authorization_endpoint=AnyHttpUrl(f"{base}/authorize"),
            token_endpoint=AnyHttpUrl(f"{base}/token"),
            code_challenge_methods_supported=["S256"],
        )
        return AgentToolOauth(
            server_url=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            metadata=metadata,
            scope=_AZURE_SQL_SCOPE,
            agent_id=tool.agent.id,
            tool_id=tool.id,
            user_id=tool.user_id,
            db=tool.db,
        )

    @asynccontextmanager
    async def get_backend(self) -> AsyncIterator[AzureSqlBackend]:
        tool = self._tool
        oauth = self._build_oauth()
        tokens = await oauth.solve_tokens()
        if not tokens:
            raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=tool.id, agent_id=tool.agent.id))
        backend = AzureSqlBackend(tool.config["server"], tokens.access_token)
        try:
            yield backend
        finally:
            backend.close()

    async def auth(self, auth_callback: ToolAuthCallback):
        tool = self._tool
        state = await ToolAuthRepository(tool.db).find_state(
            tool.user_id,
            tool.id,
            cast(ToolOAuthCallback, auth_callback).state,
        )
        if not state:
            raise ToolAuthCallbackError("OAuth state not found")
        await self._build_oauth().callback(cast(ToolOAuthCallback, auth_callback), state)
