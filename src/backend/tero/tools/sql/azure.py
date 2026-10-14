import logging
import struct
import time
from typing import Optional

from langchain_community.utilities import SQLDatabase
from sqlalchemy import Engine, create_engine, text

from ..core import UserFacingToolError
from .sql_backend import DEFAULT_MAX_STRING_LENGTH, SqlBackend

_SQL_COPT_SS_ACCESS_TOKEN = 1256
_CONNECT_RETRIES = 3
_CONNECT_RETRY_BACKOFF_S = 20

logger = logging.getLogger(__name__)


class AzureSqlBackend(SqlBackend):
    def __init__(self, server: str, access_token: str):
        self._server = server
        self._access_token = access_token
        self._engines: dict[str, Engine] = {}
        self._databases: dict[tuple[str, Optional[str], int], SQLDatabase] = {}
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
            raise UserFacingToolError(
                f"Unknown database '{database}'. "
                f"Available databases: {', '.join(self._known_databases)}"
            )

    def get_sql_database(
        self, name: str, schema: Optional[str] = None, max_string_length: Optional[int] = None
    ) -> SQLDatabase:
        self._validate_database(name)
        limit = max_string_length or DEFAULT_MAX_STRING_LENGTH
        key = (name, schema, limit)
        if key not in self._databases:
            self._databases[key] = SQLDatabase(
                self._get_engine(name), schema=schema, max_string_length=limit
            )
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
