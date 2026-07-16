from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun, Callbacks
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from ..core import StatusUpdateCallbackHandler
from .sql_backend import SqlBackend


_DATABASE_FIELD_DESCRIPTION = (
    "The name of the database to operate on. Must be one of the names returned by "
    "sql_db_list_databases."
)

_SCHEMA_FIELD_DESCRIPTION = (
    "The database schema name to filter by. Use sql_db_list_schemas to discover available schemas. "
    "If omitted, uses the default schema (dbo)."
)


class _BackendTool(BaseTool):
    backend: SqlBackend = Field(exclude=True)


class ListDatabasesTool(_BackendTool):
    name: str = "sql_db_list_databases"
    description: str = (
        "Lists the databases available on the SQL server. "
        "Takes no input, output is a comma-separated list of database names. "
        "Call this first to discover which databases you can target."
    )
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return ", ".join(self.backend.list_databases())


class _ListSchemasInput(BaseModel):
    database: str = Field(..., description=_DATABASE_FIELD_DESCRIPTION)


class ListSchemasTool(_BackendTool):
    name: str = "sql_db_list_schemas"
    description: str = (
        "Lists the schemas of a database that contain tables. "
        "Output is a comma-separated list of schema names. "
        "Use this after sql_db_list_databases to discover non-default schemas before listing "
        "tables or querying them."
    )
    args_schema: Optional[ArgsSchema] = _ListSchemasInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return ", ".join(self.backend.list_schemas(database))


class _ListTablesInput(BaseModel):
    database: str = Field(..., description=_DATABASE_FIELD_DESCRIPTION)
    db_schema: Optional[str] = Field(None, description=_SCHEMA_FIELD_DESCRIPTION)


class ListTablesTool(_BackendTool):
    name: str = "sql_db_list_tables"
    description: str = (
        "Lists the tables in a database and schema. "
        "Output is a comma-separated list of table names. "
        "Use sql_db_list_databases first to find valid database names, and sql_db_list_schemas "
        "to find available schemas."
    )
    args_schema: Optional[ArgsSchema] = _ListTablesInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        db_schema: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        db = self.backend.get_sql_database(database, db_schema)
        return ", ".join(db.get_usable_table_names())


class _SchemaInput(BaseModel):
    database: str = Field(..., description=_DATABASE_FIELD_DESCRIPTION)
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )
    db_schema: Optional[str] = Field(None, description=_SCHEMA_FIELD_DESCRIPTION)


class SchemaTool(_BackendTool):
    name: str = "sql_db_schema"
    description: str = (
        "Gets the structure (columns and types) and sample rows of database tables. "
        "Input is the database and a comma-separated list of table names; "
        "make sure the tables exist by calling sql_db_list_tables first."
    )
    args_schema: Optional[ArgsSchema] = _SchemaInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        table_names: str,
        db_schema: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        db = self.backend.get_sql_database(database, db_schema)
        return db.get_table_info_no_throw([t.strip() for t in table_names.split(",")])


class _QueryInput(BaseModel):
    database: str = Field(..., description=_DATABASE_FIELD_DESCRIPTION)
    query: str = Field(..., description="A detailed and correct SQL query.")


class QueryTool(_BackendTool):
    name: str = "sql_db_query"
    description: str = (
        "Executes a SQL query against a database and returns the result. "
        "Tables outside the default schema (dbo) must be schema-qualified in the query "
        "(e.g. SELECT * FROM sales.orders). "
        "If an error is returned, rewrite the query, check the query, and try again."
    )
    args_schema: Optional[ArgsSchema] = _QueryInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        db = self.backend.get_sql_database(database)
        return str(db.run_no_throw(query))
