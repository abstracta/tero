from typing import ClassVar, Dict, Literal, Optional, get_args

from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import CallbackManagerForToolRun, Callbacks
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, create_model, model_validator

from ..core import StatusUpdateCallbackHandler, UserFacingToolError
from .sql_backend import DEFAULT_MAX_STRING_LENGTH, MaxStringLength, SqlBackend

_ALLOWED_LENGTHS_DESC = ", ".join(str(v) for v in get_args(MaxStringLength))


_DATABASE_FIELD_DESCRIPTION = (
    "The name of the database to operate on. Must be one of the names returned by "
    "sql_db_list_databases."
)

_SCHEMA_FIELD_DESCRIPTION = (
    "The database schema name to filter by. Use sql_db_list_schemas to discover available schemas. "
    "If omitted, uses the default schema (dbo)."
)

_ALIAS_FIELD_DESCRIPTION = "The alias of the SQL connection to operate on. One of: {aliases}."

# tool results are not replayed in later turns of the conversation, so anything the agent remembers about
# a database may actually belong to another one it inspected before
_FRESHNESS_DESCRIPTION = (
    "Every result of this tool is prefixed with the connection it comes from, and with the database and "
    "schema when they apply. Only trust those labels: never answer from databases, tables or columns "
    "mentioned earlier in the conversation, always call this tool again for the database you are asked about."
)


class _BackendTool(BaseTool):
    backends: Dict[str, SqlBackend] = Field(exclude=True)
    args_schema_template: ClassVar[Optional[type[BaseModel]]] = None

    # the class level description is kept as the user facing status text, so model only instructions are
    # appended here instead
    @model_validator(mode="after")
    def solve_alias_arg(self):
        aliases = list(self.backends)
        if len(aliases) > 1:
            self.description = f"{self.description} Use the alias argument to pick the SQL connection to use ({', '.join(aliases)})."
        self.description = f"{self.description} {_FRESHNESS_DESCRIPTION}"
        self.args_schema = _build_args_schema(self.args_schema_template, aliases)
        return self

    def _resolve_backend(self, alias: Optional[str]) -> SqlBackend:
        return self.backends[self._solve_alias(alias)]

    # results are prefixed with the source they come from, so the model never has to correlate them with
    # the arguments of its own tool call, which may have been trimmed away from the context (or dropped
    # altogether, since previous turns are replayed as plain text without their tool calls)
    def _labeled(
        self,
        result: str,
        alias: Optional[str],
        database: Optional[str] = None,
        db_schema: Optional[str] = None,
    ) -> str:
        parts = [f"connection={self._solve_alias(alias)}"]
        if database:
            parts.append(f"database={database}")
        if db_schema:
            parts.append(f"schema={db_schema}")
        return f"[{' '.join(parts)}]\n{result}"

    def _solve_alias(self, alias: Optional[str]) -> str:
        if not alias:
            if len(self.backends) == 1:
                return next(iter(self.backends))
            raise UserFacingToolError(f"An alias is required to pick the SQL connection to use. Available connections: {', '.join(self.backends)}")
        if alias not in self.backends:
            raise UserFacingToolError(f"Unknown SQL connection alias '{alias}'. Available connections: {', '.join(self.backends)}")
        return alias

    # when db_schema is omitted the tables come from the default schema of the connection user, so the
    # label states that one to keep the result fully scoped. it is read from the dialect (which resolves it
    # on connection) instead of assumed to be dbo, since a label is only useful while it is exact
    def _solve_schema(self, db: SQLDatabase, db_schema: Optional[str]) -> Optional[str]:
        return db_schema or db._inspector.default_schema_name

    def _wrap_errors(self, produce):
        try:
            return produce()
        except UserFacingToolError:
            raise
        except Exception as e:
            raise UserFacingToolError(str(e)) from e


def _build_args_schema(template: Optional[type[BaseModel]], aliases: list[str]) -> type[BaseModel]:
    description = _ALIAS_FIELD_DESCRIPTION.format(aliases=", ".join(aliases))
    if len(aliases) > 1:
        alias_type, default = Literal[tuple(aliases)], ...
    else:
        alias_type, default = Optional[str], None
    return create_model(
        f"{template.__name__ if template else 'SqlToolInput'}WithAlias",
        __base__=template,
        alias=(alias_type, Field(default, description=description)),  # type: ignore[call-overload]
    )


class _ListDatabasesInput(BaseModel):
    pass


class ListDatabasesTool(_BackendTool):
    name: str = "sql_db_list_databases"
    description: str = (
        "Lists the databases available on the SQL server. "
        "Output is a comma-separated list of database names. "
        "Call this first to discover which databases you can target."
    )
    args_schema_template: ClassVar[Optional[type[BaseModel]]] = _ListDatabasesInput
    args_schema: Optional[ArgsSchema] = _ListDatabasesInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        alias: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._labeled(", ".join(self._wrap_errors(lambda: self._resolve_backend(alias).list_databases())), alias)


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
    args_schema_template: ClassVar[Optional[type[BaseModel]]] = _ListSchemasInput
    args_schema: Optional[ArgsSchema] = _ListSchemasInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        alias: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        schemas = self._wrap_errors(lambda: self._resolve_backend(alias).list_schemas(database))
        return self._labeled(", ".join(schemas), alias, database)


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
    args_schema_template: ClassVar[Optional[type[BaseModel]]] = _ListTablesInput
    args_schema: Optional[ArgsSchema] = _ListTablesInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        db_schema: Optional[str] = None,
        alias: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        db = self._wrap_errors(lambda: self._resolve_backend(alias).get_sql_database(database, db_schema))
        tables = ", ".join(db.get_usable_table_names())
        return self._labeled(tables, alias, database, self._solve_schema(db, db_schema))


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
    args_schema_template: ClassVar[Optional[type[BaseModel]]] = _SchemaInput
    args_schema: Optional[ArgsSchema] = _SchemaInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        table_names: str,
        db_schema: Optional[str] = None,
        alias: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        db = self._wrap_errors(lambda:self._resolve_backend(alias).get_sql_database(database, db_schema))
        info = db.get_table_info_no_throw([t.strip() for t in table_names.split(",")])
        return self._labeled(info, alias, database, self._solve_schema(db, db_schema))


class _QueryInput(BaseModel):
    database: str = Field(..., description=_DATABASE_FIELD_DESCRIPTION)
    query: str = Field(..., description="A detailed and correct SQL query.")
    max_string_length: Optional[MaxStringLength] = Field(
        None,
        description=(
            f"Optional. Maximum characters returned per cell value (default {DEFAULT_MAX_STRING_LENGTH}). "
            f"Allowed values: {_ALLOWED_LENGTHS_DESC}."
        ),
    )


class QueryTool(_BackendTool):
    name: str = "sql_db_query"
    description: str = (
        "Executes a SQL query against a database and returns the result. "
        "Tables outside the default schema (dbo) must be schema-qualified in the query "
        "(e.g. SELECT * FROM sales.orders). "
        f"Optionally use max_string_length ({_ALLOWED_LENGTHS_DESC}) to override the default "
        f"({DEFAULT_MAX_STRING_LENGTH}) per-cell text limit. "
        "If an error is returned, rewrite the query, check the query, and try again."
    )
    args_schema_template: ClassVar[Optional[type[BaseModel]]] = _QueryInput
    args_schema: Optional[ArgsSchema] = _QueryInput
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]

    def _run(
        self,
        database: str,
        query: str,
        max_string_length: Optional[MaxStringLength] = None,
        alias: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        db = self._wrap_errors(lambda: self._resolve_backend(alias).get_sql_database(database, max_string_length=max_string_length))
        return self._labeled(str(db.run_no_throw(query)), alias, database)
