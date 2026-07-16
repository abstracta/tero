from logging.config import fileConfig
from typing import Union, Any, Literal

from alembic import context
from alembic.autogenerate.api import AutogenContext
import alembic_postgresql_enum # noqa: F401
from sqlalchemy import engine_from_config
from sqlalchemy import pool

# need to add following type ignore to avoid intellij removing imports when organizing them, and warnings in vscode
from tero.agents.domain import *  # type: ignore
from tero.api_keys.domain import *  # type: ignore
from tero.agents.evaluators.domain import *  # type: ignore
from tero.agents.prompts.domain import *  # type: ignore
from tero.ai_models.domain import *  # type: ignore
from tero.core.env import env
from tero.core.repos import _EncryptedStrTypeDecorator
from tero.files.domain import *  # type: ignore
from tero.teams.domain import *  # type: ignore
from tero.threads.domain import *  # type: ignore
from tero.tools.docs.domain import *  # type: ignore
from tero.tools.jira import *  # type: ignore
from tero.tools.auth import *  # type: ignore
from tero.usage.domain import *  # type: ignore
from tero.users.domain import *  # type: ignore
from tero.external_agents.domain import *  # type: ignore
from tero.agents.test_cases.domain import *  # type: ignore

config = context.config
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata
db_url = env.db_url


def ignore_langchain_tables(name, type_, parent_names):
    return type_ != "table" or name not in ["upsertion_record", "langchain_pg_collection", "langchain_pg_embedding", "store", "store_migrations"]


def render_item(type_: str, obj: Any, autogen_metadata: AutogenContext) -> Union[str, Literal[False]]:
    if isinstance(obj, _EncryptedStrTypeDecorator):
        return "sqlmodel.AutoString()"
    # this avoids pyright complaining about sqlmodel.sql.sqltypes.AutoString not being valid
    elif hasattr(obj, '__class__') and obj.__class__.__module__ == 'sqlmodel.sql.sqltypes' and obj.__class__.__name__ == 'AutoString':
        params = f"length={obj.length}" if hasattr(obj, 'length') and obj.length is not None else ""
        return f"sqlmodel.AutoString({params})"
    return False


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        include_name=ignore_langchain_tables,
        include_schemas=False,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    engine_config = config.get_section(config.config_ini_section)
    if engine_config is None:
        raise ValueError("No engine config found")
    engine_config["sqlalchemy.url"] = db_url
    connectable = engine_from_config(
        engine_config,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata,
            include_name=ignore_langchain_tables,
            include_schemas=False,
            render_item=render_item
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
