"""migrate sql tool to multiple connections per agent

Revision ID: c3d9e5a71b48
Revises: f6a7b8c9d0e1
Create Date: 2026-07-31 10:00:00.000000

"""

import hashlib
import json
import re
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d9e5a71b48"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_SQL_TOOL_ID = "sql"
# tables where the sql tool configs and their oauth data are keyed by tool id
_TOOL_ID_TABLES = (
    "agent_tool_config",
    "agent_tool_config_file",
    "tool_auth_token",
    "tool_oauth_client_info",
    "tool_oauth_state",
)
# keep in sync with tero.tools.sql.tool
_MAX_ALIAS_LENGTH = 30
_AZURE_SERVER_SUFFIX = ".database.windows.net"


def _build_alias_from_server(server: str) -> str:
    ret = server.strip().lower().split(",")[0]
    if ret.endswith(_AZURE_SERVER_SUFFIX):
        ret = ret[: -len(_AZURE_SERVER_SUFFIX)]
    ret = re.sub(r"[^a-z0-9_-]", "-", ret).lstrip("-_")
    if len(ret) > _MAX_ALIAS_LENGTH:
        ret = ret[: _MAX_ALIAS_LENGTH - 7] + "-" + hashlib.sha256(server.encode()).hexdigest()[:6]
    return ret or _SQL_TOOL_ID


def _rename_tool_id(agent_id: int, prev_tool_id: str, new_tool_id: str) -> None:
    for table in _TOOL_ID_TABLES:
        op.get_bind().execute(
            sa.text(f"UPDATE {table} SET tool_id = :new_tool_id WHERE agent_id = :agent_id AND tool_id = :prev_tool_id"),
            {"new_tool_id": new_tool_id, "agent_id": agent_id, "prev_tool_id": prev_tool_id},
        )


def _update_config(agent_id: int, tool_id: str, config: dict) -> None:
    op.get_bind().execute(
        sa.text("UPDATE agent_tool_config SET config = CAST(:config AS json) WHERE agent_id = :agent_id AND tool_id = :tool_id"),
        {"config": json.dumps(config), "agent_id": agent_id, "tool_id": tool_id},
    )


def upgrade() -> None:
    rows = op.get_bind().execute(
        sa.text("SELECT agent_id, config FROM agent_tool_config WHERE tool_id = :tool_id"),
        {"tool_id": _SQL_TOOL_ID},
    ).all()
    for agent_id, config in rows:
        config = dict(config or {})
        # the alias is not persisted so it keeps being derived from the server, as it is done for the configs
        # created after this revision. only the tool id, which is derived from it, is migrated
        alias = config.get("alias") or _build_alias_from_server(config.get("server", ""))
        _rename_tool_id(agent_id, _SQL_TOOL_ID, f"{_SQL_TOOL_ID}-{alias}")


def _delete_tool_id(agent_id: int, tool_id: str) -> None:
    for table in _TOOL_ID_TABLES:
        op.get_bind().execute(
            sa.text(f"DELETE FROM {table} WHERE agent_id = :agent_id AND tool_id = :tool_id"),
            {"agent_id": agent_id, "tool_id": tool_id},
        )


def downgrade() -> None:
    rows = op.get_bind().execute(
        sa.text("SELECT agent_id, tool_id, config FROM agent_tool_config WHERE tool_id LIKE :tool_id_pattern ORDER BY agent_id, tool_id"),
        {"tool_id_pattern": f"{_SQL_TOOL_ID}-%"},
    ).all()
    migrated_agent_ids = set()
    for agent_id, tool_id, config in rows:
        # only one sql connection per agent was supported, so the extra ones are dropped
        if agent_id in migrated_agent_ids:
            _delete_tool_id(agent_id, tool_id)
            continue
        migrated_agent_ids.add(agent_id)
        config = dict(config or {})
        config.pop("alias", None)
        _rename_tool_id(agent_id, tool_id, _SQL_TOOL_ID)
        _update_config(agent_id, _SQL_TOOL_ID, config)
