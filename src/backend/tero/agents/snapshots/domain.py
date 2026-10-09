import base64
from datetime import datetime, timezone
from typing import Any, Optional

import sqlalchemy as sa
from sqlmodel import Column, Field, Relationship, SQLModel

from ...teams.domain import Team
from ..domain import AgentListItem, NAME_MAX_LENGTH


class AgentSnapshot(SQLModel, table=True):
    __tablename__: Any = "agent_snapshot"
    agent_id: int = Field(foreign_key="agent.id", primary_key=True, ondelete="CASCADE")
    team_id: int = Field(foreign_key="team.id", primary_key=True, ondelete="CASCADE")
    name: Optional[str] = Field(max_length=NAME_MAX_LENGTH, default=None)
    icon: Optional[bytes] = Field(default=None, sa_column=Column(sa.LargeBinary))
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    team: Optional[Team] = Relationship()


def to_agent_list_item(snapshot: AgentSnapshot) -> AgentListItem:
    return AgentListItem(
        id=snapshot.agent_id,
        name=snapshot.name,
        last_update=snapshot.last_update,
        team=snapshot.team,
        icon=base64.b64encode(snapshot.icon).decode("utf-8") if snapshot.icon else None,
        can_edit=False,
        user=None,
        read_only=True,
    )
