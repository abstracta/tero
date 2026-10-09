import abc
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, Text, Index
from sqlmodel import Field

from ...core.domain import CamelCaseModel


class BaseAgentPrompt(CamelCaseModel, abc.ABC):
    name: Optional[str] = Field(max_length=50, default=None)
    content: Optional[str] = Field(sa_column=Column(Text), default=None)
    shared: bool = Field(default=False, index=True)
    collapsed: bool = Field(default=False)


class AgentPromptCreate(BaseAgentPrompt):
    name: str # type: ignore
    content: str # type: ignore
    starter: bool = Field(default=False)


class AgentPromptUpdate(BaseAgentPrompt):
    pass


class AgentPromptPublic(BaseAgentPrompt):
    id: int
    last_update: datetime
    user_id: int
    can_edit: bool = False
    starter: bool = Field(default=False)


class AgentPrompt(BaseAgentPrompt, table=True):
    __tablename__ : Any = "agent_prompt"
    __table_args__ = (
        Index('ix_agent_prompt_agent_id_user_id_shared', 'agent_id', 'user_id', 'shared'),
        Index('ix_agent_prompt_user_id_last_update_shared', 'user_id', 'last_update', 'shared'),
    )
    id: int = Field(primary_key=True, default=None)
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: int = Field(foreign_key="agent.id")
    user_id: int = Field(foreign_key="user.id")
    starter: bool = Field(default=False)

    def update_with(self, update: AgentPromptUpdate, user_id: int):
        self.sqlmodel_update(update.model_dump(exclude_unset=True))
        self.last_update = datetime.now(timezone.utc)
        self.user_id = user_id

    def clone(self, agent_id: int, user_id: int) -> "AgentPrompt":
        data = self.model_dump(exclude={"id", "agent_id", "user_id", "last_update"})
        data["agent_id"] = agent_id
        data["user_id"] = user_id
        return AgentPrompt(**data)
