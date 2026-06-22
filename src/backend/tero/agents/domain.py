import abc
import base64
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Optional, cast

import sqlalchemy as sa
from sqlmodel import Field, Relationship, Index
from sqlmodel import SQLModel, Column, Text, JSON

from ..ai_models.domain import LlmModel, LlmModelType, LlmTemperature, ReasoningEffort
from ..core.domain import CamelCaseModel
from ..teams.domain import Role, Team
from ..users.domain import User, UserListItem


NAME_MAX_LENGTH = 30
CLONE_SUFFIX = "copy"


class BaseAgent(CamelCaseModel, abc.ABC):
    id: int = Field(primary_key=True, default=None)
    name: Optional[str] = Field(max_length=NAME_MAX_LENGTH, default=None)
    description: Optional[str] = Field(max_length=100, default=None)
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def set_default_name(self):
        self.name = f"Agent #{self.id}"


class AgentType(str, Enum):
    DEEP_AGENT = "DEEP_AGENT"
    REACT_AGENT = "REACT_AGENT"


class AgentUpdate(CamelCaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[LlmTemperature] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    recursion_limit: Optional[int] = None
    publish_prompts: Optional[bool] = None
    team_id: Optional[int] = None
    is_protected: Optional[bool] = None
    agent_type: Optional[AgentType] = None


class Agent(BaseAgent, table=True):
    __table_args__ = (
        Index('ix_agent_team_last_update', 'team_id', 'last_update'),
        Index('ix_agent_user_id_last_update', 'user_id', 'last_update'),
    )
    icon: Optional[bytes] = Field(default=None, sa_column=Column(sa.LargeBinary))
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    user: Optional[User] = Relationship()
    model_id: str = Field(foreign_key="llm_model.id")
    model: LlmModel = Relationship()
    system_prompt: str = Field(sa_column=Column(Text))
    temperature: LlmTemperature = LlmTemperature.NEUTRAL
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    recursion_limit: int = Field(default=20, ge=20, le=100)
    team_id: Optional[int] = Field(default=None, foreign_key="team.id")
    team: Optional[Team] = Relationship()
    evaluator_id: Optional[int] = Field(default=None, foreign_key="evaluator.id")
    is_protected: bool = Field(default=False)
    agent_type: AgentType = Field(default=AgentType.REACT_AGENT)

    def update_with(self, update: AgentUpdate):
        update_dict = update.model_dump(exclude_none=True)
        update_dict["icon"] = base64.b64decode(update_dict["icon"]) if update_dict.get("icon") else None
        update_dict["team_id"] = update.team_id
        self.sqlmodel_update(update_dict)
        self.last_update = datetime.now(timezone.utc)

    @property
    def model_temperature(self) -> Optional[float]:
        return self.temperature.get_float() if self.model.model_type == LlmModelType.CHAT else None

    @property
    def model_reasoning_effort(self) -> Optional[str]:
        return self.reasoning_effort.value.lower() if self.model.model_type == LlmModelType.REASONING else None

    def is_visible_by(self, user: User) -> bool:
        return self.user_id == user.id or user.is_member_of(cast(int, self.team_id))

    def is_editable_by(self, user: User) -> bool:
        return self.user_id == user.id or any(
            tr.role in [Role.TEAM_OWNER, Role.TEAM_EDITOR] and cast(Team, tr.team).id == self.team_id
            for tr in user.team_roles
        )

    def clone(self, user_id: int) -> "Agent":
        base_name = (self.name or "").strip()
        match = re.search(r"^(.*?)\s*\(copy(?: (\d+))?\)$", base_name)

        if match:
            name_prefix = match.group(1).strip()
            copy_number = int(match.group(2)) + 1 if match.group(2) else 2
            suffix = f"({CLONE_SUFFIX} {copy_number})"
        else:
            name_prefix = base_name
            suffix = f"({CLONE_SUFFIX})"

        truncated_prefix = name_prefix[:(NAME_MAX_LENGTH - len(suffix) - 1)].rstrip()

        return Agent(
            **self.model_dump(exclude={"id", "user_id", "team_id", "name", "last_update"}),
            name=f"{truncated_prefix} {suffix}".strip(),
            user_id=user_id,
            team_id=None
        )

class AgentListItem(BaseAgent):
    icon: Optional[str] = None
    active_users: Optional[int] = None
    can_edit: bool
    is_protected: bool = False
    user: Optional[UserListItem]
    team: Optional[Team] = None

    @staticmethod
    def from_agent(a: Agent, can_edit: bool, active_users: Optional[int] = None) -> 'AgentListItem':
        agent_dict = a.model_dump()
        agent_dict["icon"] = base64.b64encode(agent_dict["icon"]).decode("utf-8") if agent_dict.get("icon") else None
        agent_dict["user"] = UserListItem.from_user(a.user)
        agent_dict["active_users"] = active_users
        agent_dict["can_edit"] = can_edit
        agent_dict["is_protected"] = a.is_protected
        agent_dict["team"] = a.team
        return AgentListItem.model_validate(agent_dict)


class PublicAgent(BaseAgent):
    icon: Optional[str]
    user_id: Optional[int]
    can_edit: bool
    is_protected: bool = False
    model_id: Optional[str] = None
    system_prompt: str = ""
    temperature: Optional[LlmTemperature] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    recursion_limit: Optional[int] = None
    user: Optional[UserListItem] = None
    team: Optional[Team] = None
    agent_type: AgentType = AgentType.REACT_AGENT

    @staticmethod
    def from_agent(a: Agent, can_edit: bool) -> 'PublicAgent':
        agent_dict = a.model_dump()
        agent_dict["icon"] = base64.b64encode(agent_dict["icon"]).decode("utf-8") if agent_dict.get("icon") else None
        agent_dict["can_edit"] = can_edit
        agent_dict["is_protected"] = a.is_protected
        agent_dict["user"] = UserListItem.from_user(a.user) if a.user else None
        agent_dict["team"] = a.team

        if a.is_protected and not can_edit:
            agent_dict["system_prompt"] = ""
            agent_dict["model_id"] = None
            agent_dict["temperature"] = None
            agent_dict["reasoning_effort"] = None
            agent_dict["recursion_limit"] = None

        return PublicAgent.model_validate(agent_dict)


class UserAgent(SQLModel, table=True):
    __tablename__ : Any = "user_agent"
    __table_args__ = (
        Index('ix_user_agent_user_id_creation', 'user_id', 'creation'),
    )
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    agent_id: int = Field(foreign_key="agent.id", primary_key=True)
    creation: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseAgentToolConfig(CamelCaseModel, abc.ABC):
    agent_id: int = Field(foreign_key="agent.id", primary_key=True)
    tool_id: str = Field(primary_key=True, max_length=60)
    config: dict = Field(sa_column=Column(JSON))


class AgentToolConfig(BaseAgentToolConfig, table=True):
    __tablename__ : Any = "agent_tool_config"
    agent: Agent = Relationship()
    draft: bool = Field(default=False, index=True)

    def clone(self, agent_id: int) -> 'AgentToolConfig':
        return AgentToolConfig(
            draft=self.draft,
            agent_id=agent_id,
            tool_id=self.tool_id,
            config=self.config
        )


class AgentToolConfigFile(CamelCaseModel, table=True):
    __tablename__ : Any = "agent_tool_config_file"
    agent_id: int = Field(foreign_key="agent.id", primary_key=True)
    tool_id: str = Field(primary_key=True, max_length=60)
    file_id: int = Field(foreign_key="file.id", primary_key=True)


class AutomaticAgentField(Enum):
    NAME = 'NAME'
    DESCRIPTION = 'DESCRIPTION'
    SYSTEM_PROMPT = 'SYSTEM_PROMPT'
