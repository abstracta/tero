from enum import Enum
from typing import Optional, Any, List

from pydantic import BaseModel
from sqlmodel import Relationship, SQLModel, Field

from ..core.domain import CamelCaseModel


MY_TEAM_ID = 0
GLOBAL_TEAM_ID = 1


class Role(str, Enum):
    TEAM_OWNER = "owner"
    TEAM_MEMBER = "member"
    TEAM_EDITOR = "editor"
    BUDGET_OWNER = "budget_owner"


OWNER_ROLES = (Role.BUDGET_OWNER, Role.TEAM_OWNER)
EDITOR_ROLES = (*OWNER_ROLES, Role.TEAM_EDITOR)


class Team(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None)
    name: str = Field(max_length=100)
    team_roles: List["TeamRole"] = Relationship(back_populates="team")


class TeamRoleUpdate(BaseModel):
    role: Role


class TeamRoleStatus(str, Enum):
    ACCEPTED = "accepted"
    PENDING = "pending"
    REJECTED = "rejected"


class TeamRole(SQLModel, table=True):
    __tablename__: Any = "team_role"
    user_id: int = Field(foreign_key="user.id", index=True, primary_key=True)
    team_id: int = Field(foreign_key="team.id", index=True, primary_key=True)
    role: Role = Field(default=None, nullable=True)
    team: Optional[Team] = Relationship(back_populates="team_roles")
    status: TeamRoleStatus

    def update_with(self, update: TeamRoleUpdate):
        update_dict = update.model_dump()
        self.sqlmodel_update(update_dict)


class PublicTeamRole(BaseModel):
    id: int
    name: str
    role: Role
    status: TeamRoleStatus


class AddUsersToTeam(CamelCaseModel):
    username: str
    role: Role


class TeamUser(CamelCaseModel):
    id: int
    username: str
    name: Optional[str] = None
    role: Role
    role_status: TeamRoleStatus
    verified: bool = False
    exclude_from_metrics: bool = False
    monthly_usd_limit: Optional[int] = None


class TeamUpdate(CamelCaseModel):
    name: str


class TeamCreate(TeamUpdate):
    users: Optional[List[AddUsersToTeam]] = None
