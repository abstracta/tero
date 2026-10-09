import abc
from datetime import datetime, timezone
from typing import Optional, List, cast

from sqlmodel import SQLModel, Field, Relationship

from pydantic import Field as PydanticField, model_validator

from ..core.domain import CamelCaseModel
from ..teams.domain import PublicTeamRole, Team, TeamRole, GLOBAL_TEAM_ID, EDITOR_ROLES, OWNER_ROLES, TeamRoleStatus, Role


class BaseUser(SQLModel, abc.ABC):
    id: int = Field(primary_key=True, default=None)
    username: str = Field(index=True, max_length=50)
    name: Optional[str] = Field(max_length=100, default=None, index=True)


class User(BaseUser, table=True):
    monthly_usd_limit: int
    monthly_hours: int = Field(default=160)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deleted_at: Optional[datetime] = Field(default=None)
    exclude_from_metrics: bool = Field(default=False)
    team_roles: List[TeamRole] = Relationship()

    def is_active(self) -> bool:
        if self.deleted_at is None:
            return True
        deleted_at = self.deleted_at
        # Records inserted directly into the database may lack timezone info,
        # for that reason we set it to the current timezone if it's not set
        if deleted_at.tzinfo is None:
            deleted_at = deleted_at.replace(tzinfo=timezone.utc)
        return deleted_at > datetime.now(timezone.utc)

    def is_member_of(self, team_id: int) -> bool:
        return team_id == GLOBAL_TEAM_ID or any(cast(Team, tr.team).id == team_id and tr.status == TeamRoleStatus.ACCEPTED for tr in self.team_roles)

    def is_editor_of(self, team_id: int) -> bool:
        return any(
            tr.role in EDITOR_ROLES
            and tr.status == TeamRoleStatus.ACCEPTED
            and tr.team_id == team_id
            for tr in self.team_roles
        )

    def is_global_owner(self) -> bool:
        return any(
            tr.role in OWNER_ROLES
            and tr.team_id == GLOBAL_TEAM_ID
            and tr.status == TeamRoleStatus.ACCEPTED
            for tr in self.team_roles
        )

    def is_budget_owner(self) -> bool:
        return any(
            tr.role == Role.BUDGET_OWNER
            and tr.team_id == GLOBAL_TEAM_ID
            and tr.status == TeamRoleStatus.ACCEPTED
            for tr in self.team_roles
        )


class UserListItem(BaseUser):

    @staticmethod
    def from_user(user: Optional[User]) -> Optional['UserListItem']:
        return UserListItem.model_validate(user) if user else None


class UserUpdate(CamelCaseModel):
    exclude_from_metrics: Optional[bool] = None
    monthly_usd_limit: Optional[int] = PydanticField(default=None, gt=0)

    @model_validator(mode="after")
    def require_at_least_one_field(self):
        if self.exclude_from_metrics is None and self.monthly_usd_limit is None:
            raise ValueError("At least one field is required")
        return self


class UserProfile(SQLModel):
    teams: List[PublicTeamRole]

    @staticmethod
    def from_user(user: User) -> 'UserProfile':
        return UserProfile(
            teams=[
                PublicTeamRole(
                    id=tr.team.id,
                    name=tr.team.name,
                    role=tr.role,
                    status=tr.status
                )
                for tr in user.team_roles if tr.team and tr.status in [TeamRoleStatus.PENDING, TeamRoleStatus.ACCEPTED]
            ]
        )
