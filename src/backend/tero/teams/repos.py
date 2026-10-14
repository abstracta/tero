from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import selectinload
from sqlmodel import select, delete, and_, col, or_
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.repos import scalar, attr
from ..users.domain import User
from .domain import Team, TeamRole, TeamRoleStatus, TeamUser, Role, GLOBAL_TEAM_ID


class TeamRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_team(self, team_id: int) -> Optional[Team]:
        query = select(Team).where(Team.id == team_id)
        result = await self._db.exec(query)
        return result.one_or_none()

    async def find_team_users(self, team_id: int, current_user: User, limit: Optional[int] = None, offset: Optional[int] = None, search: Optional[str] = None) -> List[TeamUser]:
        query = (select(User, TeamRole.role, TeamRole.status))

        if team_id == GLOBAL_TEAM_ID:
            query = query.outerjoin(TeamRole, and_(User.id == TeamRole.user_id, TeamRole.team_id == team_id))
        else:
            query = query.join(TeamRole, and_(User.id == TeamRole.user_id)).where(TeamRole.team_id == team_id)
        query = query.where(and_(
            or_(col(User.deleted_at).is_(None), col(User.deleted_at) > datetime.now(timezone.utc)),
            col(User.name).ilike(f"%{search}%") if search else True
        )).limit(limit).offset(offset).order_by(col(User.name))

        result = await self._db.exec(query)
        rows = result.all()
        include_monthly_usd_limit = current_user.is_budget_owner()

        return [
            TeamUser(
                id=user.id,
                name=user.name,
                username=user.username,
                role=Role(role) if role else Role.TEAM_MEMBER,
                role_status=TeamRoleStatus(status) if status else TeamRoleStatus.ACCEPTED,
                verified=user.name is not None,
                exclude_from_metrics=user.exclude_from_metrics,
                monthly_usd_limit=user.monthly_usd_limit if include_monthly_usd_limit else None,
            )
            for user, role, status in rows
        ]

    async def find_team_role(self, team_id: int, user_id: int) -> Optional[TeamRole]:
        query = select(TeamRole).where(TeamRole.team_id == team_id, TeamRole.user_id == user_id)
        result = await self._db.exec(query)
        return result.one_or_none()

    async def find_all_pending_user_invitations(self, user_id: int) -> List[TeamRole]:
        query = select(TeamRole).where(TeamRole.user_id == user_id, TeamRole.status == TeamRoleStatus.PENDING).options(selectinload(attr(TeamRole.team)))
        result = await self._db.exec(query)
        return list(result.all())

    async def save_team_role(self, team_role: TeamRole):
        await self._db.merge(team_role)
        await self._db.commit()

    async def delete_team_role(self, team_id: int, user_id: int):
        stmt = (
            delete(TeamRole)
            .where(and_(TeamRole.team_id == team_id, TeamRole.user_id == user_id))
        )
        await self._db.exec(scalar(stmt))
        await self._db.commit()

    async def find_teams(self) -> List[Team]:
        stmt = (
            select(Team)
            .order_by(col(Team.name))
        )
        ret = await self._db.exec(stmt)
        return [t for t in ret.all()]

    async def add(self, team: Team) -> Team:
        self._db.add(team)
        await self._db.commit()
        await self._db.refresh(team)
        return team

    async def update(self, team: Team):
        await self._db.merge(team)
        await self._db.commit()

    async def remove_team_roles(self, team_id: int):
        stmt = (
            delete(TeamRole)
            .where(col(TeamRole.team_id) == team_id)
        )
        await self._db.exec(scalar(stmt))
        await self._db.commit()

    async def delete(self, team: Team):
        await self._db.delete(team)
        await self._db.commit()

    async def find_user_team_roles(self, user_id: int) -> List[TeamRole]:
        query = select(TeamRole).where(TeamRole.user_id == user_id)
        result = await self._db.exec(query)
        return list(result.all())
