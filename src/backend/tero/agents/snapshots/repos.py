from typing import Optional

from sqlalchemy.orm import selectinload
from sqlmodel import col, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ...core.repos import attr, scalar
from ...teams.domain import GLOBAL_TEAM_ID, TeamRoleStatus
from ...users.domain import User
from ..domain import Agent
from .domain import AgentSnapshot


def get_visible_team_ids(user: User) -> list[int]:
    return [GLOBAL_TEAM_ID, *(role.team_id for role in user.team_roles if role.status == TeamRoleStatus.ACCEPTED)]


class AgentSnapshotRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def save_previous_team_snapshot(self, agent: Agent, new_team_id: Optional[int]):
        await self._db.exec(scalar(delete(AgentSnapshot).where(
            col(AgentSnapshot.agent_id) == agent.id, col(AgentSnapshot.team_id).in_([agent.team_id, new_team_id]))))
        if agent.team_id is not None:
            self._db.add(AgentSnapshot(agent_id=agent.id, team_id=agent.team_id, name=agent.name, icon=agent.icon,
                                       last_update=agent.last_update))

    async def find_latest_snapshots_visible_to_user(self, agent_ids: list[int], user: User) -> dict[int, AgentSnapshot]:
        team_ids = get_visible_team_ids(user)
        ret = await self._db.exec(select(AgentSnapshot)
            .where(col(AgentSnapshot.agent_id).in_(agent_ids), col(AgentSnapshot.team_id).in_(team_ids))
            .order_by(col(AgentSnapshot.last_update), col(AgentSnapshot.team_id))
            .options(selectinload(attr(AgentSnapshot.team))))
        return {snapshot.agent_id: snapshot for snapshot in ret.all()}
