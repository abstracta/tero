from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional, cast

from sqlalchemy.orm import selectinload, defer
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import select, func, or_, and_, delete, col, update, not_
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import SelectOfScalar

from ..core.env import env
from ..core.repos import attr, scalar
from ..files.domain import File
from ..teams.domain import GLOBAL_TEAM_ID, TeamRoleStatus
from ..threads.domain import Thread, ThreadMessage
from ..usage.domain import Usage
from ..users.domain import User
from .domain import AgentListItem, Agent, UserAgent, AgentToolConfig, AgentToolConfigFile
from .snapshots.domain import AgentSnapshot, to_agent_list_item
from .snapshots.repos import AgentSnapshotRepository, get_visible_team_ids


class AgentRepository:

    def __init__(self, db: AsyncSession):
        self._db = db
    
    async def find_user_agents(self, text: Optional[str], user: User, limit: Optional[int] = None, offset: int = 0) -> List[AgentListItem]:
        search_condition: ColumnElement[bool] | bool = True
        if text:
            # Keep filtering and pagination in SQL before loading historical snapshots.
            team_ids = get_visible_team_ids(user)
            visible = func.coalesce(or_(Agent.user_id == user.id, col(Agent.team_id).in_(team_ids)), False)
            latest_snapshot_name = (select(col(AgentSnapshot.name))
                .where(col(AgentSnapshot.agent_id) == Agent.id, col(AgentSnapshot.team_id).in_(team_ids))
                .order_by(col(AgentSnapshot.last_update).desc(), col(AgentSnapshot.team_id).desc())
                .limit(1)
                .correlate(Agent)
                .scalar_subquery())
            escaped_text = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped_text}%"
            search_condition = or_(
                and_(visible, or_(col(Agent.name).ilike(pattern, escape="\\"), col(Agent.description).ilike(pattern, escape="\\"))),
                and_(not_(visible), latest_snapshot_name.ilike(pattern, escape="\\")))
        stmt = (
            select(Agent)
            .join(UserAgent, and_(UserAgent.agent_id == Agent.id))
            .outerjoin(Thread, and_(Thread.agent_id == Agent.id, Thread.user_id == user.id, Thread.is_test_case == False))
            .outerjoin(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .where(UserAgent.user_id == user.id, search_condition)
            .group_by(col(Agent.id), col(UserAgent.creation))
            # include agent id in order to avoid when paging getting incorrect results, since postgres does not guarantee order of rows when using limit and offset
            .order_by(func.coalesce(func.max(ThreadMessage.timestamp), UserAgent.creation).desc(), col(Agent.id).desc())
            .limit(limit)
            .offset(offset)
            # selectinload is required since sqlmodel has no way yet to lazy load async relationships (https://github.com/fastapi/sqlmodel/issues/74)
            .options(
                # avoid unnecessary loading of system_prompt that might be heavy
                defer(attr(Agent.system_prompt)),
                selectinload(attr(Agent.user)), 
                selectinload(attr(Agent.team))))
        ret = await self._db.exec(stmt)
        return list((await self.build_agent_list_items_for_user(ret.all(), user, 0)).values())

    async def build_agent_list_items_for_user(
            self, agents: Iterable[Agent], user: User, active_users: Optional[int] = None,
    ) -> dict[int, AgentListItem]:
        agents_by_id = {agent.id: agent for agent in agents}
        visible_by_agent = {agent_id: agent.is_visible_by(user) for agent_id, agent in agents_by_id.items()}
        hidden_ids = [agent_id for agent_id, visible in visible_by_agent.items() if not visible]
        snapshots = (
            await AgentSnapshotRepository(self._db).find_latest_snapshots_visible_to_user(hidden_ids, user)
            if hidden_ids else {}
        )
        return {
            agent_id: AgentListItem.from_agent(agent, agent.is_editable_by(user), active_users)
            if visible_by_agent[agent_id]
            else to_agent_list_item(snapshots[agent_id])
            if agent_id in snapshots
            else AgentListItem(id=agent_id, can_edit=False, user=None, read_only=True)
            for agent_id, agent in agents_by_id.items()
        }

    async def find_by_id(self, agent_id: int) -> Optional[Agent]:
        stmt = (
            select(Agent)
            .where(Agent.id == agent_id)
            .options(selectinload(attr(Agent.user)), selectinload(attr(Agent.team)), selectinload(attr(Agent.model)))
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_user_agent(self, agent_id: int, user_id: int) -> Optional[Agent]:
        stmt = (
            select(Agent)
            .join(UserAgent, and_(UserAgent.agent_id == Agent.id))
            .where(
                and_(Agent.id == agent_id,
                UserAgent.user_id == user_id)
            )
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_by_text(self, text: str, user: User, limit: int, offset: int) -> List[AgentListItem]:
        return await self._find_shared_agents(
            or_(col(Agent.name).ilike(f"%{text}%"),
                col(Agent.description).ilike(f"%{text}%"),
                col(User.name).ilike(f"%{text}%")),
            func.count(col(Usage.user_id).distinct()).desc(),
            user,
            limit,
            offset)

    async def find_default_agent(self) -> Optional[Agent]:
        stmt = (
            select(Agent)
            .where(and_(Agent.name == env.default_agent_name, col(Agent.user_id).is_(None), Agent.team_id  == GLOBAL_TEAM_ID))
            .options(selectinload(attr(Agent.user)), selectinload(attr(Agent.team)), selectinload(attr(Agent.model)))
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()


    async def find_own_agents(self, user: User, limit: int, offset: int) -> List[AgentListItem]:
        stmt = (
            select(Agent, func.count(col(Usage.user_id).distinct()))
            .outerjoin(Usage, and_(Usage.agent_id == Agent.id,
                                   Usage.timestamp >= datetime.now(timezone.utc) - timedelta(days=30)))
            .outerjoin(UserAgent, and_(Agent.id == UserAgent.agent_id, UserAgent.user_id == user.id))
            .where(Agent.user_id == user.id, or_(Agent.team_id != None, col(UserAgent.agent_id).is_not(None)))
            .group_by(col(Agent.id))
            .order_by(col(Agent.last_update).desc(), col(Agent.id).desc())
            .limit(limit)
            .offset(offset)
            .options(selectinload(attr(Agent.user)), selectinload(attr(Agent.team)))
        )
        ret = await self._db.exec(stmt)
        return [AgentListItem.from_agent(a, a.is_editable_by(user), c) for a, c in ret.all()]

    async def _find_shared_agents(self, condition: ColumnElement[bool] | bool, order_by: Any, user: User, limit: int, offset: int, team_id: Optional[int] = None) -> List[
        AgentListItem]:
        stmt = (
            select(Agent, func.count(col(Usage.user_id).distinct()))
            .outerjoin(User, and_(Agent.user_id == User.id))
            .outerjoin(Usage, and_(Usage.agent_id == Agent.id,
                                   Usage.timestamp >= datetime.now(timezone.utc) - timedelta(days=30),
                                   col(Usage.user_id).in_(select(col(User.id)).where(not_(User.exclude_from_metrics)).correlate(None))))
            .where(and_(and_(Agent.team_id == team_id) if team_id else or_(Agent.team_id == GLOBAL_TEAM_ID, col(Agent.team_id).in_([tr.team_id for tr in user.team_roles if tr.status == TeamRoleStatus.ACCEPTED])), condition))
            .group_by(col(Agent.id))
            # include agent id in order to avoid when paging getting incorrect results, since postgres does not guarantee order of rows when using limit and offset
            .order_by(order_by, col(Agent.id).desc())
            .limit(limit)
            .offset(offset)
            .options(selectinload(attr(Agent.user)), selectinload(attr(Agent.team))))
        ret = await self._db.exec(stmt)
        return [AgentListItem.from_agent(a, a.is_editable_by(user), c) for a, c in ret.all()]

    async def find_newest(self, user: User, limit: int, offset: int, team_id: Optional[int] = None) -> List[AgentListItem]:
        return await self._find_shared_agents(
            True,
            col(Agent.last_update).desc(),
            user,
            limit,
            offset,
            team_id)

    async def find_top_used(self, user: User, limit: int, offset: int, team_id: Optional[int] = None) -> List[AgentListItem]:
        return await self._find_shared_agents(
            True,
            func.count(col(Usage.user_id).distinct()).desc(),
            user,
            limit,
            offset,
            team_id)

    async def add_user_agent(self, agent_id: int, user_id: int):
        await self._db.merge(UserAgent(user_id=user_id, agent_id=agent_id))
        await self._db.commit()

    async def remove_user_agent(self, agent_id: int, user_id: int):
        stmt = scalar(
            delete(UserAgent)
            .where(and_(UserAgent.user_id == user_id, UserAgent.agent_id == agent_id))
        )
        await self._db.exec(stmt)
        await self._db.commit()

    async def add(self, agent: Agent) -> Agent:
        self._db.add(agent)
        await self._db.flush()
        await self._db.refresh(agent, ['id', 'user'])
        if not agent.name:
            agent.set_default_name()
        self._db.add(agent)
        self._db.add(UserAgent(user_id=cast(int, agent.user_id), agent_id=agent.id))
        await self._db.commit()
        return agent

    async def update(self, agent: Agent):
        self._db.add(agent)
        await self._db.commit()
        await self._db.refresh(agent)
        return agent
    
    async def remove_team_agents(self, team_id: int):
        stmt = (
            update(Agent)
            .where(col(Agent.team_id) == team_id)
            .values(team_id=None)
        )
        await self._db.exec(scalar(stmt))
        await self._db.commit()


class AgentToolConfigRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_agent_id(self, agent_id: int) -> List[AgentToolConfig]:
        stmt = (
            select(AgentToolConfig)
            .join(Agent, and_(AgentToolConfig.agent_id == Agent.id))
            .where(and_(AgentToolConfig.agent_id == agent_id, AgentToolConfig.draft == False)))
        ret = await self._db.exec(stmt)
        return list(ret.all())

    async def add(self, agent_tool_config: AgentToolConfig):
        await self._db.merge(agent_tool_config)
        await self._db.commit()

    async def delete(self, agent_id: int, tool_id: str):
        stmt = (
            delete(AgentToolConfig)
            .where(and_(AgentToolConfig.agent_id == agent_id, AgentToolConfig.tool_id == tool_id))
        )
        await self._db.exec(scalar(stmt))
        await self._db.commit()

    async def delete_drafts(self, agent_id: int):
        stmt = (
            delete(AgentToolConfig)
            .where(and_(AgentToolConfig.agent_id == agent_id, AgentToolConfig.draft == True))
        )
        await self._db.exec(scalar(stmt))
        await self._db.commit()

    async def find_by_ids(self, agent_id: int, tool_id, include_drafts: bool = False) -> Optional[AgentToolConfig]:
        stmt = (
            select(AgentToolConfig)
            .join(Agent, and_(AgentToolConfig.agent_id == Agent.id))
            .where(
                and_(AgentToolConfig.agent_id == agent_id,
                AgentToolConfig.tool_id == tool_id,
                or_(AgentToolConfig.draft == False, include_drafts == True)))
            .options(selectinload(attr(AgentToolConfig.agent))))
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def add_many(self, configs: List[AgentToolConfig]) -> None:
        self._db.add_all(configs)
        await self._db.commit()


class AgentToolConfigFileRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def add(self, agent_tool_config_file: AgentToolConfigFile):
        self._db.add(agent_tool_config_file)
        await self._db.commit()

    async def find_by_agent_id_and_tool_id(self, agent_id: int, tool_id: str) -> List[File]:
        stmt = (
            select(File)
            .join(AgentToolConfigFile, and_(AgentToolConfigFile.file_id == File.id))
            .where(and_(AgentToolConfigFile.agent_id == agent_id, AgentToolConfigFile.tool_id == tool_id))
            .order_by(col(File.id).asc())
            .options(defer(attr(File.content))))
        ret = await self._db.exec(stmt)
        return list(ret.all())

    async def find_by_ids(self, agent_id: int, tool_id: str, file_id: int) -> Optional[File]:
        stmt = (self._select_by_ids(agent_id, tool_id, file_id)
                .options(defer(attr(File.content))))
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    @staticmethod
    def _select_by_ids(agent_id: int, tool_id: str, file_id: int) -> SelectOfScalar[File]:
        return (select(File)
            .join(AgentToolConfigFile, and_(AgentToolConfigFile.file_id == File.id))
            .where(and_(
                AgentToolConfigFile.agent_id == agent_id,
                AgentToolConfigFile.tool_id == tool_id,
                   File.id == file_id)))

    async def find_with_content_by_ids(self, agent_id: int, tool_id: str, file_id: int) -> Optional[File]:
        stmt = self._select_by_ids(agent_id, tool_id, file_id)
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def delete(self, agent_id: int, tool_id: str, file_id: int):
        stmt = (
            delete(AgentToolConfigFile)
            .where(and_(AgentToolConfigFile.agent_id == agent_id, AgentToolConfigFile.tool_id == tool_id,
                   AgentToolConfigFile.file_id == file_id))
        )
        await self._db.exec(scalar(stmt))
        await self._db.commit()

    async def delete_by_agent_id_and_tool_id(self, agent_id: int, tool_id: str):
        stmt =(
            delete(AgentToolConfigFile)
            .where(and_(AgentToolConfigFile.agent_id == agent_id, AgentToolConfigFile.tool_id == tool_id))
        )
        await self._db.exec(scalar(stmt))
        await self._db.commit()

    async def find_with_content_by_agent_and_tool(self, agent_id: int, tool_id: str) -> List[File]:
        stmt = (
            select(File)
            .join(AgentToolConfigFile, and_(AgentToolConfigFile.file_id == File.id))
            .where(
                and_(
                    AgentToolConfigFile.agent_id == agent_id,
                    AgentToolConfigFile.tool_id == tool_id
                )
            )
            .order_by(col(File.id).asc())
        )
        ret = await self._db.exec(stmt)
        return list(ret.all())
