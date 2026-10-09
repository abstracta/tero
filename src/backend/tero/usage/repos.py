from datetime import datetime, timedelta, timezone
from typing import List, Optional, Any
from sqlmodel import not_, select, func, desc, distinct, and_, col, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy import union_all, literal_column
import sqlalchemy as sa

from .domain import PRIVATE_AGENT_ID, AgentUsageItem, UserUsageItem, Usage, ImpactSummary, AgentImpactItem, UserImpactItem, MessageUsage, UsageSummary
from ..users.domain import User
from ..threads.domain import ThreadMessage, Thread
from ..agents.domain import Agent
from ..core.repos import attr
from ..teams.domain import TeamRole, Team, TeamRoleStatus
from ..teams.domain import GLOBAL_TEAM_ID, MY_TEAM_ID
from ..external_agents.domain import ExternalAgentTimeSaving, ExternalAgent


class UsageRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_current_month_user_usage_usd(self, user_id: int) -> float:
        stmt = (
            select(func.sum(Usage.usd_cost))
            .where(Usage.user_id == user_id,
                   Usage.timestamp >= datetime.now(timezone.utc)
                   .replace(day=1, hour=0, minute=0, second=0, microsecond=0)))
        ret = await self._db.exec(stmt)
        return ret.one() or 0.0

    async def add(self, usage: Usage | MessageUsage | None):
        if not usage or usage.usd_cost == 0:
            return

        if isinstance(usage, MessageUsage):
            for usage in usage.usages():
                self._db.add(usage)
        else:
            self._db.add(usage)
        await self._db.commit()

    async def _get_human_hours(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int) -> float:
        human_hours_query = select(func.sum(User.monthly_hours)).where(and_(User.created_at <= to_date, or_(col(User.deleted_at).is_(None), col(User.deleted_at) >= from_date), 
                                                                            self._get_user_filter_condition(user_id, team_id)))
        human_hours_result = await self._db.exec(human_hours_query)
        return human_hours_result.one() or 0

    def _get_user_filter_condition(self, user_id: int, team_id: int, only_accepted: bool = True, filtered_user_id: Optional[int] = None) -> ColumnElement[bool] | bool:
        if team_id == MY_TEAM_ID:
            return User.id == user_id
        elif filtered_user_id is not None:
            return and_(
                User.id == filtered_user_id,
                self._get_team_filter_condition(team_id, only_accepted),
                not_(User.exclude_from_metrics)
            )
        else:
            return and_(
                self._get_team_filter_condition(team_id, only_accepted),
                not_(User.exclude_from_metrics)
            )

    def _get_team_filter_condition(self, team_id: int, only_accepted: bool = True) -> ColumnElement[bool] | bool:
        return col(User.team_roles).any(and_(
            TeamRole.team_id == team_id,
            (TeamRole.status == TeamRoleStatus.ACCEPTED if only_accepted else True))) if team_id != GLOBAL_TEAM_ID else True

    def _get_agent_filter_condition(self, team_id: int) -> ColumnElement[bool] | bool:
        if team_id == MY_TEAM_ID:
            return True
        else:
            return col(Agent.team_id).in_([team_id, GLOBAL_TEAM_ID])

    def _get_private_agents_filter(self, team_id: int) -> ColumnElement[bool]:
        return or_(
            col(Agent.team_id) == None, 
            and_(
                col(Agent.team_id) != GLOBAL_TEAM_ID,
                col(Agent.team_id) != team_id
            )
        )

    async def _get_ai_hours(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int) -> float:
        ai_minutes_query = (
            select(func.sum(ThreadMessage.minutes_saved))
            .join(Thread, and_(ThreadMessage.thread_id == Thread.id))
            .join(User, and_(Thread.user_id == User.id))
            .where(self._get_thread_message_filter_condition(from_date, to_date, team_id, user_id))
        )
        ai_minutes_result = await self._db.exec(ai_minutes_query)
        ai_minutes = ai_minutes_result.one() or 0

        external_ai_minutes_query = (
            select(func.sum(ExternalAgentTimeSaving.minutes_saved))
            .join(User, and_(ExternalAgentTimeSaving.user_id == User.id))
            .where(self._get_external_agent_time_saving_filter_condition(from_date, to_date, team_id, user_id))
        )
        external_ai_minutes_result = await self._db.exec(external_ai_minutes_query)
        external_ai_minutes = external_ai_minutes_result.one() or 0

        total_minutes = ai_minutes + external_ai_minutes
        return total_minutes / 60

    def _get_thread_message_filter_condition(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int, filtered_user_id: Optional[int] = None):
        return and_(
            not_(Thread.is_test_case),
            self._get_date_range_filter(ThreadMessage.timestamp, from_date, to_date),
            self._get_user_filter_condition(user_id, team_id, filtered_user_id=filtered_user_id),
        )

    def _get_external_agent_time_saving_filter_condition(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int):
        return and_(
            self._get_date_range_filter(ExternalAgentTimeSaving.date, from_date, to_date),
            self._get_user_filter_condition(user_id, team_id)
        )

    async def get_impact_summary(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int) -> ImpactSummary:
        from_previous_period = self._get_previous_period(from_date, to_date)

        human_hours = await self._get_human_hours(from_date, to_date, team_id, user_id)
        previous_human_hours = await self._get_human_hours(from_previous_period, from_date, team_id, user_id)

        ai_hours = await self._get_ai_hours(from_date, to_date, team_id, user_id)
        previous_ai_hours = await self._get_ai_hours(from_previous_period, from_date, team_id, user_id)

        return ImpactSummary(
            human_hours=int(human_hours),
            ai_hours=int(ai_hours),
            previous_human_hours=int(previous_human_hours),
            previous_ai_hours=int(previous_ai_hours),
        )

    async def get_impact_top_agents(self, from_date: datetime, to_date: datetime, team_id: int, search: Optional[str], limit: Optional[int], offset: Optional[int], user_id: int, filtered_user_id: Optional[int]) -> List[AgentImpactItem]:
        from_previous_month = self._get_previous_period(from_date, to_date)

        thread_messages_subquery = self._get_minutes_saved_by_agent_subquery(from_date, to_date, team_id, user_id, filtered_user_id)
        previous_thread_messages_subquery = self._get_minutes_saved_by_agent_subquery(from_previous_month, from_date, team_id, user_id, filtered_user_id)

        active_users_subquery = self._get_active_users_by_agent_subquery(from_date, to_date, team_id, user_id, filtered_user_id)
        previous_active_users_subquery = self._get_active_users_by_agent_subquery(from_previous_month, from_date, team_id, user_id, filtered_user_id)

        shared_agents_query = (
            select(  # type: ignore
                col(Agent.id).label("agent_id"),
                col(Agent.name).label("agent_name"),
                col(Agent.icon).label("icon"),
                col(Team.id).label("team_id"),
                col(Team.name).label("team_name"),
                col(User.name).label("author_name"),
                literal_column("False").label("is_external_agent"),
                func.coalesce(func.max(active_users_subquery.c.active_users), 0).label("active_users"),
                func.coalesce(func.max(thread_messages_subquery.c.minutes_saved), 0).label("minutes_saved"),
                func.coalesce(func.max(previous_active_users_subquery.c.active_users), 0).label("previous_active_users"),
                func.coalesce(func.max(previous_thread_messages_subquery.c.minutes_saved), 0).label("previous_minutes_saved"),
            )
            .outerjoin(User, Agent.user_id == User.id)
            .outerjoin(Team, Agent.team_id == Team.id)
            .outerjoin(active_users_subquery, active_users_subquery.c.agent_id == Agent.id)
            .outerjoin(previous_active_users_subquery, previous_active_users_subquery.c.agent_id == Agent.id)
            .outerjoin(thread_messages_subquery, thread_messages_subquery.c.agent_id == Agent.id)
            .outerjoin(previous_thread_messages_subquery, previous_thread_messages_subquery.c.agent_id == Agent.id)
            .where(self._get_agent_filter_condition(team_id))
            .group_by(
                col(Agent.id),
                col(Agent.name),
                col(Agent.icon),
                col(Agent.team_id),
                col(Team.id),
                col(Team.name),
                col(User.name),
            )
            .options(selectinload(attr(Agent.user)))
        )

        external_agents_query = self._get_external_agents_subquery(from_date, to_date, from_previous_month, team_id, filtered_user_id or (user_id if team_id == MY_TEAM_ID else None))

        if team_id == MY_TEAM_ID:
            combined_query = union_all(shared_agents_query, external_agents_query).subquery()
        else:
            combined_query = union_all(self._get_private_agents_impact_subquery(from_date, to_date, from_previous_month, team_id, user_id, filtered_user_id), shared_agents_query, external_agents_query).subquery()

        query = (
            select(  # type: ignore
                combined_query.c.agent_id,
                combined_query.c.agent_name,
                combined_query.c.icon,
                combined_query.c.team_id,
                combined_query.c.team_name,
                combined_query.c.author_name,
                combined_query.c.active_users,
                combined_query.c.minutes_saved,
                combined_query.c.previous_active_users,
                combined_query.c.previous_minutes_saved,
                combined_query.c.is_external_agent
            )
            .where(
                and_(
                    or_(
                        combined_query.c.minutes_saved > 0,
                        combined_query.c.previous_minutes_saved > 0
                    ),
                    col(combined_query.c.agent_name).ilike(f"%{search}%") if search else True
                )
            )
            .order_by(desc(combined_query.c.minutes_saved), desc(combined_query.c.active_users))
            .limit(limit)
            .offset(offset)
        )

        result = await self._db.exec(query)

        return [
            AgentImpactItem(
                agent_id=row.agent_id,
                agent_name=row.agent_name,
                active_users=row.active_users,
                minutes_saved=row.minutes_saved,
                previous_active_users=row.previous_active_users,
                previous_minutes_saved=row.previous_minutes_saved,
                icon_bytes=row.icon,
                team=Team(id=row.team_id, name=row.team_name) if row.team_id else None,
                author_name=row.author_name,
                is_external_agent=row.is_external_agent
            )
            for row in result
        ]

    def _get_private_agents_impact_subquery(self, from_date: datetime, to_date: datetime, from_previous_month: datetime, team_id: int, user_id: int, filtered_user_id: Optional[int] = None):
        agent_filter = self._get_private_agents_filter(team_id)

        usage_subquery = self._get_active_users_subquery(from_date, to_date, team_id, user_id, agent_filter, filtered_user_id)
        previous_usage_subquery = self._get_active_users_subquery(from_previous_month, from_date, team_id, user_id, agent_filter, filtered_user_id)

        minutes_saved_subquery = self._get_minutes_saved_subquery(from_date, to_date, team_id, user_id, agent_filter, filtered_user_id)
        previous_minutes_saved_subquery = self._get_minutes_saved_subquery(from_previous_month, from_date, team_id, user_id, agent_filter, filtered_user_id)

        return select(  # type: ignore
            literal_column("-1").label("agent_id"),
            literal_column("NULL").label("agent_name"),
            literal_column("NULL").label("icon"),
            literal_column("NULL").label("team_id"),
            literal_column("NULL").label("team_name"),
            literal_column("NULL").label("author_name"),
            literal_column("False").label("is_external_agent"),
            func.coalesce(usage_subquery.c.active_users, 0).label("active_users"),
            func.coalesce(minutes_saved_subquery.c.minutes_saved, 0).label("minutes_saved"),
            func.coalesce(previous_usage_subquery.c.active_users, 0).label("previous_active_users"),
            func.coalesce(previous_minutes_saved_subquery.c.minutes_saved, 0).label("previous_minutes_saved")
        )

    def _get_external_agents_subquery(self, from_date: datetime, to_date: datetime, from_previous_month: datetime, team_id: int, user_id: Optional[int]):
        external_agents_current_subquery = self._build_external_agent_subquery(from_date, to_date, team_id, user_id)

        external_agents_previous_subquery = self._build_external_agent_subquery(from_previous_month, from_date, team_id, user_id, True)

        external_agents_query = (
            select(  # type: ignore
                col(ExternalAgent.id).label("agent_id"),
                col(ExternalAgent.name).label("agent_name"),
                col(ExternalAgent.icon).label("icon"),
                literal_column(str(GLOBAL_TEAM_ID)).label("team_id"),
                (select(Team.name).where(Team.id == GLOBAL_TEAM_ID).scalar_subquery()).label("team_name"),
                literal_column("NULL").label("author_name"),
                literal_column("True").label("is_external_agent"),
                func.coalesce(external_agents_current_subquery.c.active_users, 0).label("active_users"),
                func.coalesce(external_agents_current_subquery.c.minutes_saved, 0).label("minutes_saved"),
                func.coalesce(external_agents_previous_subquery.c.previous_active_users, 0).label("previous_active_users"),
                func.coalesce(external_agents_previous_subquery.c.previous_minutes_saved, 0).label("previous_minutes_saved")
            )
            .outerjoin(external_agents_current_subquery, external_agents_current_subquery.c.external_agent_id == ExternalAgent.id)
            .outerjoin(external_agents_previous_subquery, external_agents_previous_subquery.c.external_agent_id == ExternalAgent.id)
            .where(
                or_(
                    external_agents_current_subquery.c.active_users > 0,
                    external_agents_previous_subquery.c.previous_active_users > 0
                )
            )
        )
        return external_agents_query

    def _build_external_agent_subquery(self, from_date: datetime, to_date: datetime, team_id: int, user_id: Optional[int], previous: bool = False):
        if user_id is not None:
            external_agent_filter_condition = col(ExternalAgentTimeSaving.user_id) == user_id
        else:
            external_agent_filter_condition = col(ExternalAgentTimeSaving.user_id).in_(
                select(User.id)
                .where(self._get_team_filter_condition(team_id))
            )
        return (
            select(
                ExternalAgentTimeSaving.external_agent_id,
                func.count(distinct(ExternalAgentTimeSaving.user_id)).label(f"{'previous_' if previous else ''}active_users"),
                (func.sum(ExternalAgentTimeSaving.minutes_saved)).label(f"{'previous_' if previous else ''}minutes_saved")
            )
            .join(User, and_(User.id == ExternalAgentTimeSaving.user_id))
            .where(
                and_(
                    self._get_date_range_filter(ExternalAgentTimeSaving.date, from_date, to_date),
                    external_agent_filter_condition,
                    not_(User.exclude_from_metrics) if team_id != MY_TEAM_ID else True,
                )
            )
            .group_by(col(ExternalAgentTimeSaving.external_agent_id))
            .subquery()
        )

    async def get_impact_top_users(self, from_date: datetime, to_date: datetime, team_id: int, search: Optional[str], limit: Optional[int], offset: Optional[int], user_id: int, agent_id: Optional[int], is_external_agent: Optional[bool]=None) -> List[UserImpactItem]:
        from_previous_period = self._get_previous_period(from_date, to_date)
        agent_filter = (Thread.agent_id == agent_id) if agent_id is not None and agent_id != PRIVATE_AGENT_ID else True
        external_agent_filter = (ExternalAgentTimeSaving.external_agent_id == agent_id) if is_external_agent and agent_id is not None else agent_id is None

        previous_thread_messages_subquery = (
            select(
                Thread.user_id,
                func.sum(ThreadMessage.minutes_saved).label("previous_minutes_saved")
            )
            .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .outerjoin(Agent, and_(Thread.agent_id == Agent.id))
            .where(
                and_(
                    not_(Thread.is_test_case),
                    self._get_date_range_filter(ThreadMessage.timestamp, from_previous_period, from_date),
                    agent_filter
                )
            )
            .group_by(col(Thread.user_id))
            .subquery()
        )

        external_agent_entries_subquery = self._get_external_agent_minutes(from_date, to_date, external_agent_filter)

        previous_external_agent_entries_subquery = self._get_external_agent_minutes(from_previous_period, from_date, external_agent_filter)

        query = (
            select(  # type: ignore
                col(User.id),
                col(User.name),
                (func.sum(ThreadMessage.minutes_saved) if not is_external_agent else sa.literal(0)).label("minutes_saved"),
                col(User.monthly_hours),
                func.coalesce(previous_thread_messages_subquery.c.previous_minutes_saved, 0).label("previous_minutes_saved"),
                func.coalesce(external_agent_entries_subquery.c.external_minutes_saved, 0).label("external_minutes_saved"),
                func.coalesce(previous_external_agent_entries_subquery.c.external_minutes_saved, 0).label("previous_external_minutes_saved"),
            )
            .outerjoin(Thread, and_(Thread.user_id == User.id, not_(Thread.is_test_case)))
            .outerjoin(ThreadMessage, and_(
                ThreadMessage.thread_id == Thread.id,
                self._get_date_range_filter(ThreadMessage.timestamp, from_date, to_date),
                agent_filter
            ))
            .outerjoin(Agent, and_(Thread.agent_id == Agent.id))
            .outerjoin(previous_thread_messages_subquery, previous_thread_messages_subquery.c.user_id == User.id)
            .outerjoin(external_agent_entries_subquery, external_agent_entries_subquery.c.user_id == User.id)
            .outerjoin(previous_external_agent_entries_subquery, previous_external_agent_entries_subquery.c.user_id == User.id)
            .outerjoin(TeamRole, and_(TeamRole.user_id == User.id, TeamRole.team_id == team_id))
            .where(
                and_(
                    col(User.name) != None,
                    or_(col(User.deleted_at) == None, col(User.deleted_at) >= from_date),
                    col(User.name).ilike(f"%{search}%") if search else True,
                    self._get_user_filter_condition(user_id, team_id, False),
                    and_(
                        self._get_private_agents_filter(team_id),
                        self._get_team_filter_condition(team_id)
                    ) if agent_id == PRIVATE_AGENT_ID else True,
                    self._get_agent_user_filter_from_usage(from_date, to_date, team_id, agent_id, is_external_agent)
                )
            )
            .group_by(
                col(User.id), 
                previous_thread_messages_subquery.c.previous_minutes_saved, 
                external_agent_entries_subquery.c.external_minutes_saved,
                previous_external_agent_entries_subquery.c.external_minutes_saved,
            )
            .order_by(desc((func.coalesce(func.sum(ThreadMessage.minutes_saved), 0) + func.coalesce(external_agent_entries_subquery.c.external_minutes_saved, 0)) / User.monthly_hours), col(User.id))
            .limit(limit)
            .offset(offset)
        )

        result = await self._db.exec(query)

        return [UserImpactItem(
            user_id=user_id,
            user_name=user_name,
            minutes_saved=(minutes_saved or 0) + (external_minutes_saved or 0),
            monthly_hours=monthly_hours or 0,
            previous_minutes_saved=(previous_minutes_saved or 0) + (previous_external_minutes_saved or 0),
        ) for user_id, user_name, minutes_saved, monthly_hours, previous_minutes_saved, external_minutes_saved, previous_external_minutes_saved in result]

    def _get_agent_user_filter_from_usage(self, from_date: datetime, to_date: datetime, team_id: int, agent_id: Optional[int], is_external_agent: Optional[bool]):
        subquery = None

        if is_external_agent and agent_id:
            subquery = (
                select(ExternalAgentTimeSaving.user_id)
                .where(
                    and_(
                        self._get_date_range_filter(ExternalAgentTimeSaving.date, from_date, to_date),
                        ExternalAgentTimeSaving.external_agent_id == agent_id,
                    )
                )
            )
        elif agent_id == PRIVATE_AGENT_ID:
            subquery = (
                select(distinct(Thread.user_id))
                .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
                .outerjoin(Agent, and_(Agent.id == Thread.agent_id))
                .where(
                    and_(
                        not_(Thread.is_test_case),
                        self._get_date_range_filter(ThreadMessage.timestamp, from_date, to_date),
                        self._get_private_agents_filter(team_id),
                    )
                )
            )
        elif agent_id:
            subquery = (
                select(distinct(Thread.user_id))
                .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
                .where(
                    and_(
                        not_(Thread.is_test_case),
                        self._get_date_range_filter(ThreadMessage.timestamp, from_date, to_date),
                        Thread.agent_id == agent_id
                    )
                )
            )

        return col(User.id).in_(subquery) if subquery is not None else True

    async def get_usage_summary(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int) -> UsageSummary:
        from_previous_period = self._get_previous_period(from_date, to_date)

        active_users_query = self._get_active_users_subquery(from_date, to_date, team_id, user_id)
        total_threads_query = self._get_total_threads_subquery(from_date, to_date, team_id, user_id)

        previous_active_users_query = self._get_active_users_subquery(from_previous_period, from_date, team_id, user_id)
        previous_total_threads_query = self._get_total_threads_subquery(from_previous_period, from_date, team_id, user_id)

        query = select(
            func.coalesce(active_users_query.c.active_users, 0).label("active_users"),
            func.coalesce(total_threads_query.c.total_threads, 0).label("total_threads"),
            func.coalesce(previous_active_users_query.c.active_users, 0).label("previous_active_users"),
            func.coalesce(previous_total_threads_query.c.total_threads, 0).label("previous_total_threads")
        )

        active_users, total_threads, previous_active_users, previous_total_threads = (await self._db.exec(query)).one()

        return UsageSummary(
            active_users=active_users,
            total_threads=total_threads,
            previous_active_users=previous_active_users,
            previous_total_threads=previous_total_threads
        )

    async def get_usage_top_agents(self, from_date: datetime, to_date: datetime, team_id: int, search: Optional[str], limit: Optional[int], offset: Optional[int], user_id: int, filtered_user_id: Optional[int]) -> List[AgentUsageItem]:
        from_previous_period = self._get_previous_period(from_date, to_date)

        total_threads_query = self._get_threads_by_agent_subquery(from_date, to_date, team_id, user_id, filtered_user_id)
        previous_total_threads_query = self._get_threads_by_agent_subquery(from_previous_period, from_date, team_id, user_id, filtered_user_id)

        active_users_query = self._get_active_users_by_agent_subquery(from_date, to_date, team_id, user_id, filtered_user_id)
        previous_active_users_query = self._get_active_users_by_agent_subquery(from_previous_period, from_date, team_id, user_id, filtered_user_id)

        shared_agents_query = (
            select(  # type: ignore
                col(Agent.id).label("agent_id"),
                col(Agent.name).label("agent_name"),
                col(Agent.icon).label("icon"),
                col(Team.id).label("team_id"),
                col(Team.name).label("team_name"),
                col(User.name).label("author_name"),
                func.coalesce(func.max(active_users_query.c.active_users), 0).label("active_users"),
                func.coalesce(func.max(total_threads_query.c.total_threads), 0).label("total_threads"),
                func.coalesce(func.max(previous_active_users_query.c.active_users), 0).label("previous_active_users"),
                func.coalesce(func.max(previous_total_threads_query.c.total_threads), 0).label("previous_total_threads")
            )
            .outerjoin(User, Agent.user_id == User.id)
            .outerjoin(Team, Agent.team_id == Team.id)
            .outerjoin(active_users_query, active_users_query.c.agent_id == Agent.id)
            .outerjoin(previous_active_users_query, previous_active_users_query.c.agent_id == Agent.id)
            .outerjoin(total_threads_query, total_threads_query.c.agent_id == Agent.id)
            .outerjoin(previous_total_threads_query, previous_total_threads_query.c.agent_id == Agent.id)
            .where(self._get_agent_filter_condition(team_id))
            .group_by(
                col(Agent.id),
                col(Agent.name),
                col(Agent.icon),
                col(Agent.team_id),
                col(Team.id),
                col(Team.name),
                col(User.name),
            )
            .options(selectinload(attr(Agent.user)))
        )

        if team_id == MY_TEAM_ID:
            combined_query = shared_agents_query.subquery()
        else:
            combined_query = union_all(self._get_private_agents_usage_subquery(from_date, to_date, from_previous_period, team_id, user_id, filtered_user_id), shared_agents_query).subquery()

        query = (
            select(  # type: ignore
                combined_query.c.agent_id,
                combined_query.c.agent_name,
                combined_query.c.icon,
                combined_query.c.team_id,
                combined_query.c.team_name,
                combined_query.c.author_name,
                combined_query.c.active_users,
                combined_query.c.total_threads,
                combined_query.c.previous_active_users,
                combined_query.c.previous_total_threads
            )
            .where(
                and_(
                    or_(
                        combined_query.c.total_threads > 0,
                        combined_query.c.previous_total_threads > 0
                    ),
                    col(combined_query.c.agent_name).ilike(f"%{search}%") if search else True
                )
            )
            .order_by(desc(combined_query.c.total_threads), desc(combined_query.c.active_users))
            .limit(limit)
            .offset(offset)
        )

        result = await self._db.exec(query)

        return [
            AgentUsageItem(
                agent_id=row.agent_id,
                agent_name=row.agent_name,
                active_users=row.active_users,
                total_threads=row.total_threads,
                previous_active_users=row.previous_active_users,
                previous_total_threads=row.previous_total_threads,
                icon_bytes=row.icon,
                team=Team(id=row.team_id, name=row.team_name) if row.team_id else None,
                author_name=row.author_name
            )
            for row in result
        ]

    def _get_private_agents_usage_subquery(self, from_date: datetime, to_date: datetime, from_previous_period: datetime, team_id: int, user_id: int, filtered_user_id: Optional[int] = None):
        agent_filter = self._get_private_agents_filter(team_id)

        active_users_query = self._get_active_users_subquery(from_date, to_date, team_id, user_id, agent_filter, filtered_user_id)
        previous_active_users_query = self._get_active_users_subquery(from_previous_period, from_date, team_id, user_id, agent_filter, filtered_user_id)
        total_threads_query = self._get_total_threads_subquery(from_date, to_date, team_id, user_id, agent_filter, False, filtered_user_id)
        previous_total_threads_query = self._get_total_threads_subquery(from_previous_period, from_date, team_id, user_id, agent_filter, False, filtered_user_id)

        return select(  # type: ignore
            literal_column("-1").label("agent_id"),
            literal_column("NULL").label("agent_name"),
            literal_column("NULL").label("icon"),
            literal_column("NULL").label("team_id"),
            literal_column("NULL").label("team_name"),
            literal_column("NULL").label("author_name"),
            func.coalesce(active_users_query.c.active_users, 0).label("active_users"),
            func.coalesce(total_threads_query.c.total_threads, 0).label("total_threads"),
            func.coalesce(previous_active_users_query.c.active_users, 0).label("previous_active_users"),
            func.coalesce(previous_total_threads_query.c.total_threads, 0).label("previous_total_threads")
        )

    async def get_usage_top_users(self, from_date: datetime, to_date: datetime, team_id: int, search: Optional[str], limit: Optional[int], offset: Optional[int], user_id: int, agent_id: Optional[int]) -> List[UserUsageItem]:
        from_previous_period = self._get_previous_period(from_date, to_date)

        agent_filter = (Thread.agent_id == agent_id) if agent_id is not None and agent_id != PRIVATE_AGENT_ID else True

        if agent_id == PRIVATE_AGENT_ID:
            agent_filter = self._get_private_agents_filter(team_id)

        total_threads_query  = self._get_total_threads_subquery(from_date, to_date, team_id, user_id, agent_filter, True)
        previous_total_threads_query = self._get_total_threads_subquery(from_previous_period, from_date, team_id, user_id, agent_filter, True)

        query = (
            select(  # type: ignore
                col(User.id),
                col(User.name),
                func.coalesce(total_threads_query.c.total_threads, 0).label("total_threads"),
                func.coalesce(previous_total_threads_query.c.total_threads, 0).label("previous_total_threads")
            )
            .outerjoin(total_threads_query, total_threads_query.c.user_id == User.id)
            .outerjoin(previous_total_threads_query, previous_total_threads_query.c.user_id == User.id)
            .outerjoin(TeamRole, and_(TeamRole.user_id == User.id, TeamRole.team_id == team_id))
            .where(
                and_(
                    col(User.name) != None,
                    or_(col(User.deleted_at) == None, col(User.deleted_at) >= from_date),
                    col(User.name).ilike(f"%{search}%") if search else True,
                    self._get_user_filter_condition(user_id, team_id, False),
                    self._get_agent_user_filter_from_threads(from_date, to_date, team_id, agent_id) if agent_id else True,
                    or_(
                        func.coalesce(total_threads_query.c.total_threads, 0) > 0,
                        func.coalesce(previous_total_threads_query.c.total_threads, 0) > 0
                    )
                )
            )
            .group_by(
                col(User.id),
                total_threads_query.c.total_threads,
                previous_total_threads_query.c.total_threads
            )
            .order_by(desc(func.coalesce(total_threads_query.c.total_threads, 0)), col(User.id))
            .limit(limit)
            .offset(offset)
        )

        result = await self._db.exec(query)

        return [UserUsageItem(
            user_id=user_id,
            user_name=user_name,
            total_threads=total_threads or 0,
            previous_total_threads=previous_total_threads or 0
        ) for user_id, user_name, total_threads, previous_total_threads in result]

    def _get_active_users_subquery(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int, agent_filter: ColumnElement[bool] | bool = True, filtered_user_id: Optional[int] = None):
        return (
            select(func.count(distinct(Thread.user_id)).label("active_users"))
            .join(User, and_(Thread.user_id == User.id))
            .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .outerjoin(Agent, and_(Thread.agent_id == Agent.id))
            .where(
                and_(
                    self._get_thread_message_filter_condition(from_date, to_date, team_id, user_id, filtered_user_id),
                    agent_filter
                )
            )
        ).subquery()

    def _get_threads_by_agent_subquery(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int, filtered_user_id: Optional[int]):
        return (
            select(
                Thread.agent_id,
                func.count(distinct(Thread.id)).label("total_threads")
            )
            .join(User, and_(Thread.user_id == User.id))
            .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .where(self._get_thread_message_filter_condition(from_date, to_date, team_id, user_id, filtered_user_id))
            .group_by(col(Thread.agent_id))
            .subquery()
        )

    def _get_active_users_by_agent_subquery(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int, filtered_user_id: Optional[int]):
        return (
            select(
                Thread.agent_id,
                func.count(distinct(Thread.user_id)).label("active_users")
            )
            .join(User, and_(Thread.user_id == User.id))
            .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .where(self._get_thread_message_filter_condition(from_date, to_date, team_id, user_id, filtered_user_id))
            .group_by(col(Thread.agent_id))
            .subquery()
        )

    def _get_minutes_saved_by_agent_subquery(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int, filtered_user_id: Optional[int]):
        return (
            select(
                Thread.agent_id,
                func.sum(ThreadMessage.minutes_saved).label("minutes_saved")
            )
            .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .join(User, and_(Thread.user_id == User.id))
            .where(self._get_thread_message_filter_condition(from_date, to_date, team_id, user_id, filtered_user_id))
            .group_by(col(Thread.agent_id))
            .subquery()
        )

    def _get_external_agent_minutes(self, from_date: datetime, to_date: datetime, external_agent_filter: ColumnElement[bool] | bool):
        return (
            select(
                ExternalAgentTimeSaving.user_id,
                func.sum(ExternalAgentTimeSaving.minutes_saved).label("external_minutes_saved")
            )
            .where(
                and_(self._get_date_range_filter(ExternalAgentTimeSaving.date, from_date, to_date), external_agent_filter)
            )
            .group_by(col(ExternalAgentTimeSaving.user_id))
            .subquery()
        )

    def _get_minutes_saved_subquery(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int, agent_filter: ColumnElement[bool] | bool = True, filtered_user_id: Optional[int] = None):
        return (
            select(
                func.sum(ThreadMessage.minutes_saved).label("minutes_saved")
            )
            .join(Thread, and_(ThreadMessage.thread_id == Thread.id))
            .join(User, and_(Thread.user_id == User.id))
            .outerjoin(Agent, and_(Agent.id == Thread.agent_id))
            .where(
                and_(
                    self._get_thread_message_filter_condition(from_date, to_date, team_id, user_id, filtered_user_id),
                    agent_filter
                )
            )
        ).subquery()

    def _get_total_threads_subquery(self, from_date: datetime, to_date: datetime, team_id: int, user_id: int, agent_filter: ColumnElement[bool] | bool = True, group_by_user: bool = False, filtered_user_id: Optional[int] = None):
        base_query = select(func.count(distinct(Thread.id)).label("total_threads"))

        if group_by_user:
            base_query = select(
                Thread.user_id,
                func.count(distinct(Thread.id)).label("total_threads")
            )

        query = (
            base_query
            .join(User, and_(Thread.user_id == User.id))
            .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .outerjoin(Agent, and_(Thread.agent_id == Agent.id))
            .where(
                and_(
                    not_(Thread.is_test_case),
                    self._get_date_range_filter(ThreadMessage.timestamp, from_date, to_date),
                    self._get_user_filter_condition(user_id, team_id, filtered_user_id=filtered_user_id) if not group_by_user else True,
                    agent_filter
                )
            )
        )

        if group_by_user:
            query = query.group_by(col(Thread.user_id))

        return query.subquery()

    def _get_agent_user_filter_from_threads(self, from_date: datetime, to_date: datetime, team_id: int, agent_id: Optional[int]):
        subquery = None

        if agent_id == PRIVATE_AGENT_ID:
            subquery = (
                select(Thread.user_id)
                .join(Agent, and_(Thread.agent_id == Agent.id))
                .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
                .where(
                    and_(
                        not_(Thread.is_test_case),
                        self._get_date_range_filter(ThreadMessage.timestamp, from_date, to_date),
                        self._get_private_agents_filter(team_id),
                    )
                )
            )
        elif agent_id:
            subquery = (
                select(Thread.user_id)
                .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
                .where(
                    and_(
                        not_(Thread.is_test_case),
                        self._get_date_range_filter(ThreadMessage.timestamp, from_date, to_date),
                        Thread.agent_id == agent_id
                    )
                )
            )

        return col(User.id).in_(subquery) if subquery is not None else True

    def _get_previous_period(self, from_date: datetime, to_date: datetime) -> datetime:
        date_range = to_date - from_date
        return from_date - timedelta(days=date_range.days)

    def _get_date_range_filter(self, column: Any, from_date: datetime, to_date: datetime)-> ColumnElement[bool]:
        return and_(col(column) >= from_date, col(column) < to_date)
