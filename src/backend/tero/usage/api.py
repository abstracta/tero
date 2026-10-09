from typing import Annotated, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.repos import AgentRepository
from ..core.api import BASE_PATH
from ..core.auth import get_current_user
from ..core.domain import CamelCaseModel
from ..core.repos import get_db
from ..external_agents.repos import ExternalAgentRepository
from ..teams.domain import OWNER_ROLES, TeamRoleStatus, MY_TEAM_ID, GLOBAL_TEAM_ID
from ..teams.repos import TeamRepository
from ..usage.domain import PRIVATE_AGENT_ID
from ..users.domain import User
from .domain import ImpactSummary, AgentImpactItem, UserImpactItem, UsageSummary, AgentUsageItem, UserUsageItem
from .repos import UsageRepository


router = APIRouter()


class UserBudget(CamelCaseModel):
    usage_percent: float


@router.get(f"{BASE_PATH}/budget")
async def get_user_budget(user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> UserBudget:
    usage = await UsageRepository(db).find_current_month_user_usage_usd(user.id)
    return UserBudget(usage_percent=usage / user.monthly_usd_limit)

IMPACT_PATH = f"{BASE_PATH}/impact"

@router.get(f"{IMPACT_PATH}/summary")
async def get_impact_summary(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    from_date: datetime,
    to_date: datetime,
    team_id: int
) -> ImpactSummary:
    _verify_owner_role(user, team_id)
    _verify_dates(from_date, to_date)
    repo = UsageRepository(db)
    return await repo.get_impact_summary(from_date, to_date, team_id, user.id)

@router.get(f"{IMPACT_PATH}/agents")
async def get_impact_top_agents(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    from_date: datetime,
    to_date: datetime,
    team_id: int,
    user_id: Optional[int] = None,
    search: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
) -> List[AgentImpactItem]:
    _verify_owner_role(user, team_id)
    _verify_dates(from_date, to_date)
    if user_id is not None:
        team_role = await TeamRepository(db).find_team_role(team_id, user_id)
        if team_role is None and team_id != GLOBAL_TEAM_ID:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not belong to team")
    repo = UsageRepository(db)
    return await repo.get_impact_top_agents(from_date, to_date, team_id, search, limit, offset, user.id, user_id)

@router.get(f"{IMPACT_PATH}/users")
async def get_impact_top_users(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    from_date: datetime,
    to_date: datetime,
    team_id: int,
    agent_id: Optional[int] = None,
    is_external_agent: Optional[bool] = None,
    search: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
) -> List[UserImpactItem]:
    _verify_owner_role(user, team_id)
    _verify_dates(from_date, to_date)
    await check_filtered_agent(db, agent_id, team_id, is_external_agent)
    repo = UsageRepository(db)
    return await repo.get_impact_top_users(from_date, to_date, team_id, search, limit, offset, user.id, agent_id, is_external_agent)

USAGE_PATH = f"{BASE_PATH}/usage"

@router.get(f"{USAGE_PATH}/summary")
async def get_usage_summary(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    from_date: datetime,
    to_date: datetime,
    team_id: int,
) -> UsageSummary:
    _verify_owner_role(user, team_id)
    _verify_dates(from_date, to_date)
    repo = UsageRepository(db)
    return await repo.get_usage_summary(from_date, to_date, team_id, user.id)

@router.get(f"{USAGE_PATH}/agents")
async def get_usage_top_agents(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    from_date: datetime,
    to_date: datetime,
    team_id: int,
    user_id: Optional[int] = None,
    search: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> List[AgentUsageItem]:
    _verify_owner_role(user, team_id)
    _verify_dates(from_date, to_date)
    if user_id is not None:
        team_role = await TeamRepository(db).find_team_role(team_id, user_id)
        if team_role is None and team_id != GLOBAL_TEAM_ID:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not belong to team")
    repo = UsageRepository(db)
    return await repo.get_usage_top_agents(from_date, to_date, team_id, search, limit, offset, user.id, user_id)

@router.get(f"{USAGE_PATH}/users")
async def get_usage_top_users(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    from_date: datetime,
    to_date: datetime,
    team_id: int,
    agent_id: Optional[int] = None,
    search: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
) -> List[UserUsageItem]:
    _verify_owner_role(user, team_id)
    _verify_dates(from_date, to_date)
    await check_filtered_agent(db, agent_id, team_id)
    repo = UsageRepository(db)
    return await repo.get_usage_top_users(from_date, to_date, team_id, search, limit, offset, user.id, agent_id)


def _verify_owner_role(user: User, team_id: int) -> None:
    if team_id == MY_TEAM_ID:
        return
    if not any(
        tr.status == TeamRoleStatus.ACCEPTED
        and tr.role in OWNER_ROLES
        and (tr.team_id == team_id or tr.team_id == GLOBAL_TEAM_ID)
        for tr in user.team_roles
    ):

        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner access required")

def _verify_dates(from_date: datetime, to_date: datetime) -> None:
    if from_date > to_date:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="From date must be before to date")

async def check_filtered_agent(db: AsyncSession, agent_id: Optional[int], team_id: int, is_external_agent: Optional[bool] = False) -> None:
    if agent_id is not None and agent_id != PRIVATE_AGENT_ID:
        if is_external_agent:
            external_agent = await ExternalAgentRepository(db).find_by_id(agent_id)
            if external_agent is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="External agent not found")
        else:
            agent = await AgentRepository(db).find_by_id(agent_id)
            if agent is None or (team_id != MY_TEAM_ID and agent.team_id not in (team_id, GLOBAL_TEAM_ID)):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
