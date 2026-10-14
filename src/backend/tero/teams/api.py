from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.repos import AgentRepository
from ..core.api import BASE_PATH
from ..core.auth import get_current_user
from ..core.env import env
from ..core.repos import get_db
from ..users.domain import User
from ..users.repos import UserRepository
from .domain import GLOBAL_TEAM_ID, OWNER_ROLES, AddUsersToTeam, Role, TeamRole, TeamRoleStatus,\
    TeamRoleUpdate, TeamUser, TeamCreate, Team, TeamUpdate
from .repos import TeamRepository


router = APIRouter()
TEAMS_PATH = f"{BASE_PATH}/teams"
TEAM_USERS_PATH = f"{TEAMS_PATH}/{{team_id}}/users"
TEAM_PATH = f"{TEAMS_PATH}/{{team_id}}"


@router.get(TEAM_USERS_PATH)
async def get_team_users(team_id: int, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)], limit: Optional[int] = None, offset: Optional[int] = None, search: Optional[str] = None) -> List[TeamUser]:
    await _find_team(team_id, TeamRepository(db))
    await _check_team_owner(team_id, user)
    return await TeamRepository(db).find_team_users(team_id, user, limit, offset, search)


@router.get(TEAMS_PATH, status_code=status.HTTP_200_OK)
async def find_teams(user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> List[Team]:
    await _check_team_owner(GLOBAL_TEAM_ID, user)
    teams = await TeamRepository(db).find_teams()
    return teams


@router.post(TEAMS_PATH, status_code=status.HTTP_201_CREATED)
async def create_team(team: TeamCreate, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> Team:
    await _check_team_owner(GLOBAL_TEAM_ID, user)
    repo = TeamRepository(db)
    created = await repo.add(Team(name=team.name))
    if team.users:
        await _add_users_to_team(created.id, team.users, db, user)
    return created


@router.put(TEAM_PATH)
async def update_team(team_id: int, updated: TeamUpdate, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> None:
    await _check_team_owner(GLOBAL_TEAM_ID, user)
    await _check_global_team(team_id)
    team = await _find_team(team_id, TeamRepository(db))
    team.name = updated.name
    await TeamRepository(db).update(team)


@router.delete(TEAM_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def delete_team(team_id: int, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> None:
    await _check_team_owner(GLOBAL_TEAM_ID, user)
    await _check_global_team(team_id)
    team_repo = TeamRepository(db)
    team = await _find_team(team_id, team_repo)
    await AgentRepository(db).remove_team_agents(team.id)
    await team_repo.remove_team_roles(team.id)
    await team_repo.delete(team)


async def _check_team_owner(team_id: int, user: User) -> None:
    if any(
        tr.role in OWNER_ROLES
        and tr.status == TeamRoleStatus.ACCEPTED
        and (tr.team_id == team_id or tr.team_id == GLOBAL_TEAM_ID)
        for tr in user.team_roles
    ):
        return

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


def _check_can_assign_budget_owner(role: Role, team_id: int, user: User, current_role: Optional[Role] = None) -> None:
    if role == Role.BUDGET_OWNER or current_role == Role.BUDGET_OWNER:
        if role == Role.BUDGET_OWNER and team_id != GLOBAL_TEAM_ID:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Budget owner role can only be assigned in the global team")
        if not user.is_budget_owner():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Budget owner access required to modify the budget owner role")


async def _check_global_team(team_id: int) -> None:
    if team_id == GLOBAL_TEAM_ID:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Global team cannot be updated")


async def _find_team(team_id: int, repo: TeamRepository) -> Team:
    team = await repo.find_team(team_id)
    if team is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    return team


@router.post(TEAM_USERS_PATH)
async def add_users_to_team(team_id: int, new_users: List[AddUsersToTeam], user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> List[TeamUser]:
    await _find_team(team_id, TeamRepository(db))
    await _check_team_owner(team_id, user)
    return await _add_users_to_team(team_id, new_users, db, user)


async def _add_users_to_team(team_id: int, new_users: List[AddUsersToTeam], db: Annotated[AsyncSession, Depends(get_db)], user: User) -> List[TeamUser]:
    users_repo = UserRepository(db)
    team_repo = TeamRepository(db)
    added_users = []

    for new_user in new_users:
        found_user = await users_repo.find_by_username(new_user.username)
        if found_user is None:
            found_user = await users_repo.create_user(User(username=new_user.username, monthly_usd_limit=env.monthly_usd_limit_default))
        current_role = next((tr.role for tr in found_user.team_roles if tr.team_id == team_id), None)
        _check_can_assign_budget_owner(new_user.role, team_id, user, current_role)
        if team_id == GLOBAL_TEAM_ID and new_user.role == Role.TEAM_MEMBER:
            continue
        user_status = TeamRoleStatus.ACCEPTED if (user.id and found_user.id == user.id) or (team_id == GLOBAL_TEAM_ID and new_user.role in OWNER_ROLES) else TeamRoleStatus.PENDING
        await team_repo.save_team_role(TeamRole(user_id=found_user.id, team_id=team_id, role=new_user.role, status=user_status))

        added_users.append(TeamUser(
            id=found_user.id,
            username=found_user.username,
            name=found_user.name,
            role=new_user.role,
            role_status=user_status,
            verified=found_user.name is not None,
            exclude_from_metrics=found_user.exclude_from_metrics,
            monthly_usd_limit=found_user.monthly_usd_limit if user.is_budget_owner() else None,
        ))

    return added_users

TEAM_USER_PATH = f"{TEAM_USERS_PATH}/{{user_id}}"


@router.put(TEAM_USER_PATH)
async def update_user_role_in_team(team_id: int, user_id: int, request: TeamRoleUpdate, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]):
    target_user = await _find_user(user_id, db)
    await _check_team_owner(team_id, user)
    current_role = next((tr.role for tr in target_user.team_roles if tr.team_id == team_id), None)
    _check_can_assign_budget_owner(request.role, team_id, user, current_role)
    repo = TeamRepository(db)
    if team_id == GLOBAL_TEAM_ID and request.role == Role.TEAM_MEMBER:
        await repo.delete_team_role(team_id, user_id)
    else:
        role_status = await _find_team_role_status(team_id, user_id, db)
        await repo.save_team_role(TeamRole(user_id=user_id, team_id=team_id, role=request.role, status=role_status))


async def _find_user(user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    user = await UserRepository(db).find_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return user


async def _find_team_role_status(team_id: int, user_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    team_role = await TeamRepository(db).find_team_role(team_id, user_id)
    if team_role is None and team_id == GLOBAL_TEAM_ID:
        return TeamRoleStatus.ACCEPTED
    if team_role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team role not found")
    return team_role.status


@router.delete(TEAM_USER_PATH)
async def delete_user_from_team(team_id: int, user_id: int, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]):
    user_to_delete = await _find_user(user_id, db)
    await _find_team(team_id, TeamRepository(db))
    await _check_team_owner(team_id, user)
    team_repo = TeamRepository(db)
    await team_repo.delete_team_role(team_id, user_id)
    team_roles = await team_repo.find_user_team_roles(user_id)
    if not team_roles and not user_to_delete.name:
        await UserRepository(db).delete_user(user_id)
