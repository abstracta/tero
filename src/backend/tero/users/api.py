from typing import Annotated

from fastapi import APIRouter, Depends, status, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.api import BASE_PATH
from ..core.auth import get_current_user
from ..core.repos import get_db
from ..teams.domain import OWNER_ROLES, TeamRoleStatus
from ..teams.repos import TeamRepository
from .domain import User, UserListItem, UserProfile, UserUpdate
from .repos import UserRepository


router = APIRouter()
USERS_PATH = f"{BASE_PATH}/users"


@router.get(USERS_PATH, response_model=list[UserListItem])
async def get_users(user: Annotated[User, Depends(get_current_user)], db: AsyncSession = Depends(get_db)):
    if not any(tr.role in OWNER_ROLES for tr in user.team_roles):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return await UserRepository(db).find_all_users()


CURRENT_USER_PATH = f"{USERS_PATH}/current"


@router.get(CURRENT_USER_PATH)
# Legacy endpoint, to be removed
@router.get(f"{USERS_PATH}/profile")
async def get_user_profile(user: Annotated[User, Depends(get_current_user)]) -> UserProfile:
    return UserProfile.from_user(user)


CURRENT_USER_TEAM_PATH = f"{CURRENT_USER_PATH}/teams/{{team_id}}"


@router.put(CURRENT_USER_TEAM_PATH)
async def accept_team_invitation(team_id: int, user: Annotated[User, Depends(get_current_user)], db: AsyncSession = Depends(get_db)):
    team_role = await _find_pending_team_role(team_id, user.id, db)
    team_role.status = TeamRoleStatus.ACCEPTED
    await TeamRepository(db).save_team_role(team_role)


async def _find_pending_team_role(team_id: int, user_id: int, db: AsyncSession):
    team_role = await TeamRepository(db).find_team_role(team_id, user_id)
    if team_role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if team_role.status != TeamRoleStatus.PENDING:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    return team_role


@router.delete(CURRENT_USER_TEAM_PATH)
async def reject_team_invitation(team_id: int, user: Annotated[User, Depends(get_current_user)], db: AsyncSession = Depends(get_db)):
    team_role = await _find_pending_team_role(team_id, user.id, db)
    team_role.status = TeamRoleStatus.REJECTED
    await TeamRepository(db).save_team_role(team_role)


USER_PATH = f"{USERS_PATH}/{{user_id}}"


@router.put(USER_PATH)
async def update_user(user_id: int, request: UserUpdate, current_user: Annotated[User, Depends(get_current_user)], db: AsyncSession = Depends(get_db)) -> UserUpdate:
    if request.exclude_from_metrics is not None and not current_user.is_global_owner():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only global team owners can update this")
    if request.monthly_usd_limit is not None and not current_user.is_budget_owner():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Budget owner access required to update user budget")

    user_repo = UserRepository(db)
    user_to_update = await user_repo.find_by_id(user_id)
    if user_to_update is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    if request.exclude_from_metrics is not None:
        user_to_update.exclude_from_metrics = request.exclude_from_metrics
    if request.monthly_usd_limit is not None:
        user_to_update.monthly_usd_limit = request.monthly_usd_limit
    await user_repo.update_user(user_to_update)

    return UserUpdate(
        exclude_from_metrics=user_to_update.exclude_from_metrics,
        monthly_usd_limit=user_to_update.monthly_usd_limit if current_user.is_budget_owner() else None,
    )
