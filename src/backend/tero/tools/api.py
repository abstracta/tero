import logging
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.repos import AgentToolConfigRepository
from ..core.api import BASE_PATH
from ..core.auth import get_current_user
from ..core.repos import get_db
from ..users.domain import User
from .core import AgentTool
from .auth import ToolAuthCallbackError, ToolOAuthCallback, ToolAuthTokenCallback, ToolAuthRepository, ToolOAuthState
from .repos import ToolRepository
from .notion import NOTION_TOOL_ID


logger = logging.getLogger(__name__)
router = APIRouter()
TOOLS_PATH = f"{BASE_PATH}/tools"


def require_owner_for_owner_configured_tool(agent_user_id: Optional[int], tool_id: str, user_id: int) -> None:
    if tool_id in [NOTION_TOOL_ID] and agent_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the agent owner can configure this tool",
        )


@router.get(TOOLS_PATH)
# user is added to contract just to require authentication to get the available agent tools
async def find_agent_tools(_: Annotated[User, Depends(get_current_user)]) -> List[AgentTool]:
    return ToolRepository().find_agent_tools()


@router.put(f"{TOOLS_PATH}/{{tool_id}}/agents/{{agent_id}}/auth")
async def complete_tool_auth(tool_id: str, agent_id: int, callback: ToolOAuthCallback | ToolAuthTokenCallback, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]):
    if isinstance(callback, ToolAuthTokenCallback) and not callback.auth_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Auth token is required")
    if isinstance(callback, ToolOAuthCallback) and not callback.code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Authentication cancelled")
    config = await AgentToolConfigRepository(db).find_by_ids(agent_id, tool_id, include_drafts=True)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    tool = _find_tool(tool_id)
    require_owner_for_owner_configured_tool(config.agent.user_id, tool.id, user.id)
    tool.configure(config.agent, user.id, config.config, db)
    try:
        await tool.auth(callback)
    except ToolAuthCallbackError:
        logger.exception(f"Error during tool auth for tool {tool_id} and user {user.id}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)


def _find_tool(tool_id: str) -> AgentTool:
    tool = ToolRepository().find_by_id(tool_id)
    if not tool:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool not found")
    return tool


@router.delete(f"{TOOLS_PATH}/{{tool_id}}/oauth/{{state_id}}")
async def delete_oauth_auth(tool_id: str, state_id: str, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]):
    _find_tool(tool_id)
    state = await _find_state(user.id, tool_id, state_id, db)
    await ToolAuthRepository(db).delete_state(user.id, state.agent_id, tool_id)


async def _find_state(user_id: int, tool_id: str, state_id: str, db: AsyncSession) -> ToolOAuthState:
    oauth_repo = ToolAuthRepository(db)
    state = await oauth_repo.find_state(user_id, tool_id, state_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return state
