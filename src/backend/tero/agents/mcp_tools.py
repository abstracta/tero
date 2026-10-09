from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.api import MCP_PATH
from ..core.auth import get_current_user
from ..core.repos import get_db
from ..users.domain import User
from .repos import AgentRepository


class McpAgent(BaseModel):
    id: int
    name: Optional[str]
    description: Optional[str]


router = APIRouter()


@router.post(f"{MCP_PATH}/list-agents", operation_id="list-agents")
async def list_agents(
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
        text: Annotated[Optional[str], Query(description="Optional text to filter agents by name or description")] = None,
) -> List[McpAgent]:
    """
    List available AI agents.

    Returns the agents accessible to the current user, optionally filtered by
    a text search on name or description.
    """
    repo = AgentRepository(db)
    agents = await repo.find_user_agents(text, user)
    return [McpAgent(id=a.id, name=a.name, description=a.description) for a in agents if not a.read_only]
