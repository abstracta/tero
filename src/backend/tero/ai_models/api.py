from typing import Annotated, List

from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from .domain import LlmModelResponse
from .repos import AiModelRepository
from ..core.api import BASE_PATH
from ..core.auth import get_current_user
from ..core.env import env
from ..core.repos import get_db
from ..users.domain import User

router = APIRouter()

@router.get(f"{BASE_PATH}/models")
# user is added to contract just to require authentication to get the available agent tools
async def find_models(_: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) \
        -> List[LlmModelResponse]:
    models = await AiModelRepository(db).find_all()
    base_model_id = env.agent_default_model
    base_cost = next((m.completion_1k_token_usd for m in models if m.id == base_model_id), None)
    return [LlmModelResponse.from_model(m, base_cost, base_model_id) for m in models]
