from typing import List, Optional

from sqlmodel import select, col
from sqlmodel.ext.asyncio.session import AsyncSession

from .domain import LlmModel
from .ai_factory import has_valid_provider


class AiModelRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_all(self) -> List[LlmModel]:
        ret = await self._db.exec(select(LlmModel).order_by(col(LlmModel.completion_1k_token_usd), col(LlmModel.id)))
        return [model for model in ret.all() if has_valid_provider(model.id)]

    async def find_by_id(self, model_id: str) -> Optional[LlmModel]:
        return await self._db.get(LlmModel, model_id)
