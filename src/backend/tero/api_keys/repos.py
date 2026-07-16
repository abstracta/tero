from datetime import datetime, timezone
from typing import Optional

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .domain import ApiKey


class ApiKeyRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def create(self, api_key: ApiKey) -> ApiKey:
        self._db.add(api_key)
        await self._db.commit()
        await self._db.refresh(api_key, ["id"])
        return api_key

    async def find_by_id(self, api_key_id: int) -> Optional[ApiKey]:
        stmt = select(ApiKey).where(ApiKey.id == api_key_id)
        result = await self._db.exec(stmt)
        return result.one_or_none()

    async def find_by_key_id(self, key_id: str) -> Optional[ApiKey]:
        stmt = select(ApiKey).where(ApiKey.key_id == key_id)
        result = await self._db.exec(stmt)
        return result.one_or_none()

    async def update_last_used(self, api_key: ApiKey) -> None:
        api_key.last_used_at = datetime.now(timezone.utc)
        self._db.add(api_key)
        await self._db.commit()
