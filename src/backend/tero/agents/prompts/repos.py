from typing import Optional, List

from sqlmodel import select, or_, update, and_, delete
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import asc

from ...core.repos import scalar
from .domain import AgentPrompt


class AgentPromptRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def add(self, prompt: AgentPrompt) -> AgentPrompt:
        self._db.add(prompt)
        await self._db.commit()
        await self._db.refresh(prompt)
        return prompt

    async def find_by_id(self, prompt_id: int) -> Optional[AgentPrompt]:
        stmt = (
            select(AgentPrompt)
            .where(
                AgentPrompt.id == prompt_id,
            )
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_user_agent_prompts(self, user_id: int, agent_id: int, only_private: bool = False) -> List[AgentPrompt]:
        stmt = (
            select(AgentPrompt)
            .where(AgentPrompt.agent_id == agent_id,
                   and_(AgentPrompt.user_id == user_id, AgentPrompt.shared == False) if only_private
                   else or_(AgentPrompt.user_id == user_id, AgentPrompt.shared == True))
            .order_by(asc(AgentPrompt.name))
        )
        ret = await self._db.exec(stmt)
        return list(ret.all())
    
    async def publish_all_private_prompts(self, agent_id: int):
        stmt = scalar(
            update(AgentPrompt)
            .where(and_(AgentPrompt.agent_id == agent_id, 
                        AgentPrompt.shared == False))
            .values(shared=True)
        )
        await self._db.exec(stmt)
        await self._db.commit()

    async def update(self, prompt: AgentPrompt):
        self._db.add(prompt)
        await self._db.commit()
        await self._db.refresh(prompt)
        return prompt

    async def delete(self, prompt: AgentPrompt):
        await self._db.delete(prompt)
        await self._db.commit()

    
    async def delete_user_agent_prompts(self, user_id: int, agent_id: int):
        stmt = (
            delete(AgentPrompt)
            .where(and_(AgentPrompt.agent_id == agent_id,
                        or_(AgentPrompt.user_id == user_id, AgentPrompt.shared == True)))
        )
        await self._db.exec(stmt)
        await self._db.commit()


    async def add_many(self, prompts: List[AgentPrompt]) -> None:
        self._db.add_all(prompts)
        await self._db.commit()
