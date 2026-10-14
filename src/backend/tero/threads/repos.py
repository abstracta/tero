from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import selectinload, aliased
from sqlmodel import select, func, or_, and_, col, delete
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.domain import Agent, AgentListItem, AgentThread
from ..agents.repos import AgentRepository
from ..core.repos import attr, scalar
from ..files.domain import File
from ..users.domain import User
from .domain import Thread, ThreadListItem, ThreadMessage, ThreadMessageFile


class ThreadRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def add(self, thread: Thread) -> Thread:
        self._db.add(thread)
        await self._db.commit()
        await self._db.refresh(thread, ['id', 'agent', 'user'])
        await self._db.refresh(thread.agent, ['user'])
        if not thread.name:
            thread.set_default_name()
        self._db.add(thread)
        await self._db.commit()
        return thread

    async def find_by_text(self, text: Optional[str], user: User, limit: int, agent_id: Optional[int], exclude_empty: bool = False,
            offset: int = 0) -> List[ThreadListItem]:
        stmt = (
            select(Thread, func.max(ThreadMessage.timestamp).label('last_message'))
            .outerjoin(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .where(Thread.user_id == user.id,
                Thread.agent_id == agent_id if agent_id is not None else True,
                or_(col(Thread.name).ilike(f"%{text}%"),
                    col(ThreadMessage.text).ilike(f"%{text}%")) if text else True,
                   Thread.deleted == False,
                   Thread.is_test_case == False,
                   col(ThreadMessage.id).isnot(None) if exclude_empty else True
            )
            .group_by(col(Thread.id))
            .order_by(func.coalesce(func.max(ThreadMessage.timestamp), Thread.creation).desc(), col(Thread.id).desc())
            .limit(limit)
            .offset(offset)
            .options(
                selectinload(attr(Thread.agent)).selectinload(attr(Agent.user)),
                selectinload(attr(Thread.agent)).selectinload(attr(Agent.team)),
                selectinload(attr(Thread.user)))
        )
        threads = (await self._db.exec(stmt)).all()
        agents = await AgentRepository(self._db).build_agent_list_items_for_user((thread.agent for thread, _ in threads), user)
        return [ThreadListItem.from_thread(thread, last_message, agent=agents[thread.agent_id]) for thread, last_message in threads]

    async def find_latest_by_agents(self, agents: List[AgentListItem], user_id: int, limit: int) -> List[AgentListItem]:
        last_message = func.max(ThreadMessage.timestamp)
        agent_rank = func.row_number().over(partition_by=col(Thread.agent_id), order_by=(last_message.desc(), col(Thread.id).desc()))
        # inner join with messages excludes empty threads
        ranked = (
            select(Thread, last_message.label("last_message"), agent_rank.label("agent_rank"))
            .join(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .where(Thread.user_id == user_id, col(Thread.agent_id).in_([agent.id for agent in agents]),
                   Thread.deleted == False, Thread.is_test_case == False)
            .group_by(col(Thread.id))
            .subquery())
        thread = aliased(Thread, ranked)
        # fetch one extra chat per agent to know if there are more
        ret = await self._db.exec(select(thread, ranked.c.last_message).where(ranked.c.agent_rank <= limit + 1).order_by(ranked.c.agent_rank))
        chats: dict[int, List[AgentThread]] = {agent.id: [] for agent in agents}
        for t, last in ret.all():
            chats[t.agent_id].append(AgentThread(id=t.id, name=t.name, creation=t.creation, last_message=last))
        return [agent.model_copy(update={"chats": chats[agent.id][:limit], "has_more_chats": len(chats[agent.id]) > limit}) for agent in agents]

    async def find_by_id(self, thread_id: int, user_id: int, is_test_case: bool = False) -> Optional[Thread]:
        stmt = (
            select(Thread)
            .where(Thread.id == thread_id,
                   Thread.user_id == user_id,
                   Thread.deleted == False,
                   Thread.is_test_case == is_test_case)
            # this is required since sqlmodel has no way yet to lazy load async relationships (https://github.com/fastapi/sqlmodel/issues/74)
            .options(
                selectinload(attr(Thread.agent)).selectinload(attr(Agent.model)),
                selectinload(attr(Thread.agent)).selectinload(attr(Agent.user)),
                selectinload(attr(Thread.agent)).selectinload(attr(Agent.team)),
                selectinload(attr(Thread.user))))
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def delete(self, thread: Thread):
        thread.deleted = True
        self._db.add(thread)
        await self._db.commit()

    async def update(self, thread: Thread):
        merged_thread = await self._db.merge(thread)
        await self._db.commit()
        await self._db.refresh(merged_thread)
        return merged_thread

    async def find_empty_thread(self, agent_id: int, user_id: int) -> Optional[Thread]:
        stmt = (
            select(Thread)
            .outerjoin(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .where(Thread.agent_id == agent_id,
                   Thread.user_id == user_id,
                   Thread.deleted == False,
                   Thread.is_test_case == False)
            .group_by(col(Thread.id))
            .having(func.count(col(ThreadMessage.id)) == 0)
            .order_by(col(Thread.creation).desc())
            .limit(1)
            .options(
                selectinload(attr(Thread.agent)).selectinload(attr(Agent.user)),
                selectinload(attr(Thread.user)))
        )
        ret = await self._db.exec(stmt)
        return ret.first()


class ThreadMessageRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def add(self, thread_message: ThreadMessage):
        self._db.add(thread_message)
        await self._db.commit()
        await self._db.refresh(thread_message, ['files', 'timestamp'])
        return thread_message

    async def refresh_with_files(self, thread_message: ThreadMessage) -> ThreadMessage:
        self._db.expire(thread_message, ['files'])
        stmt = (
            select(ThreadMessage)
            .where(ThreadMessage.id == thread_message.id)
            .options(
                selectinload(attr(ThreadMessage.files)).selectinload(attr(ThreadMessageFile.file))
            ))
        ret = await self._db.exec(stmt)
        return ret.one()

    async def update(self, thread_message: ThreadMessage):
        self._db.add(thread_message)
        await self._db.commit()

    async def find_by_thread_id(self, thread_id: int) -> List[ThreadMessage]:
        stmt = (
            select(ThreadMessage)
            .where(and_(ThreadMessage.thread_id == thread_id))
            .order_by(col(ThreadMessage.timestamp))
            .options(
                selectinload(attr(ThreadMessage.files)).selectinload(attr(ThreadMessageFile.file))
            ))
        ret = await self._db.exec(stmt)
        return list(ret.all())

    async def find_last_by_thread_id(self, thread_id: int) -> Optional[ThreadMessage]:
        stmt = (
            select(ThreadMessage)
            .where(ThreadMessage.thread_id == thread_id)
            .order_by(col(ThreadMessage.timestamp).desc())
            .limit(1)
        )
        ret = await self._db.exec(stmt)
        return ret.first()

    async def find_previous_messages(self, message: ThreadMessage) -> List[ThreadMessage]:
        parents: List[ThreadMessage] = []
        current = message
        while current.parent_id is not None:
            parent = await self.find_by_id(current.parent_id)
            if parent is None:
                break
            parents.append(parent)
            current = parent
        parents.reverse()
        return parents

    async def find_by_id(self, message_id: int, user_id: Optional[int] = None) -> Optional[ThreadMessage]:
        stmt = select(ThreadMessage).where(ThreadMessage.id == message_id)
        if user_id is not None:
            stmt = stmt.join(Thread, and_(ThreadMessage.thread_id == Thread.id)).where(
                Thread.user_id == user_id,
                Thread.deleted == False)
        stmt = stmt.options(
            selectinload(attr(ThreadMessage.files)).selectinload(attr(ThreadMessageFile.file))
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_by_thread_id_and_message_id(self, thread_id: int, message_id: int) -> Optional[ThreadMessage]:
        stmt = (
            select(ThreadMessage)
            .where(and_(ThreadMessage.thread_id == thread_id, ThreadMessage.id == message_id))
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_feedback_messages(self, agent_id: int, user_id: int, limit: int) -> List[ThreadMessage]:
        messages = []
        strategies = [
            [Thread.agent_id == agent_id, Thread.user_id == user_id],
            [Thread.agent_id == agent_id],
            [Thread.user_id == user_id],
        ]

        for strategy_conditions in strategies:
            if len(messages) >= limit:
                break

            conditions = strategy_conditions + [~col(ThreadMessage.id).in_([msg.id for msg in messages])] if messages else []
            messages.extend(await self._find_feedback_messages_with_conditions(conditions, limit - len(messages)))

        return messages[:limit]

    async def _find_feedback_messages_with_conditions(self, additional_conditions, limit: int) -> List[ThreadMessage]:
        parent_message = aliased(ThreadMessage)
        stmt = (
            select(ThreadMessage, parent_message)
            .join(Thread, and_(ThreadMessage.thread_id == Thread.id))
            .join(parent_message, and_(ThreadMessage.parent_id == parent_message.id))
            .where(
                or_(ThreadMessage.has_positive_feedback == True, ThreadMessage.has_positive_feedback == False),
                Thread.deleted == False,
                *additional_conditions
            )
            .order_by(col(ThreadMessage.timestamp).desc())
            .limit(limit)
        )

        ret = await self._db.exec(stmt)
        return [item for message, parent in ret.all() for item in [parent, message]]

    async def delete_from_date(self, thread_id: int, date:datetime) -> None:
        stmt = scalar(delete(ThreadMessage)
                .where(and_(ThreadMessage.thread_id == thread_id,
                       ThreadMessage.timestamp >= date)))
        await self._db.exec(stmt)
        await self._db.commit()

    async def delete_by_thread_id(self, thread_id: int):
        stmt = scalar(delete(ThreadMessage).where(and_(ThreadMessage.thread_id == thread_id)))
        await self._db.exec(stmt)
        await self._db.commit()


class ThreadMessageFileRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def add(self, thread_message_file: ThreadMessageFile) -> ThreadMessageFile:
        self._db.add(thread_message_file)
        await self._db.commit()
        await self._db.refresh(thread_message_file)
        return thread_message_file

    async def find_by_thread_id_and_file_id(self, thread_id: int, file_id: int) -> Optional[ThreadMessageFile]:
        stmt = (select(ThreadMessageFile)
            .join(ThreadMessage, and_(ThreadMessageFile.thread_message_id == ThreadMessage.id, ThreadMessage.thread_id == thread_id))
            .where(ThreadMessageFile.file_id == file_id)
            .limit(1))
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_with_content_by_ids(self, thread_id: int, file_id: int) -> Optional[File]:
        stmt = (select(File)
            .join(ThreadMessageFile, and_(ThreadMessageFile.file_id == File.id))
            .join(ThreadMessage, and_(ThreadMessageFile.thread_message_id == ThreadMessage.id, ThreadMessage.thread_id == thread_id))
            .where(File.id == file_id)
            .limit(1))
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_with_content_by_thread_id(self, thread_id: int) -> List[File]:
        stmt = (
            select(File)
            .join(ThreadMessageFile, and_(ThreadMessageFile.file_id == File.id))
            .join(ThreadMessage, and_(ThreadMessageFile.thread_message_id == ThreadMessage.id))
            .where(ThreadMessage.thread_id == thread_id)
            .order_by(col(ThreadMessage.timestamp).desc(), col(File.id).desc())
        )
        seen: set[int] = set()
        files: List[File] = []
        for file in (await self._db.exec(stmt)).all():
            if file.id not in seen:
                seen.add(file.id)
                files.append(file)
        return files

    async def delete(self, thread_message_file: ThreadMessageFile):
        await self._db.delete(thread_message_file)
        await self._db.commit()
