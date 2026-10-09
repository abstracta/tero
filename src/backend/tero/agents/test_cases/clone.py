from datetime import datetime, timezone
from typing import Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from ...threads.domain import Thread, ThreadMessage
from ...threads.repos import ThreadRepository, ThreadMessageRepository
from ..evaluators.repos import EvaluatorRepository
from .domain import TestCase
from .repos import TestCaseRepository


async def clone_test_case(
    original_test_case: TestCase,
    target_agent_id: int,
    user_id: int,
    db: AsyncSession,
    test_case_name: Optional[str] = None
) -> TestCase:
    thread_name = test_case_name if test_case_name is not None else original_test_case.thread.name

    cloned_thread = await ThreadRepository(db).add(
        Thread(
            agent_id=target_agent_id,
            user_id=user_id,
            is_test_case=True,
            name=thread_name,
            creation=datetime.now(timezone.utc)
        )
    )

    cloned_test_case = TestCase(
        thread_id=cloned_thread.id,
        agent_id=target_agent_id,
        last_update=datetime.now(timezone.utc)
    )

    if original_test_case.evaluator_id:
        evaluator_repo = EvaluatorRepository(db)
        original_evaluator = await evaluator_repo.find_by_id(original_test_case.evaluator_id)
        if original_evaluator:
            cloned_evaluator = await evaluator_repo.save(original_evaluator.clone())
            cloned_test_case.evaluator_id = cloned_evaluator.id

    cloned_test_case = await TestCaseRepository(db).save(cloned_test_case)

    thread_message_repo = ThreadMessageRepository(db)
    original_messages = await thread_message_repo.find_by_thread_id(original_test_case.thread_id)
    for message in original_messages:
        await thread_message_repo.add(ThreadMessage(
            thread_id=cloned_thread.id,
            origin=message.origin,
            text=message.text,
            collapsed_name=message.collapsed_name,
            timestamp=message.timestamp,
        ))

    return cloned_test_case
