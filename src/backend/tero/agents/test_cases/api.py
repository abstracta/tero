import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import logging
import re
from typing import Annotated, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from sse_starlette.event import ServerSentEvent
from langgraph.store.base import BaseStore

from ...core.api import with_heartbeat
from ...core.auth import get_current_user
from ...core.repos import get_db
from ...users.domain import User
from ...threads.agents_store import get_store
from ...threads.domain import Thread, ThreadMessage, ThreadMessageOrigin, ThreadMessagePublic, MAX_THREAD_NAME_LENGTH
from ...threads.engine import AgentEngine
from ...threads.repos import ThreadMessageRepository, ThreadRepository
from ...tools.auth import ToolAuthRequestException, build_tool_auth_request_http_exception
from ..api import AGENT_PATH, find_editable_agent
from ..domain import Agent, CLONE_SUFFIX
from ..repos import AgentRepository
from .clone import clone_test_case
from .domain import TestCase, PublicTestCase, NewTestCaseMessage, UpdateTestCaseMessage, UpdateTestCase, TestSuiteRun, TestSuiteRunStatus, RunTestSuiteRequest, TestCaseResult, TestSuiteEventType, CreateTestCase
from .events import listen_for_suite_run_events, monitor_cancellation
from .name_generation import generate_test_case_name
from .repos import TestCaseRepository, TestCaseResultRepository, TestSuiteRunRepository, TestSuiteRunEventRepository
from .runner import BackgroundTestSuiteRunner


logger = logging.getLogger(__name__)
router = APIRouter()
TEST_CASES_PATH = f"{AGENT_PATH}/tests"


@router.get(TEST_CASES_PATH, response_model=List[PublicTestCase])
async def find_test_cases(agent_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) \
        -> List[TestCase]:
    agent = await find_editable_agent(agent_id, user, db)
    return await TestCaseRepository(db).find_by_agent(agent.id)


@router.post(TEST_CASES_PATH, response_model=PublicTestCase, status_code=status.HTTP_201_CREATED)
async def add_test_case(
    agent_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    create_test_case_options: Optional[CreateTestCase] = None
) -> TestCase:
    agent = await find_editable_agent(agent_id, user, db)
    test_cases_repo = TestCaseRepository(db)
    thread_message_repo = ThreadMessageRepository(db)

    source_chat_messages = None
    if create_test_case_options and create_test_case_options.from_thread_id is not None:
        source_thread = await ThreadRepository(db).find_by_id(create_test_case_options.from_thread_id, user.id)
        if source_thread is None or source_thread.agent_id != agent.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="threadNotFound")

        if create_test_case_options.branch_message_id is not None:
            leaf_message = await thread_message_repo.find_by_thread_id_and_message_id(
                create_test_case_options.from_thread_id, create_test_case_options.branch_message_id
            )
        else:
            leaf_message = await thread_message_repo.find_last_by_thread_id(create_test_case_options.from_thread_id)

        if leaf_message is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="noMessageFound")
        source_chat_messages = await thread_message_repo.find_previous_messages(leaf_message)
        source_chat_messages.append(leaf_message)
    else:
        empty_test_case = await test_cases_repo.find_empty_test_case(agent.id)
        if empty_test_case:
            return empty_test_case

    test_case_count = await test_cases_repo.count_by_agent(agent.id)
    new_thread = await ThreadRepository(db).add(
        Thread(
            agent_id=agent.id,
            user_id=user.id,
            is_test_case=True,
            name=f"Test Case #{test_case_count + 1}"
        )
    )

    test_case = await test_cases_repo.save(
        TestCase(
            thread_id=new_thread.id,
            agent_id=agent.id
        )
    )

    if source_chat_messages:
        prev_origin = None
        for message in source_chat_messages:
            # Insert empty agent message if two consecutive user messages
            if prev_origin == ThreadMessageOrigin.USER and message.origin == ThreadMessageOrigin.USER:
                await thread_message_repo.add(ThreadMessage(
                    thread_id=new_thread.id,
                    origin=ThreadMessageOrigin.AGENT,
                    text=""
                ))
            await thread_message_repo.add(ThreadMessage(
                thread_id=new_thread.id,
                origin=message.origin,
                text=message.text or ""
            ))
            prev_origin = message.origin
        try:
            test_case = await _generate_test_case_name(test_case, user, db)
        except Exception:
            logger.warning("Failed to generate test case name for test case %s", test_case.thread_id, exc_info=True)
        return test_case

    return test_case


TEST_SUITE_RUNS_PATH = f"{TEST_CASES_PATH}/runs"


@router.get(TEST_SUITE_RUNS_PATH)
async def get_test_suite_runs(agent_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)], limit: int = 20, offset: int = 0) -> List[TestSuiteRun]:
    agent = await find_editable_agent(agent_id, user, db)
    return await TestSuiteRunRepository(db).find_by_agent_id(agent.id, limit=limit, offset=offset)


@router.post(TEST_SUITE_RUNS_PATH, status_code=status.HTTP_201_CREATED)
async def run_test_suite(
    agent_id: int,
    request: RunTestSuiteRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    store: Annotated[BaseStore, Depends(get_store)],
) -> TestSuiteRun:
    agent = await find_editable_agent(agent_id, user, db)

    all_test_cases = await TestCaseRepository(db).find_by_agent(agent.id)
    if not all_test_cases:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    if request.test_case_ids is not None:
        test_case_ids_set = set(request.test_case_ids)
        test_cases_to_run = [tc for tc in all_test_cases if tc.thread_id in test_case_ids_set]
        if not test_cases_to_run:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    else:
        test_cases_to_run = all_test_cases

    suite_repo = TestSuiteRunRepository(db)
    last_suite = await suite_repo.find_latest_by_agent_id(agent.id)
    if last_suite and last_suite.status == TestSuiteRunStatus.RUNNING:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)

    # Initialize engine to trigger any tool authentication requirements before creating suite run
    try:
        engine = AgentEngine(agent, user.id, db, store)
        async with AsyncExitStack() as stack:
            await engine.load_tools(stack)
    except ToolAuthRequestException as e:
        raise build_tool_auth_request_http_exception(e.request)

    suite_run = await suite_repo.add(TestSuiteRun(
        agent_id=agent.id,
        status=TestSuiteRunStatus.RUNNING,
        total_tests=len(all_test_cases),
        passed_tests=0,
        failed_tests=0,
        error_tests=0,
        skipped_tests=0
    ))

    stop_event = asyncio.Event()
    runner = BackgroundTestSuiteRunner()
    all_test_case_ids = [tc.thread_id for tc in all_test_cases]
    test_case_ids_to_run = [tc.thread_id for tc in test_cases_to_run]

    async def run_wrapper():
        monitor_task = asyncio.create_task(
            monitor_cancellation(suite_run.id, agent.id, stop_event)
        )
        try:
            await runner.run(
                agent.id,
                all_test_case_ids,
                test_case_ids_to_run,
                user.id,
                suite_run.id,
                stop_event,
                store,
            )
        finally:
            stop_event.set()
            try:
                await monitor_task
            except Exception:
                pass

    asyncio.create_task(run_wrapper())
    return suite_run


TEST_SUITE_RUN_PATH = f"{TEST_SUITE_RUNS_PATH}/{{suite_run_id}}"


@router.get(TEST_SUITE_RUN_PATH)
async def get_test_suite_run(agent_id: int, suite_run_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> TestSuiteRun:
    return await _find_test_suite_run(suite_run_id, agent_id, user, db)


async def _find_test_suite_run(suite_run_id: int, agent_id: int, user: User, db: AsyncSession) -> TestSuiteRun:
    await find_editable_agent(agent_id, user, db)
    suite_run = await TestSuiteRunRepository(db).find_by_id_and_agent_id(suite_run_id, agent_id)
    if suite_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return suite_run


@router.delete(TEST_SUITE_RUN_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def delete_test_suite_run(agent_id: int, suite_run_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]):
    suite_run = await _find_test_suite_run(suite_run_id, agent_id, user, db)
    await TestSuiteRunRepository(db).delete(suite_run)


@router.post(f"{TEST_SUITE_RUN_PATH}/stop", status_code=status.HTTP_200_OK)
async def stop_test_suite_run(
        agent_id: int,
        suite_run_id: int,
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]):
    suite_run = await _find_test_suite_run(suite_run_id, agent_id, user, db)
    if suite_run.status != TestSuiteRunStatus.RUNNING:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="suiteRunNotRunning")

    suite_run.status = TestSuiteRunStatus.CANCELLING
    await TestSuiteRunRepository(db).save(suite_run)


@router.get(f"{TEST_SUITE_RUN_PATH}/stream")
async def stream_test_suite_updates(
    agent_id: int,
    suite_run_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> StreamingResponse:
    await _find_test_suite_run(suite_run_id, agent_id, user, db)

    async def event_generator():
        events_repo = TestSuiteRunEventRepository(db)
        last_event_id = None
        stop_event = asyncio.Event()

        try:
            async for notification in listen_for_suite_run_events(suite_run_id, stop_event):
                if notification is None:
                    events = await events_repo.find_current_test_events(suite_run_id)
                else:
                    events = await events_repo.find_by_suite_run(suite_run_id, after_id=last_event_id)

                for db_event in events:
                    yield ServerSentEvent(
                        event=db_event.type,
                        data=db_event.data
                    ).encode()
                    last_event_id = db_event.id
                    if db_event.type in [TestSuiteEventType.COMPLETE.value, TestSuiteEventType.ERROR.value]:
                        stop_event.set()
                        return

        except asyncio.CancelledError:
            stop_event.set()
        except Exception:
            logger.exception(f"Error streaming events for suite {suite_run_id}")
            stop_event.set()

    return StreamingResponse(
        with_heartbeat(event_generator()),
        media_type="text/event-stream"
    )


TEST_SUITE_RUN_RESULTS_PATH = f"{TEST_SUITE_RUN_PATH}/results"


@router.get(TEST_SUITE_RUN_RESULTS_PATH)
async def get_test_suite_run_results(agent_id: int, suite_run_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> List[TestCaseResult]:
    await _find_test_suite_run(suite_run_id, agent_id, user, db)
    return await TestCaseResultRepository(db).find_by_suite_run_id(suite_run_id)


TEST_SUITE_RUN_RESULT_MESSAGE_PATH = f"{TEST_SUITE_RUN_RESULTS_PATH}/{{result_id}}/messages"


@router.get(TEST_SUITE_RUN_RESULT_MESSAGE_PATH, response_model=List[ThreadMessagePublic])
async def get_test_suite_run_result_messages(
    agent_id: int,
    suite_run_id: int,
    result_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> List[ThreadMessage]:
    await _find_test_suite_run(suite_run_id, agent_id, user, db)

    results_repo = TestCaseResultRepository(db)
    result = await results_repo.find_by_id_and_suite_run_id(result_id, suite_run_id)

    if result is None or result.thread_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    return await ThreadMessageRepository(db).find_by_thread_id(result.thread_id)


TEST_CASE_PATH = f"{TEST_CASES_PATH}/{{test_case_id}}"


@router.get(TEST_CASE_PATH, response_model=PublicTestCase)
async def find_test_case(agent_id: int, test_case_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> TestCase:
    return await find_test_case_by_id(test_case_id, agent_id, user, db)


async def find_test_case_by_id(test_case_id: int, agent_id: int, user: User, db: AsyncSession) -> TestCase:
    await find_editable_agent(agent_id, user, db)
    test_case = await TestCaseRepository(db).find_by_id(test_case_id, agent_id)
    if test_case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return test_case


TEST_CASE_CLONE_PATH = f"{TEST_CASE_PATH}/clone"


@router.post(TEST_CASE_CLONE_PATH, response_model=PublicTestCase, status_code=status.HTTP_201_CREATED)
async def clone_test_case_endpoint(
    agent_id: int,
    test_case_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> TestCase:
    original_test_case = await find_test_case_by_id(test_case_id, agent_id, user, db)
    clone_name = _build_cloned_test_case_name(original_test_case.thread.name)
    return await clone_test_case(original_test_case, agent_id, user.id, db, test_case_name=clone_name)


def _build_cloned_test_case_name(original_name: Optional[str]) -> str:
    base_name = (original_name or "Test Case").strip()
    match = re.search(r"^(.*?)\s*\(copy(?: (\d+))?\)$", base_name)

    if match:
        name_prefix = match.group(1).strip()
        copy_number = int(match.group(2)) + 1 if match.group(2) else 2
        suffix = f"({CLONE_SUFFIX} {copy_number})"
    else:
        name_prefix = base_name
        suffix = f"({CLONE_SUFFIX})"

    truncated_prefix = name_prefix[:(MAX_THREAD_NAME_LENGTH - len(suffix) - 1)].rstrip()

    return f"{truncated_prefix} {suffix}".strip()


@router.put(TEST_CASE_PATH, response_model=PublicTestCase)
async def update_test_case(agent_id: int, test_case_id: int, update: UpdateTestCase,
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> TestCase:
    test_case = await find_test_case_by_id(test_case_id, agent_id, user, db)
    test_case.thread.name = update.name
    await ThreadRepository(db).update(test_case.thread)
    return test_case


@router.delete(TEST_CASE_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def delete_test_case(agent_id: int, test_case_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]):
    test_case = await find_test_case_by_id(test_case_id, agent_id, user, db)
    await TestCaseRepository(db).delete(test_case)


TEST_CASE_MESSAGES_PATH = f"{TEST_CASE_PATH}/messages"


@router.get(TEST_CASE_MESSAGES_PATH, response_model=List[ThreadMessagePublic])
async def find_messages(agent_id: int, test_case_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> List[ThreadMessage]:
    test_case = await find_test_case_by_id(test_case_id, agent_id, user, db)
    return await ThreadMessageRepository(db).find_by_thread_id(test_case.thread_id)


@router.post(TEST_CASE_MESSAGES_PATH, response_model=ThreadMessagePublic)
async def add_message(agent_id: int, test_case_id: int, message: NewTestCaseMessage, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> ThreadMessage:
    test_case = await find_test_case_by_id(test_case_id, agent_id, user, db)
    repo = ThreadMessageRepository(db)

    last_message = await repo.find_last_by_thread_id(test_case.thread_id)
    last_message_origin = last_message.origin if last_message else ThreadMessageOrigin.AGENT
    if not last_message_origin != message.origin:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    added_message = await repo.add(ThreadMessage(
        thread_id=test_case.thread_id,
        text=message.text,
        origin=message.origin,
    ))
    test_case.last_update = added_message.timestamp
    await TestCaseRepository(db).save(test_case)
    await _generate_test_case_name(test_case, user, db)
    return added_message


async def _generate_test_case_name(test_case: TestCase, user: User, db: AsyncSession) -> TestCase:
    if not test_case.is_default_name():
        return test_case

    messages = await ThreadMessageRepository(db).find_by_thread_id(test_case.thread_id)
    user_message = next((m for m in messages if m.origin == ThreadMessageOrigin.USER and m.text and m.text.strip()), None)
    agent_message = next((m for m in messages if m.origin == ThreadMessageOrigin.AGENT and m.text and m.text.strip()), None)
    if not user_message or not agent_message:
        return test_case

    agent = cast(Agent, await AgentRepository(db).find_by_id(test_case.agent_id))
    try:
        generated_name = await generate_test_case_name(agent, user_message.text.strip(), agent_message.text.strip(), user.id, db)
    except Exception:
        logger.warning("Failed to generate test case name for test case %s", test_case.thread_id, exc_info=True)
        return test_case

    trimmed_name = generated_name.strip()
    if not trimmed_name or trimmed_name == test_case.thread.name:
        return test_case
    test_case.thread.name = trimmed_name
    test_case.thread = await ThreadRepository(db).update(test_case.thread)
    return test_case


TEST_CASE_MESSAGE_PATH = f"{TEST_CASE_MESSAGES_PATH}/{{message_id}}"


@router.put(TEST_CASE_MESSAGE_PATH, response_model=ThreadMessagePublic)
async def update_message(agent_id: int, test_case_id: int, message_id: int, message: UpdateTestCaseMessage, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> ThreadMessage:
    test_case = await find_test_case_by_id(test_case_id, agent_id, user, db)
    repo = ThreadMessageRepository(db)
    test_case_message = await repo.find_by_id(message_id)
    if test_case_message is None or test_case_message.thread_id != test_case.thread_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    test_case_message.text = message.text
    await repo.update(test_case_message)
    test_case.last_update = datetime.now(timezone.utc)
    await TestCaseRepository(db).save(test_case)
    return test_case_message


@router.delete(TEST_CASE_MESSAGE_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def delete_message_and_following(agent_id: int, test_case_id: int, message_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]):
    test_case = await find_test_case_by_id(test_case_id, agent_id, user, db)
    repo = ThreadMessageRepository(db)
    test_case_message = await repo.find_by_id(message_id)
    if test_case_message is None or test_case_message.thread_id != test_case.thread_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    await repo.delete_from_date(test_case.thread_id, test_case_message.timestamp)
    test_case.last_update = datetime.now(timezone.utc)
    await TestCaseRepository(db).save(test_case)
