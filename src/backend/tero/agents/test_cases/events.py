import asyncio
from dataclasses import dataclass
import json
import logging
from typing import AsyncIterator, Callable, Dict, Any, Optional, TypeVar

import psycopg
from psycopg import sql
from sqlmodel.ext.asyncio.session import AsyncSession

from ...core import repos
from .domain import TestSuiteRunStatus
from .repos import TestSuiteRunRepository


logger = logging.getLogger(__name__)
T = TypeVar('T')


@dataclass
class SuiteRunEventNotification:
    suite_run_id: int
    event_id: int


@dataclass
class SuiteStatusChange:
    suite_run_id: int
    status: TestSuiteRunStatus


async def _listen_to_channel(
    channel: str,
    stop_event: asyncio.Event,
    transform: Callable[[Dict[str, Any]], T],
    timeout
) -> AsyncIterator[Optional[T]]:
    db_url = repos.plain_postgresql_url()

    try:
        async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
            await conn.execute(sql.SQL("LISTEN {}").format(sql.Identifier(channel)))

            yield None  # Signal that connection is established

            while not stop_event.is_set():
                try:
                    async for notify in conn.notifies(timeout=timeout):
                        yield transform(json.loads(notify.payload))
                except asyncio.TimeoutError:
                    # Timeout allows checking stop_event periodically
                    continue
                except StopAsyncIteration:
                    break
    except Exception:
        logger.warning(f"Error in LISTEN/NOTIFY loop for {channel}", exc_info=True)


async def listen_for_suite_run_events(
    suite_run_id: int,
    stop_event: asyncio.Event,
) -> AsyncIterator[Optional[SuiteRunEventNotification]]:
    def transform(payload: Dict[str, Any]) -> SuiteRunEventNotification:
        return SuiteRunEventNotification(
            suite_run_id=payload["suite_run_id"],
            event_id=payload["event_id"]
        )

    async for event in _listen_to_channel("test_suite_events", stop_event, transform, 1):
        if event is None or event.suite_run_id == suite_run_id:
            yield event


async def listen_for_suite_run_status_changes(
    suite_run_id: int,
    stop_event: asyncio.Event,
) -> AsyncIterator[Optional[SuiteStatusChange]]:
    def transform(payload: Dict[str, Any]) -> SuiteStatusChange:
        return SuiteStatusChange(
            suite_run_id=payload["suite_run_id"],
            status=TestSuiteRunStatus(payload["status"])
        )

    async for status_change in _listen_to_channel("test_suite_status", stop_event, transform, 1):
        if status_change is None or status_change.suite_run_id == suite_run_id:
            yield status_change


async def monitor_cancellation(suite_run_id: int, agent_id: int, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            async with AsyncSession(repos.engine) as session:
                suite_run = await TestSuiteRunRepository(session).find_by_id_and_agent_id(suite_run_id, agent_id)
                if suite_run and suite_run.status == TestSuiteRunStatus.CANCELLING:
                    stop_event.set()
                    return
                elif not suite_run or suite_run.status != TestSuiteRunStatus.RUNNING:
                    return

            async for status_change in listen_for_suite_run_status_changes(suite_run_id, stop_event):
                if status_change is None:
                    continue
                elif status_change.status == TestSuiteRunStatus.CANCELLING:
                    stop_event.set()
                    return
                elif status_change.status != TestSuiteRunStatus.RUNNING:
                    return
        except Exception:
            # DB or LISTEN/NOTIFY connection may fail; wait before retrying to avoid tight loops
            logger.exception(f"Error monitoring cancellation for suite {suite_run_id}")
            await asyncio.sleep(2)
