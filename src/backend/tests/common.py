import asyncio
from datetime import datetime
import json
import logging
import os
from typing import List, Sequence, AsyncContextManager, Optional
from unittest.mock import AsyncMock, patch

import aiofiles
from fastapi import status # noqa: F401  # used by test files importing common
from freezegun import freeze_time # noqa: F401  # used by test files importing common
from httpx import Response, AsyncClient
from pydantic import BaseModel
import pytest
from sqlalchemy.orm import Mapped
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession

from tero.agents.api import AGENT_TOOLS_PATH, AGENT_TOOL_FILES_PATH
from tero.core.api import BASE_PATH # noqa: F401  # used by test files importing common
from tero.core.assets import solve_asset_path
from tero.core.env import env # noqa: F401  # used by test files importing common
from tero.files.domain import FileStatus
from tero.threads.api import THREAD_MESSAGES_PATH, THREADS_PATH, ThreadCreateApi
from tero.threads.domain import ThreadMessage


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.asyncio
PAST_TIME = parse_date("2025-02-21T12:00:00")
CURRENT_TIME = parse_date("2025-02-22T12:00:00")
USER_ID = 1
OTHER_USER_ID = 2
AGENT_ID = 1
NON_EDITABLE_AGENT_ID = 3
NON_VISIBLE_AGENT_ID = 4
THREAD_ID = 1
OTHER_THREAD_ID = 2
OTHER_USER_THREAD_ID = 3
GLOBAL_TEAM_ID = 1
TEST_ICON = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjWCMg9B8ABAgBzkPo1OYAAAAASUVORK5CYII="


@pytest.fixture
def stub_docs_tool_generate_description():
    with patch("tero.tools.docs.tool.DocsTool._generate_description", new=AsyncMock(return_value="stub description")):
        yield


def assert_response(resp: Response, expected: Sequence[BaseModel] | BaseModel):
    resp.raise_for_status()
    assert resp.json() == (
        [json.loads(e.model_dump_json(by_alias=True)) for e in expected] if isinstance(expected, Sequence) else json.loads(
            expected.model_dump_json(by_alias=True)))


async def configure_agent_tool(agent_id: int, tool_id: str, config: dict, client: AsyncClient) -> Response:
    resp = await try_configure_agent_tool(agent_id, tool_id, config, client)
    resp.raise_for_status()
    return resp


async def try_configure_agent_tool(agent_id: int, tool_id: str, config: dict, client: AsyncClient) -> Response:
    return await client.post(AGENT_TOOLS_PATH.format(agent_id=agent_id), json={"toolId": tool_id, "config": config})


async def upload_agent_tool_config_file(agent_id: int, tool_id: str, client: AsyncClient, filename: str = "test.txt",
        content: bytes = b"Hello") -> int:
    resp = await try_upload_agent_tool_config_file(agent_id, tool_id, client, filename, content)
    resp.raise_for_status()
    return resp.json()["id"]


async def try_upload_agent_tool_config_file(agent_id: int, tool_id: str, client: AsyncClient, filename: str = "test.txt",
        content: bytes = b"Hello") -> Response:
    return await client.post(AGENT_TOOL_FILES_PATH.format(agent_id=agent_id, tool_id=tool_id),
                            files={"file": (filename, content)})


async def await_files_processed(agent_id: int, tool_id: str, file_id: int, client: AsyncClient) -> Response:
    timeout_seconds = 10
    start_time = asyncio.get_event_loop().time()
    while True:
        resp = await find_agent_tool_config_files(agent_id, tool_id, client)
        resp.raise_for_status()
        files = resp.json()
        if all(file["status"] != FileStatus.PENDING.value for file in files):
            return resp
        if asyncio.get_event_loop().time() - start_time >= timeout_seconds:
            raise TimeoutError(f"File {file_id} processing timed out after {timeout_seconds} seconds")
        await asyncio.sleep(1)


async def find_agent_tool_config_files(agent_id: int, tool_id: str, client: AsyncClient) -> Response:
    return await client.get(AGENT_TOOL_FILES_PATH.format(agent_id=agent_id, tool_id=tool_id))


async def create_thread(agent_id: int, client: AsyncClient) -> Response:
    return await client.post(THREADS_PATH, json=ThreadCreateApi(agent_id=agent_id).model_dump())


def add_message_to_thread(client: AsyncClient, thread_id: int, message: str, parent_message_id:Optional[int] = None, files:List[str] = [], file_ids:List[int] = [], isInAgentEdition:Optional[bool] = False) -> AsyncContextManager[Response]:
    form_data = {"text": message, "origin": "USER"}
    if parent_message_id is not None:
        form_data["parentMessageId"] = str(parent_message_id)
    if isInAgentEdition is not None:
        form_data["isInAgentEdition"] = str(isInAgentEdition)
    if file_ids:
        form_data["fileIds"] = ",".join(str(file_id) for file_id in file_ids)
    file_data = []
    if files:
        for file_path in files:
            with open(file_path, 'rb') as f:
                file_data.append(("files", (os.path.basename(file_path), f.read())))

    return client.stream("POST", THREAD_MESSAGES_PATH.format(thread_id=thread_id), data=form_data, files=file_data)


async def find_asset_text(filename: str) -> str:
    async with aiofiles.open(solve_asset_path(filename, __file__), 'r') as file:
        return await file.read()


async def find_asset_bytes(filename: str) -> bytes:
    async with aiofiles.open(solve_asset_path(filename, __file__), 'rb') as file:
        return await file.read()


async def find_last_id(column: Mapped[int], db: AsyncSession) -> int:
    result = await db.exec(select(func.max(column)))
    return result.one()


async def find_last_message_id_for_thread(thread_id: int, db: AsyncSession) -> int:
    """
    Find the last message id in a thread, assuming it has only one branch.
    This is useful for tests that need to add messages to a thread.
    """
    result = await db.exec(
        select(func.max(ThreadMessage.id))
        .where(ThreadMessage.thread_id == thread_id)
    )
    return result.one()
