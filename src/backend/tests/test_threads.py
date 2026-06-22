from datetime import timezone
import re
import threading
import time
from typing import Any, Callable, Optional, cast

from sqlalchemy import select
from sse_starlette import ServerSentEvent

from .common import *
from .common import find_last_message_id_for_thread

from tero.agents.domain import AgentListItem
from tero.files.domain import FileMetadata, FileProcessor, FileMetadataWithContent
from tero.threads.api import THREADS_PATH, THREAD_PATH, THREAD_MESSAGES_PATH, THREAD_MESSAGE_PATH, THREAD_FILE_PATH
from tero.threads.domain import MAX_THREAD_NAME_LENGTH, ThreadListItem, ThreadMessageOrigin, ThreadMessagePublic
from tero.tools.core import AgentActionEvent, AgentAction
from tero.usage.domain import Usage, UsageType


LAST_FILE_ID = 1


@pytest.fixture(name="threads")
def threads_fixture(agents: List[AgentListItem]) -> List[ThreadListItem]:
    agents = _remove_active_users_from_agents(agents)
    return [
        ThreadListItem(id=THREAD_ID, agent_id=AGENT_ID, name="Thread 1", user_id=USER_ID,
                      agent=agents[0], creation=PAST_TIME, last_message=parse_date("2025-02-21T12:01:00")),
        ThreadListItem(id=OTHER_THREAD_ID, agent_id=2, name="Thread 2", user_id=USER_ID,
                      agent=agents[1], creation=parse_date("2025-02-21T12:01:00"), last_message=parse_date("2025-02-21T12:03:00")),
    ]


@pytest.fixture(name="messages")
def messages_fixture() -> List[ThreadMessagePublic]:
    return [
        ThreadMessagePublic(id=1, thread_id=THREAD_ID, origin=ThreadMessageOrigin.USER, text="This is a message", timestamp=PAST_TIME, minutes_saved=5,
            children=[ThreadMessagePublic(id=2, thread_id=THREAD_ID, origin=ThreadMessageOrigin.AGENT, text="This is a response",
                            timestamp=parse_date("2025-02-21T12:01:00"), minutes_saved=30, parent_id=1)])
    ]


def _remove_active_users_from_agents(agents: List[AgentListItem]):
    for a in agents:
        a.active_users = None
    return agents


async def test_find_all_threads(client: AsyncClient, threads: List[ThreadListItem]):
    resp = await _find_threads(client)
    assert_response(resp, [threads[1], threads[0]])


async def _find_threads(client: AsyncClient, params: Optional[dict[str, Any]] = None) -> Response:
    return await client.get(THREADS_PATH, params=params if params else {})


async def test_find_all_threads_limit(client: AsyncClient, threads: List[ThreadListItem]):
    resp = await _find_threads(client, {"limit": 1})
    assert_response(resp, [threads[1]])


async def test_find_threads_by_text(client: AsyncClient, threads: List[ThreadListItem]):
    resp = await _find_threads(client, {"text": "another"})
    threads[1].creation = parse_date('2025-02-21T12:01:00')
    threads[1].last_message = parse_date('2025-02-21T12:02:00')
    assert_response(resp, [threads[1]])

async def test_find_threads_by_agent_id(client: AsyncClient, threads: List[ThreadListItem]):
    resp = await _find_threads(client, {"agent_id": AGENT_ID})
    assert_response(resp, [threads[0]])


async def test_find_threads_excluding_empty(client: AsyncClient, override_user, agents: List[AgentListItem]):
    override_user(OTHER_USER_ID)
    agents = _remove_active_users_from_agents(agents)
    agents[1].can_edit = False
    resp = await _find_threads(client, {"exclude_empty": True})
    assert_response(resp, [
        ThreadListItem(id=6, agent_id=5, name="Thread 6", user_id=OTHER_USER_ID,
                      agent=agents[3], creation=parse_date("2025-02-21T12:05:00"),
                      last_message=parse_date("2025-02-21T12:09:00")),
        ThreadListItem(id=4, agent_id=2, name="Thread 4", user_id=OTHER_USER_ID,
                      agent=agents[1], creation=parse_date("2025-02-21T12:03:00"),
                      last_message=parse_date("2025-01-20T12:05:00"))
    ])


@freeze_time(CURRENT_TIME)
async def test_find_created_thread(threads: List[ThreadListItem], agents: List[AgentListItem], last_thread_id: int, client: AsyncClient):
    agents = _remove_active_users_from_agents(agents)
    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    assert resp.status_code == status.HTTP_201_CREATED
    resp = await _find_threads(client)
    new_thread = ThreadListItem(
        id=last_thread_id + 1, agent_id=AGENT_ID, name=f"Chat #{last_thread_id + 1}", user_id=USER_ID, agent=agents[0],
        creation=CURRENT_TIME, last_message=None)
    assert_response(resp, [new_thread, threads[1], threads[0]])


async def test_create_thread_in_non_visible_agent(client: AsyncClient):
    resp = await create_thread(NON_VISIBLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST


async def test_reuse_empty_thread_when_creating_thread(client: AsyncClient):
    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    empty_thread_id = resp.json().get("id")
    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    assert resp.json().get("id") == empty_thread_id


async def test_find_thread(client: AsyncClient, threads: List[ThreadListItem]):
    resp = await _find_thread(THREAD_ID, client)
    threads[0].creation = PAST_TIME
    threads[0].last_message = None
    assert_response(resp, threads[0])


async def _find_thread(thread_id: int, client: AsyncClient) -> Response:
    return await client.get(THREAD_PATH.format(thread_id=thread_id))


async def test_find_thread_from_other_user(client: AsyncClient):
    resp = await _find_thread(OTHER_USER_THREAD_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_thread(client: AsyncClient, threads: List[ThreadListItem], agents: List[AgentListItem]):
    name = "Updated name"
    await _update_thread(THREAD_ID, {"name": name}, client)
    resp = await _find_threads(client)
    updated_thread = threads[0]
    updated_thread.name = name
    assert_response(resp, [threads[1], updated_thread])


async def _update_thread(thread_id: int, body: dict[str, Any], client: AsyncClient) -> Response:
    return await client.put(THREAD_PATH.format(thread_id=thread_id), json=body)


async def test_delete_thread(threads:List[ThreadListItem], client: AsyncClient):
    resp = await _delete_thread(THREAD_ID, client)
    resp.raise_for_status()
    resp = await _find_threads(client)
    assert_response(resp, [threads[1]])


async def _delete_thread(thread_id: int, client: AsyncClient) -> Response:
    return await client.delete(THREAD_PATH.format(thread_id=thread_id))


async def test_delete_deleted_thread(client: AsyncClient):
    resp = await _delete_thread(THREAD_ID, client)
    resp.raise_for_status()
    resp = await _delete_thread(THREAD_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@freeze_time(CURRENT_TIME)
async def test_add_thread_message(last_message_id: int, client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID,
                                     "Which is the first natural number? Only provide the number. Don't ask follow up questions", parent_message_id=parent_message_id) as resp:
        await _assert_response(resp, "1", last_message_id + 1)


async def _assert_response(
    response: Response,
    expected_text: str,
    user_message_id: int,
    metadata_minutes_saved: Optional[int] = None,
    message_updated_minutes_saved: Optional[int] = 0,
    stopped: bool = False,
    send_pre_model_status: bool = True,
    status_updates: List[AgentActionEvent] = [],
    ignore_status_updates: bool = False,
    user_files: List[FileMetadata] = [],
):
    buffer, events = [], []
    separator = "\r\n\r\n"

    def flush_buffer():
        if buffer:
            events.append(f"data: {''.join(buffer)}{separator}".encode())
            buffer.clear()

    async for chunk in response.aiter_bytes():
        decoded_chunk = chunk.decode()
        for event in decoded_chunk.split(separator):
            if event.startswith("data: "):
                buffer.append(event[6:])
            elif event.startswith("event: heartbeat"):
                continue
            else:
                flush_buffer()
                if event.startswith("event: userMessage"):
                    event = re.sub(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z', r'\1', event)
                if event: events.append(f"{event}{separator}".encode())

    flush_buffer()

    if ignore_status_updates:
        events = [e for e in events if not e.startswith(b"event: status")]
    if message_updated_minutes_saved is None:
        events = [e for e in events if not e.startswith(b"event: messageUpdated")]
    events = [e for e in events if not e.startswith(b"event: threadUpdated")]

    expected_events = [
        ServerSentEvent(
            event="userMessage",
            data=json.dumps({
                "id": user_message_id,
                "files": [f.model_dump(mode="json", by_alias=True) for f in user_files]
            })
        ).encode()]
    if not ignore_status_updates:
        if send_pre_model_status:
            expected_events.append(
                ServerSentEvent(
                    event="status",
                    data=str(json.dumps(AgentActionEvent(action=AgentAction.PRE_MODEL_HOOK).model_dump(mode="json", by_alias=True)))
                ).encode())
        for status_update in status_updates:
            expected_events.append(
                ServerSentEvent(
                    event="status",
                    data=str(json.dumps(status_update.model_dump(mode="json", by_alias=True)))
                ).encode())
    if expected_text:
        expected_events.append(ServerSentEvent(data=expected_text).encode())
    answer_message_id = user_message_id + 1
    expected_events.append(
        ServerSentEvent(
            event="metadata",
            data=str(json.dumps({
                "answerMessageId": answer_message_id,
                "files": [],
                "minutesSaved": metadata_minutes_saved,
                "stopped": stopped
            }))).encode())
    if message_updated_minutes_saved is not None:
        expected_events.append(
            ServerSentEvent(
                event="messageUpdated",
                data=str(json.dumps({
                    "messageId": answer_message_id,
                    "minutesSaved": message_updated_minutes_saved,
                }))).encode())
    assert events == expected_events


@freeze_time(CURRENT_TIME)
async def test_add_thread_message_with_deep_agent(last_message_id: int, client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(OTHER_THREAD_ID, session)
    async with add_message_to_thread(client, OTHER_THREAD_ID,
                                     "Which is the first natural number? Only provide the number", parent_message_id=parent_message_id) as resp:
        await _assert_response(resp, "1", last_message_id + 1)


@freeze_time(CURRENT_TIME)
async def test_add_thread_message_stopped_response(last_message_id: int, client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)

    def stop_with_delay_thread(loop, async_client):
        time.sleep(1.0)
        async def send_stop_request():
            await async_client.post(THREAD_PATH.format(thread_id=THREAD_ID) + "/stop")
        asyncio.run_coroutine_threadsafe(send_stop_request(), loop)

    # Start the stop request in a separate thread due to the add message stream response blocking the event loop
    stop_thread = threading.Thread(target=stop_with_delay_thread, args=(asyncio.get_event_loop(), client))
    stop_thread.start()

    try:
        async with add_message_to_thread(client, THREAD_ID, "Escribe un cuento de no menos de 500 palabras", parent_message_id=parent_message_id) as resp:
            # Check metadata event is stopped, ignoring other events
            response_text = await resp.aread()
            metadata_match = re.search(r'event: metadata\r\ndata: ({.*?})\r\n', response_text.decode())
            assert metadata_match and json.loads(metadata_match.group(1)) == {
                "answerMessageId": last_message_id + 2,
                "files": [],
                "minutesSaved": 0,
                "stopped": True
            }
    finally:
        stop_thread.join(timeout=2)


@freeze_time(CURRENT_TIME)
async def test_add_thread_message_with_reasoning_model(last_message_id: int, client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID,
                                     "Which is the first natural number? Only provide the number", parent_message_id=parent_message_id) as resp:
        await _assert_response(resp, "1", last_message_id + 1)


@freeze_time(CURRENT_TIME)
async def test_add_thread_message_with_edition_message(last_message_id: int, client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID,
                                     "Which is the first natural number? Only provide the number", parent_message_id=parent_message_id, isInAgentEdition=True) as resp:
        await _assert_response(resp, "1", last_message_id + 1, metadata_minutes_saved=0, message_updated_minutes_saved=None)


async def test_add_message_in_invalid_thread(client: AsyncClient):
    async with add_message_to_thread(client, OTHER_USER_THREAD_ID, "Hello") as resp:
        assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_add_message_with_parent_id_from_another_thread(client: AsyncClient):
    async with add_message_to_thread(client, THREAD_ID, "Hello", parent_message_id=3) as resp:
        assert resp.status_code == status.HTTP_400_BAD_REQUEST


@freeze_time(CURRENT_TIME)
@pytest.mark.no_stub_estimate_minutes_saved
async def test_thread_message_add_positive_feedback(last_message_id: int, client: AsyncClient, messages: List[ThreadMessagePublic], session: AsyncSession):
    message_text = "Which is the first natural number? Only provide the number. Don't ask follow up questions"
    minutes_saved = 30
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID, message_text, parent_message_id=parent_message_id) as resp:
        await _assert_response(resp, "1", last_message_id + 1, message_updated_minutes_saved=None)
    await _update_thread_message(client, THREAD_ID, last_message_id + 2, {"hasPositiveFeedback": True, "minutesSaved": minutes_saved})
    resp = await _find_thread_messages(THREAD_ID, client)
    messages[0].children[0].children = _build_thread_messages_response(THREAD_ID, last_message_id + 1, parent_message_id, message_text, "1", CURRENT_TIME, True, minutes_saved)
    assert_response(resp, [messages[0]])


@pytest.mark.no_stub_estimate_minutes_saved
async def test_thread_message_minutes_saved_estimation_with_positive_feedback(last_message_id: int, client: AsyncClient, session: AsyncSession):
    minutes_saved = 8
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID, "Which is 1 + 1? Only provide the number", parent_message_id=parent_message_id) as resp:
        await _assert_response(resp, "2", last_message_id + 1, message_updated_minutes_saved=None)
    await _update_thread_message(client, THREAD_ID, last_message_id + 2, {"hasPositiveFeedback": True, "minutesSaved": minutes_saved})
    async with add_message_to_thread(client, THREAD_ID, "Which is 2 + 2? Only provide the number", parent_message_id=last_message_id + 2) as resp:
        await _assert_response(resp, "4", last_message_id + 3, message_updated_minutes_saved=minutes_saved)


async def _update_thread_message(client: AsyncClient, thread_id: int, message_id: int, body: dict[str, Any]) -> Response:
    return await client.put(THREAD_MESSAGE_PATH.format(thread_id=thread_id, message_id=message_id), json=body)


def _build_thread_messages_response(thread_id: int, message_id: int, parent_message_id: Optional[int], message_text: str, response_text: str, timestamp: datetime, has_positive_feedback: Optional[bool], minutes_saved: Optional[int], feedback_text: Optional[str] = None) -> List[ThreadMessagePublic]:
    return [
        ThreadMessagePublic(id=message_id, text=message_text, thread_id=thread_id,
            origin=ThreadMessageOrigin.USER,
            timestamp=timestamp,
            parent_id=parent_message_id,
            status_updates=None,
            children=[ThreadMessagePublic(id=message_id + 1, text=response_text, thread_id=thread_id,
                    origin=ThreadMessageOrigin.AGENT,
                    has_positive_feedback=has_positive_feedback,
                    timestamp=timestamp,
                    parent_id=message_id, minutes_saved=minutes_saved, feedback_text=feedback_text, status_updates=[AgentActionEvent(action=AgentAction.PRE_MODEL_HOOK)])
            ])
    ]


@freeze_time(CURRENT_TIME)
@pytest.mark.no_stub_estimate_minutes_saved
async def test_thread_message_add_negative_feedback(last_message_id: int, client: AsyncClient, messages: List[ThreadMessagePublic], session: AsyncSession):
    message_text = "Which is the first natural number? Only provide the number"
    negative_feedback_text = "This is a negative feedback"
    minutes_saved = -15
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID, message_text, parent_message_id=parent_message_id) as resp:
        await _assert_response(resp, "1", last_message_id + 1, message_updated_minutes_saved=None)
    await _update_thread_message(client, THREAD_ID, last_message_id + 2, {"hasPositiveFeedback": False, "feedbackText": negative_feedback_text, "minutesSaved": minutes_saved})
    resp = await _find_thread_messages(THREAD_ID, client)
    messages[0].children[0].children = _build_thread_messages_response(THREAD_ID, last_message_id + 1, parent_message_id, message_text, "1", CURRENT_TIME, False, minutes_saved, negative_feedback_text)
    assert_response(resp, [messages[0]])


@freeze_time(CURRENT_TIME)
async def test_thread_message_remove_feedback(last_message_id: int, client: AsyncClient, messages: List[ThreadMessagePublic], session: AsyncSession):
    message_text = "Which is the first natural number? Only provide the number"
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID, message_text, parent_message_id=parent_message_id) as resp:
        await _assert_response(resp, "1", last_message_id + 1)
    await _update_thread_message(client, THREAD_ID, last_message_id + 2, {"hasPositiveFeedback": True, "minutesSaved": 30})
    await _update_thread_message(client, THREAD_ID, last_message_id + 2, {})
    resp = await _find_thread_messages(THREAD_ID, client)
    messages[0].children[0].children = _build_thread_messages_response(THREAD_ID, last_message_id + 1, parent_message_id, message_text, "1", CURRENT_TIME, None, 0)
    assert_response(resp, [messages[0]])


async def test_add_message_over_monthly_limit(client: AsyncClient, session: AsyncSession):
    # consuming the quota
    session.add(Usage(message_id=1, user_id=USER_ID, agent_id=AGENT_ID, model_id="gpt-5-mini",
                      timestamp=datetime.now(timezone.utc), quantity=1000, usd_cost=5.0, type=UsageType.PROMPT_TOKENS))
    session.add(Usage(message_id=1, user_id=USER_ID, agent_id=AGENT_ID, model_id="gpt-5-mini",
                      timestamp=datetime.now(timezone.utc), quantity=1000, usd_cost=5.0, type=UsageType.COMPLETION_TOKENS))
    await session.commit()
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID, "Hello", parent_message_id=parent_message_id) as resp:
        assert resp.status_code == status.HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.no_stub_build_thread_name
async def test_thread_name_after_first_message(client: AsyncClient):
    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    thread_id = resp.json().get("id")
    async with add_message_to_thread(client, thread_id, "Hello") as stream_resp:
        stream_resp.raise_for_status()
        name = await _thread_name_from_sse_response(stream_resp)
    assert name != f"Chat #{thread_id}"
    assert 0 < len(name) <= MAX_THREAD_NAME_LENGTH


async def _thread_name_from_sse_response(response: Response) -> str:
    separator = "\r\n\r\n"
    async for chunk in response.aiter_bytes():
        for event in chunk.decode().split(separator):
            if not event.startswith("event: threadUpdated"):
                continue
            for line in event.split("\r\n"):
                if line.startswith("data: "):
                    return json.loads(line[6:])["name"]
    pytest.fail("No threadUpdated event in response")


async def _add_message_to_thread(thread_id: int, message_text:str, client: AsyncClient,
                                 parent_message_id:Optional[int]=None, files:List[str]=[]) -> Optional[dict]:
    user_message_data = None
    async with add_message_to_thread(client, thread_id, message_text, parent_message_id, files) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_text():
            if "event: error" in chunk:
                raise AssertionError(f"Error event received: {chunk}")
            for event in chunk.split("\r\n\r\n"):
                if "event: userMessage" in event:
                    for line in event.split("\r\n"):
                        if line.startswith("data: "):
                            user_message_data = json.loads(line[6:])
                            break
    return user_message_data


async def test_find_thread_messages(messages: List[ThreadMessagePublic], client: AsyncClient):
    resp = await _find_thread_messages(THREAD_ID, client)
    assert_response(resp, [*messages])


async def _find_thread_messages(thread_id: int, client: AsyncClient) -> Response:
    return await client.get(THREAD_MESSAGES_PATH.format(thread_id=thread_id))


@freeze_time(CURRENT_TIME)
async def test_edit_thread_message(last_message_id: int, messages: List[ThreadMessagePublic], client: AsyncClient):
    message_text = "Which is the first natural number? Only provide the number"
    parent_id = 1
    await _add_message_to_thread(THREAD_ID, message_text, client, parent_id)
    res = await _find_thread_messages(THREAD_ID, client)
    messages[0].children += _build_thread_messages_response(THREAD_ID, last_message_id + 1, parent_id, message_text, "1", CURRENT_TIME, None, 0)
    assert_response(res, [*messages])


async def test_find_messages_from_invalid_thread(client: AsyncClient):
    resp = await _find_thread_messages(OTHER_USER_THREAD_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.parametrize("file_name,content_type,user_message,agent_response", [
    ("number-10.png", "image/png", "What number is shown in this image?. Output just the number", "10"),
    pytest.param("Emma's routine.pdf", "application/pdf", "What time does Emma wake up according to the document? Output only the time in H:MM format. Don't use clock tool.", "7:35", marks=pytest.mark.skipif(not env.azure_doc_intelligence_key, reason="Azure Doc Intelligence not configured")),
    ("sample.txt", "text/plain; charset=ascii", "Output only the exact content of the uploaded file", "Sample test"),
    ("users.csv", "text/csv; charset=ascii", "What is the first name of the first user? Output only the name", "Rachel"),
    ("sheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "What is the 'lastname' of Dulce? Output only the lastname", "Abril"),
    ("sheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "What is the 'country' of Dulce? Output only the country name", "España"),
    ("sheet.xls", "application/vnd.ms-excel", "What is the 'lastname' of Dulce? Output only the lastname", "Abril"),
    ("sheet.xls", "application/vnd.ms-excel", "What is the 'country' of Dulce? Output only the country name", "España"),
    ])
@freeze_time(CURRENT_TIME)
async def test_add_thread_message_with_attachment(last_message_id: int, client: AsyncClient, file_name: str, content_type: str, user_message: str, agent_response: str, session: AsyncSession):
    file_path = solve_asset_path(file_name, __file__)
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID, user_message, parent_message_id=parent_message_id, files=[file_path]) as resp:
        await _assert_response(resp, agent_response, last_message_id + 1,
            ignore_status_updates=True,
            user_files=[FileMetadata(id=LAST_FILE_ID + 1, name=file_name, content_type=content_type, user_id=USER_ID, timestamp=CURRENT_TIME, status=FileStatus.PROCESSED, file_processor=FileProcessor.ENHANCED)])


@freeze_time(CURRENT_TIME)
async def test_add_thread_message_with_existing_file_attachment(last_message_id: int, client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    file_id = await _add_thread_file(THREAD_ID, client, parent_message_id=parent_message_id)
    async with add_message_to_thread(client, THREAD_ID, "Output only the exact content of the uploaded file",
                                        parent_message_id=last_message_id + 2, file_ids=[file_id]) as resp:
        # we add 3 to message id since _add_thread_file adds 2 messages + 1 message that is the user message that generated this answer
        await _assert_response(resp, "Sample test", last_message_id + 3,
            ignore_status_updates=True,
            user_files=[_build_sample_txt_thread_message_file(file_id)])


def _build_sample_txt_thread_message_file(file_id: int) -> FileMetadata:
    return FileMetadata(id=file_id,
        name="sample.txt",
        content_type="text/plain; charset=ascii",
        user_id=USER_ID,
        timestamp=CURRENT_TIME,
        status=FileStatus.PROCESSED,
        file_processor=FileProcessor.ENHANCED)


async def test_add_thread_message_with_file_id_from_another_thread(client: AsyncClient, session: AsyncSession):
    parent_message_id_other = await find_last_message_id_for_thread(OTHER_THREAD_ID, session)
    file_id = await _add_thread_file(OTHER_THREAD_ID, client, parent_message_id=parent_message_id_other)
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    async with add_message_to_thread(client, THREAD_ID, "Output only the exact content of the uploaded file",
                                        parent_message_id=parent_message_id, file_ids=[file_id]) as resp:
        assert resp.status_code == status.HTTP_400_BAD_REQUEST


async def _add_thread_file(thread_id: int, client: AsyncClient, parent_message_id: Optional[int] = None) -> int:
    txt_path = os.path.join(os.path.dirname(__file__), "assets", "sample.txt")
    user_message = await _add_message_to_thread(thread_id, "Hello, remember this file", client, parent_message_id, files=[txt_path])
    return cast(dict, user_message)["files"][0]["id"]


@pytest.mark.parametrize("model_id", ["claude-sonnet-4", "gemini-2.5-flash"])
async def test_thread_with_model(model_id: str, last_message_id: int, client: AsyncClient, session: AsyncSession):
    resp = await client.post(f"{BASE_PATH}/agents")
    resp.raise_for_status()
    agent_id = resp.json()["id"]

    update_resp = await client.put(f"{BASE_PATH}/agents/{agent_id}", json={
        "modelId": model_id,
        "systemPrompt": "You are a helpful AI assistant. Answer questions clearly and concisely.",
        "temperature": "PRECISE",
        "reasoningEffort": "LOW"
    })
    update_resp.raise_for_status()

    thread_resp = await create_thread(agent_id, client)
    thread_resp.raise_for_status()
    thread_id = thread_resp.json()["id"]

    async with add_message_to_thread(client, thread_id, "What is 2 + 2? Only provide the number.") as resp:
        await _assert_response(resp, "4", last_message_id + 1)

    usages = await session.execute(select(Usage).where(Usage.agent_id == agent_id))
    for usage in usages.scalars().all():
        assert usage.model_id == model_id
        assert usage.quantity > 0
        assert usage.usd_cost > 0


@freeze_time(CURRENT_TIME)
async def test_find_thread_message_file(client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    file_id = await _add_thread_file(THREAD_ID, client, parent_message_id=parent_message_id)
    resp = await _find_thread_file(THREAD_ID, file_id, client)
    resp.raise_for_status()
    file_metadata = _build_sample_txt_thread_message_file(file_id)
    assert_response(resp, FileMetadataWithContent(**{**file_metadata.model_dump(), "processed_content": "Sample test"}))


async def _find_thread_file(thread_id: int, file_id: int, client: AsyncClient) -> Response:
    return await client.get(THREAD_FILE_PATH.format(thread_id=THREAD_ID, file_id=file_id))


async def test_find_thread_message_file_from_another_user_thread(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(OTHER_USER_ID)
    file_id = await _add_thread_file(OTHER_USER_THREAD_ID, client)
    override_user(USER_ID)
    resp = await _find_thread_file(OTHER_USER_THREAD_ID, file_id, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_find_thread_message_file_from_another_thread(client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(OTHER_THREAD_ID, session)
    file_id = await _add_thread_file(OTHER_THREAD_ID, client, parent_message_id=parent_message_id)
    resp = await _find_thread_file(THREAD_ID, file_id, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_download_thread_file(client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(THREAD_ID, session)
    file_id = await _add_thread_file(THREAD_ID, client, parent_message_id=parent_message_id)
    resp = await _download_thread_file(THREAD_ID, file_id, client)
    resp.raise_for_status()
    assert resp.content == b"Sample test"


async def _download_thread_file(thread_id: int, file_id: int, client: AsyncClient) -> Response:
    return await client.get(f"{THREAD_FILE_PATH.format(thread_id=THREAD_ID, file_id=file_id)}/content")


async def test_download_thread_file_from_another_user_thread(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(OTHER_USER_ID)
    file_id = await _add_thread_file(OTHER_USER_THREAD_ID, client)
    override_user(USER_ID)
    resp = await _download_thread_file(OTHER_USER_THREAD_ID, file_id, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_download_thread_file_from_another_thread(client: AsyncClient, session: AsyncSession):
    parent_message_id = await find_last_message_id_for_thread(OTHER_THREAD_ID, session)
    file_id = await _add_thread_file(OTHER_THREAD_ID, client, parent_message_id=parent_message_id)
    resp = await _download_thread_file(THREAD_ID, file_id, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
