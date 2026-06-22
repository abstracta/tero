import asyncio
from contextlib import AsyncExitStack
from enum import Enum
import io
import json
import logging
from typing import Annotated, Optional, List, AsyncIterator, cast

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, Request, File as FastAPIFile
from fastapi.responses import StreamingResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from sse_starlette.event import ServerSentEvent
from langgraph.errors import GraphRecursionError
from langgraph.store.base import BaseStore

from ..agents.repos import AgentRepository
from ..ai_models import ai_factory
from ..core.api import BASE_PATH, with_heartbeat
from ..core.auth import get_current_user
from ..core.domain import CamelCaseModel
from ..core.env import env
from ..core import repos
from ..core.repos import get_db
from .agents_store import get_store
from ..files.api import build_file_download_response
from ..files.core import FileQuota, CurrentQuota, QuotaExceededError, add_encoding_to_content_type
from ..files.domain import File, FileStatus, FileMetadata, FileProcessor, FileMetadataWithContent
from ..files.parser import extract_file_text
from ..files.processors.pdf import is_enhanced_pdf_processor_available
from ..files.repos import FileRepository
from ..tools.auth import ToolAuthRequestException, build_tool_auth_request_http_exception
from ..usage.domain import Usage, UsageType, MessageUsage
from ..usage.repos import UsageRepository
from ..users.domain import User
from .domain import ThreadListItem, Thread, ThreadMessage, ThreadMessageOrigin, ThreadUpdate,\
    ThreadMessagePublic, ThreadMessageFile, ThreadMessageUpdate, AgentActionEvent, AgentFileEvent,\
    AgentMessageEvent, ThreadTranscriptionResult, ModelRateLimitError
from .engine import build_thread_name, AgentEngine, agent_store_namespace
from .repos import ThreadRepository, ThreadMessageRepository, ThreadMessageFileRepository
from .time_saved_estimation import estimate_minutes_saved


logger = logging.getLogger(__name__)
router = APIRouter()
THREADS_PATH = f"{BASE_PATH}/threads"
active_streaming_connections: dict[int, asyncio.Event] = {}


@router.get(THREADS_PATH)
async def find_threads(user: Annotated[User, Depends(get_current_user)],
                       db: Annotated[AsyncSession, Depends(get_db)],
                       text: Optional[str] = None,
                       limit: int = 10,
                       agent_id: Optional[int] = None,
                       exclude_empty: bool = False) -> List[ThreadListItem]:
    return await ThreadRepository(db).find_by_text(text, user.id, limit, agent_id, exclude_empty)


class ThreadCreateApi(CamelCaseModel):
    agent_id: int


async def find_or_create_thread(agent_id: int, user: User, db: AsyncSession) -> Thread:
    agent = await AgentRepository(db).find_by_id(agent_id)
    if not agent or not agent.is_visible_by(user):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Agent not found")
    repo = ThreadRepository(db)
    empty_thread = await repo.find_empty_thread(agent_id, user.id)
    if empty_thread:
        return empty_thread
    return await repo.add(Thread(agent_id=agent_id, user_id=user.id))


@router.post(THREADS_PATH, status_code=status.HTTP_201_CREATED)
async def start_thread(thread: ThreadCreateApi, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> ThreadListItem:
    ret = await find_or_create_thread(thread.agent_id, user, db)
    return ThreadListItem.from_thread(ret)


THREAD_PATH = f"{THREADS_PATH}/{{thread_id}}"


@router.get(THREAD_PATH)
async def find_thread(thread_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> ThreadListItem:
    ret = await _find_thread(thread_id, user.id, db)
    return ThreadListItem.from_thread(ret)


async def _find_thread(thread_id: int, user_id: int, db: AsyncSession) -> Thread:
    ret = await ThreadRepository(db).find_by_id(thread_id, user_id)
    if not ret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return ret


@router.put(THREAD_PATH)
async def update_thread(thread_id: int, thread: ThreadUpdate, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> ThreadListItem:
    ret = await _find_thread(thread_id, user.id, db)
    ret.update_with(thread)
    await ThreadRepository(db).update(ret)
    return ThreadListItem.from_thread(ret)


@router.delete(THREAD_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(thread_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
        store: Annotated[BaseStore, Depends(get_store)]) -> None:
    thread = await _find_thread(thread_id, user.id, db)
    try:
        namespace = agent_store_namespace(user.id, thread_id)
        while True:
            items = await store.asearch(namespace, limit=100)
            if not items:
                break
            for item in items:
                await store.adelete(namespace, item.key)

    except Exception:
        logger.warning(
            f"Failed to clean up store namespace for thread {thread_id} (user {user.id})",
            exc_info=True,
        )
    await ThreadRepository(db).delete(thread)


THREAD_MESSAGES_PATH = f"{THREAD_PATH}/messages"


@router.get(THREAD_MESSAGES_PATH, response_model=List[ThreadMessagePublic])
async def find_messages(thread_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> List[ThreadMessagePublic]:
    thread = await _find_thread(thread_id, user.id, db)
    messages = await ThreadMessageRepository(db).find_by_thread_id(thread.id)
    return _map_messages_to_tree(messages)


def _map_messages_to_tree(messages: List[ThreadMessage]) -> List[ThreadMessagePublic]:
    public_messages = [ThreadMessagePublic.from_message(msg) for msg in messages]
    message_lookup = {msg.id: msg for msg in public_messages}
    tree = []
    for msg in public_messages:
        if msg.parent_id in message_lookup:
            message_lookup[msg.parent_id].children.append(msg)
        else:
            tree.append(msg)
    return tree


class ThreadMessageOriginApi(Enum):
    USER = "USER"


@router.post(THREAD_MESSAGES_PATH, status_code=status.HTTP_201_CREATED)
async def add_message(thread_id: int, request: Request, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
        store: Annotated[BaseStore, Depends(get_store)], files: List[UploadFile] = []) -> StreamingResponse:
    thread = await _find_thread(thread_id, user.id, db)
    current_usage = await UsageRepository(db).find_current_month_user_usage_usd(user.id)
    if current_usage >= user.monthly_usd_limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="quotaExceeded")

    form = await request.form()
    message_text = cast(str, form.get("text", ""))
    message_origin = cast(str, form.get("origin", "USER"))
    file_ids_str = cast(str, form.get("fileIds"))
    file_ids = [ int(file_id) for file_id in file_ids_str.split(",") ] if file_ids_str else []
    parent_message_id_str = cast(Optional[str], form.get("parentMessageId"))
    is_in_agent_edition = cast(str, form.get("isInAgentEdition", "false")).lower() == "true"
    parent_message_id = int(parent_message_id_str) if parent_message_id_str else None

    if parent_message_id is not None:
        await _check_parent_message_id_exists(parent_message_id, thread.id, db)
    existing_files = [ await _find_thread_message_file(thread_id, file_id, db) for file_id in file_ids ]
    try:
        # initialize engine so any tool authentication requirements are triggered before creating anything on db
        engine = AgentEngine(thread.agent, user.id, db, store)
        async with AsyncExitStack() as stack:
            await engine.load_tools(stack)

        initial_thread_message = ThreadMessage(
            thread_id=thread.id,
            text=message_text,
            origin=ThreadMessageOrigin[message_origin],
            parent_id=parent_message_id
        )
        repo = ThreadMessageRepository(db)
        user_message = await repo.add(initial_thread_message)

        await _attach_existing_files_to_message(existing_files, user_message, db)
        await _handle_file_contents(files, user_message, user, thread, db)
        user_message = await repo.refresh_with_files(user_message)

        return StreamingResponse(
            with_heartbeat(_agent_response(user_message, thread, user.id, db, is_in_agent_edition, store)),
            media_type="text/event-stream",
        )
    except ToolAuthRequestException as e:
        raise build_tool_auth_request_http_exception(e.request)

    except QuotaExceededError:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="quotaExceeded")


async def _check_parent_message_id_exists(parent_message_id: int, thread_id: int, db: AsyncSession):
    existing_message = await ThreadMessageRepository(db).find_by_thread_id_and_message_id(thread_id, parent_message_id)
    if not existing_message:
        raise HTTPException(status.HTTP_400_BAD_REQUEST)


async def _find_thread_message_file(thread_id: int, file_id: int, db: AsyncSession) -> ThreadMessageFile:
    ret = await ThreadMessageFileRepository(db).find_by_thread_id_and_file_id(thread_id, file_id)
    if not ret:
        raise HTTPException(status.HTTP_400_BAD_REQUEST)
    return ret


async def _attach_existing_files_to_message(files: List[ThreadMessageFile], user_message: ThreadMessage, db: AsyncSession):
    repo = ThreadMessageFileRepository(db)
    for f in files:
        await repo.add(ThreadMessageFile(thread_message_id=user_message.id, file_id=f.file_id))


async def _handle_file_contents(files: List[UploadFile], user_message: ThreadMessage, user: User, thread: Thread, db: AsyncSession):
    file_repo = FileRepository(db)
    if files:
        for f in files:
            content = await f.read()
            content_type = add_encoding_to_content_type(f.content_type, content)
            file_processor = FileProcessor.ENHANCED if is_enhanced_pdf_processor_available() else FileProcessor.BASIC
            file = File(name=f.filename or "uploaded-file", content_type=content_type, content=content, user_id=user.id, file_processor=file_processor)
            pdf_parsing_usage = Usage(message_id=user_message.id, user_id=user.id, agent_id=thread.agent_id, model_id=None, type=UsageType.PDF_PARSING)
            current_usage = await UsageRepository(db).find_current_month_user_usage_usd(user.id)
            file_quota = FileQuota(pdf_parsing_usage, thread.agent, CurrentQuota(current_usage, user.monthly_usd_limit))
            try:
                file.processed_content = await extract_file_text(file, file_quota)
                file.status = FileStatus.PROCESSED
                saved_file = await file_repo.add(file)
                await ThreadMessageFileRepository(db).add(ThreadMessageFile(thread_message_id=user_message.id, file_id=saved_file.id))
            finally:
                await UsageRepository(db).add(pdf_parsing_usage)


async def _agent_response(message: ThreadMessage, thread: Thread, user_id: int, db: AsyncSession, is_in_agent_edition: bool, store: BaseStore) \
        -> AsyncIterator[bytes]:
    message_usage = None
    repo = ThreadMessageRepository(db)
    yield ServerSentEvent(event="userMessage", data=json.dumps({
        "id": message.id,
        "files": [FileMetadata.from_file(f.file).model_dump(mode="json", by_alias=True) for f in message.files if f.file]
    })).encode()
    complete_answer = ""
    files: List[FileMetadata] = []
    status_updates: List[AgentActionEvent] = []
    try:
        stop_event = asyncio.Event()
        active_streaming_connections[thread.id] = stop_event

        message_usage = MessageUsage(user_id=user_id, agent_id=thread.agent_id, model_id=thread.agent.model_id, message_id=message.id)
        thread_messages = await repo.find_previous_messages(message)

        thread_name_task = None
        if len(thread_messages) == 0:
            thread_name_task = asyncio.create_task(_build_and_save_thread_name(
                thread_id=thread.id,
                user_id=user_id,
                message_text=message.text,
                parent_message_id=message.id,
            ))

        answer_stream = AgentEngine(thread.agent, user_id, db, store).answer([*thread_messages, message], message_usage, stop_event)
        thread_title_sent = False

        async for event in answer_stream:
            if thread_name_task is not None and not thread_title_sent and thread_name_task.done():
                thread_title_sent = True
                thread_name = thread_name_task.result()
                if thread_name:
                    yield ServerSentEvent(event="threadUpdated", data=json.dumps({"name": thread_name})).encode()
            if isinstance(event, AgentActionEvent):
                status_updates.append(event)
                payload = json.dumps(event.model_dump(mode="json", by_alias=True))
                yield ServerSentEvent(event="status", data=payload).encode()
            elif isinstance(event, AgentFileEvent):
                files.append(event.file)
            elif isinstance(event, AgentMessageEvent):
                complete_answer = complete_answer + event.content
                yield ServerSentEvent(data=event.content).encode()
            else:
                raise RuntimeError(f"Unsupported event type: {type(event)}")

        should_estimate_minutes_saved = not stop_event.is_set() and not is_in_agent_edition
        minutes_saved = 0 if not should_estimate_minutes_saved else None

        answer = await repo.add(ThreadMessage(
            thread_id=thread.id,
            text=complete_answer,
            origin=ThreadMessageOrigin.AGENT,
            parent_id=message.id,
            minutes_saved=minutes_saved,
            stopped=stop_event.is_set(),
            status_updates=_dump_status_updates(status_updates)
        ))
        for f in files:
            await ThreadMessageFileRepository(db).add(ThreadMessageFile(thread_message_id=answer.id, file_id=f.id))

        yield ServerSentEvent(event="metadata", data=json.dumps({
            "answerMessageId": answer.id,
            "files": [f.model_dump(mode="json", by_alias=True) for f in files],
            "minutesSaved": answer.minutes_saved,
            "stopped": answer.stopped
        })).encode()

        if should_estimate_minutes_saved:
            minutes_saved = await _estimate_and_save_minutes_saved(
                answer_message=answer,
                user_message=message.text,
                agent_response=complete_answer,
                thread=thread,
                thread_messages=thread_messages,
                user_id=user_id,
                db=db,
            )
            if minutes_saved is not None:
                yield ServerSentEvent(event="messageUpdated", data=json.dumps({
                    "messageId": answer.id,
                    "minutesSaved": minutes_saved,
                })).encode()

        if thread_name_task is not None and not thread_title_sent:
            thread_name = await thread_name_task
            if thread_name:
                yield ServerSentEvent(event="threadUpdated", data=json.dumps({
                    "name": thread_name,
                })).encode()

    except* asyncio.CancelledError:
        logger.info(f"Streaming response cancelled for thread {thread.id}, message {message.id}")

        has_partial = bool(complete_answer)
        interrupted_msg = ThreadMessage(
            thread_id=thread.id,
            text=complete_answer if has_partial else "ERROR_INTERRUPTED",
            origin=ThreadMessageOrigin.AGENT if has_partial else ThreadMessageOrigin.SYSTEM,
            parent_id=message.id,
            minutes_saved=0 if has_partial else None,
            stopped=has_partial,
            status_updates=_dump_status_updates(status_updates),
        )

        save_task = asyncio.create_task(_save_interrupted_message(thread.id, interrupted_msg, files))
        try:
            await asyncio.shield(save_task)
        except asyncio.CancelledError:
            pass
        raise
    except* GraphRecursionError:
        await repo.add(ThreadMessage(
            thread_id=thread.id,
            text="ERROR_RECURSION_LIMIT_EXCEEDED",
            origin=ThreadMessageOrigin.SYSTEM,
            parent_id=message.id,
            status_updates=_dump_status_updates(status_updates)
        ))
        yield ServerSentEvent(event="error", data="recursionLimitExceeded").encode()
    except* ModelRateLimitError:
        await repo.add(ThreadMessage(
            thread_id=thread.id,
            text="ERROR_MODEL_RATE_LIMIT",
            origin=ThreadMessageOrigin.SYSTEM,
            parent_id=message.id,
            status_updates=_dump_status_updates(status_updates)
        ))
        yield ServerSentEvent(event="error", data="modelRateLimitExceeded").encode()
    except* Exception:
        logger.exception(f"Problem answering message in thread {thread.id}")
        await repo.add(ThreadMessage(
            thread_id=thread.id,
            text="ERROR_GENERIC",
            origin=ThreadMessageOrigin.SYSTEM,
            parent_id=message.id,
            status_updates=_dump_status_updates(status_updates)
        ))
        yield ServerSentEvent(event="error").encode()
    finally:
        await UsageRepository(db).add(message_usage)
        del active_streaming_connections[thread.id]


async def _build_and_save_thread_name(
        thread_id: int,
        user_id: int,
        message_text: str,
        parent_message_id: int,
) -> Optional[str]:
    try:
        async with AsyncSession(repos.engine, expire_on_commit=False) as db:
            thread = await _find_thread(thread_id, user_id, db)
            title_usage = MessageUsage(
                user_id=user_id,
                agent_id=thread.agent_id,
                model_id=env.internal_generator_model,
                message_id=parent_message_id,
            )
            thread.name = await build_thread_name(message_text, title_usage, db)
            await ThreadRepository(db).update(thread)
            await UsageRepository(db).add(title_usage)
            return thread.name
    except Exception:
        logger.exception(f"Failed to build thread name for thread {thread_id}")
        return None


async def _estimate_and_save_minutes_saved(
        answer_message: ThreadMessage,
        user_message: str,
        agent_response: str,
        thread: Thread,
        thread_messages: List[ThreadMessage],
        user_id: int,
        db: AsyncSession,
) -> Optional[int]:
    try:
        evaluator_usage = MessageUsage(
            user_id=user_id,
            agent_id=thread.agent_id,
            model_id=cast(str, env.internal_evaluator_model),
            message_id=answer_message.parent_id,
        )
        minutes_saved = await estimate_minutes_saved(
            user_message=user_message,
            agent_response=agent_response,
            thread=thread,
            thread_messages=thread_messages,
            message_usage=evaluator_usage,
            db=db,
        )
        answer_message.minutes_saved = minutes_saved
        await ThreadMessageRepository(db).update(answer_message)
        await UsageRepository(db).add(evaluator_usage)
        return minutes_saved
    except Exception:
        logger.exception(f"Failed to estimate minutes saved for message {answer_message.id}")
        return None


async def _save_interrupted_message(thread_id: int, interrupted_msg: ThreadMessage, files: List[FileMetadata]) -> None:
    try:
        async with AsyncSession(repos.engine, expire_on_commit=False) as iso_session:
            saved_interrupted_msg = await ThreadMessageRepository(iso_session).add(interrupted_msg)
            for f in files:
                await ThreadMessageFileRepository(iso_session).add(
                    ThreadMessageFile(thread_message_id=saved_interrupted_msg.id, file_id=f.id)
                )
    except Exception:
        logger.exception(f"Failed saving interrupted message for thread {thread_id}")


def _dump_status_updates(status_updates: List[AgentActionEvent]) -> Optional[List[dict]]:
    return [event.model_dump(mode="json", by_alias=True) for event in status_updates] if status_updates else None

@router.post(THREAD_PATH + "/stop", status_code=status.HTTP_200_OK)
async def stop_message(thread_id: int,
                      user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]):
    stop_event = active_streaming_connections.get(thread_id)
    if stop_event:
        stop_event.set()
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)


THREAD_MESSAGE_PATH = f"{THREAD_PATH}/messages/{{message_id}}"


@router.put(THREAD_MESSAGE_PATH)
async def update_message(thread_id: int, message_id: int, updated_message: ThreadMessageUpdate,
                      user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]):
    thread = await _find_thread(thread_id, user.id, db)
    thread_message = await _find_thread_message(message_id, db)
    has_positive_feedback = thread_message.has_positive_feedback
    thread_message.update_with(updated_message)
    if has_positive_feedback is not None and updated_message.has_positive_feedback is None:
        minutes_saved = 0
        thread_message_repo = ThreadMessageRepository(db)
        if thread_message.parent_id is not None:
            parent_message = cast(ThreadMessage, await thread_message_repo.find_by_id(thread_message.parent_id))
            message_usage = MessageUsage(user_id=user.id, agent_id=thread.agent_id, model_id=env.internal_generator_model, message_id=parent_message.id)
            minutes_saved = await estimate_minutes_saved(
                user_message=parent_message.text,
                agent_response=thread_message.text,
                thread=thread,
                thread_messages=await thread_message_repo.find_previous_messages(parent_message),
                message_usage=message_usage,
                db=db
            )
            await UsageRepository(db).add(message_usage)
        thread_message.minutes_saved = minutes_saved
    await ThreadMessageRepository(db).update(thread_message)
    return thread_message


async def _find_thread_message(message_id:int, db: AsyncSession) -> ThreadMessage:
    ret = await ThreadMessageRepository(db).find_by_id(message_id)
    if not ret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return ret

THREAD_FILE_PATH = f"{THREAD_PATH}/files/{{file_id}}"


@router.get(THREAD_FILE_PATH)
async def find_thread_file(thread_id: int, file_id: int, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> FileMetadataWithContent:
    await _find_thread(thread_id, user.id, db)
    message_file = await ThreadMessageFileRepository(db).find_by_thread_id_and_file_id(thread_id, file_id)
    if not message_file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    file = cast(File, await FileRepository(db).find_by_id(message_file.file_id))
    return FileMetadataWithContent.from_file(file)


@router.get(f"{THREAD_FILE_PATH}/content")
async def download_thread_file(thread_id: int, file_id: int, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> StreamingResponse:
    await _find_thread(thread_id, user.id, db)
    file = await ThreadMessageFileRepository(db).find_with_content_by_ids(thread_id, file_id)
    return build_file_download_response(file)


AUDIO_FORMAT = "audio/webm"


@router.post(f"{THREAD_PATH}/transcriptions")
async def get_transcript(user: Annotated[User, Depends(get_current_user)], thread_id: int, file: UploadFile = FastAPIFile(...)) -> ThreadTranscriptionResult:
    return await transcribe_audio(file, env.transcription_model)


async def transcribe_audio(file: UploadFile, model: str) -> ThreadTranscriptionResult:
    provider = ai_factory.get_provider(model)
    if file.content_type != AUDIO_FORMAT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"File type invalid: {file.content_type}"
        )

    audio_file = await _prepare_audio_file(file)
    transcription = await provider.transcribe_audio(audio_file, model)
    return ThreadTranscriptionResult(transcription=transcription)


async def _prepare_audio_file(file: UploadFile) -> io.BytesIO:
    file_bytes = await file.read()
    audio_file = io.BytesIO(file_bytes)
    audio_file.name = file.filename
    return audio_file
