import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.errors import GraphRecursionError
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.api import MCP_PATH
from ..core.auth import get_current_user
from ..core.repos import get_db
from .agents_store import get_store
from ..tools.auth import ToolAuthRequestException
from ..usage.domain import MessageUsage
from ..usage.repos import UsageRepository
from ..users.domain import User
from .api import find_or_create_thread, _find_thread, _build_and_save_thread_name
from .domain import ThreadMessage, ThreadMessageOrigin, AgentMessageEvent
from .engine import AgentEngine
from .repos import ThreadMessageRepository

logger = logging.getLogger(__name__)

router = APIRouter()


class McpThread(BaseModel):
    thread_id: int
    thread_name: Optional[str]


class McpSendMessageRequest(BaseModel):
    message: str = Field(description="The message text to send to the agent")


class McpMessageResponse(BaseModel):
    response: str
    thread_id: int


@router.post(f"{MCP_PATH}/start-conversation", operation_id="start-conversation")
async def start_conversation(
        agent_id: Annotated[int, Query(description="The ID of the agent to start a conversation with")],
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
) -> McpThread:
    """
    Start a new conversation with an agent.

    Creates a new conversation thread for the specified agent, or returns
    an existing empty thread if one already exists.
    """
    thread = await find_or_create_thread(agent_id, user, db)
    return McpThread(thread_id=thread.id, thread_name=thread.name)


@router.post(f"{MCP_PATH}/send-message", operation_id="send-message")
async def send_message(
        thread_id: Annotated[int, Query(description="The ID of the conversation thread")],
        body: McpSendMessageRequest,
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
        store: Annotated[BaseStore, Depends(get_store)],
) -> McpMessageResponse:
    """
    Send a message in a conversation and get the agent's full response.

    Sends a text message to the agent in the specified thread and returns
    the complete response. This is a non-streaming call that waits for the
    full agent response before returning.
    """
    thread = await _find_thread(thread_id, user.id, db)

    current_usage = await UsageRepository(db).find_current_month_user_usage_usd(user.id)
    if current_usage >= user.monthly_usd_limit:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Monthly usage quota exceeded")

    try:
        engine = AgentEngine(thread.agent, user.id, db, store)
        async with AsyncExitStack() as stack:
            await engine.load_tools(stack)
    except ToolAuthRequestException:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent tools require authentication that cannot be completed via MCP"
        )

    repo = ThreadMessageRepository(db)
    user_message = await repo.add(ThreadMessage(
        thread_id=thread.id,
        text=body.message,
        origin=ThreadMessageOrigin.USER,
    ))

    message_usage = MessageUsage(
        user_id=user.id, agent_id=thread.agent_id,
        model_id=thread.agent.model_id, message_id=user_message.id,
    )

    thread_messages = await repo.find_previous_messages(user_message)
    if len(thread_messages) == 0:
        asyncio.create_task(_build_and_save_thread_name(
            thread_id=thread.id,
            user_id=user.id,
            message_text=body.message,
            parent_message_id=user_message.id,
        ))

    complete_answer = ""
    try:
        stop_event = asyncio.Event()
        answer_stream = AgentEngine(thread.agent, user.id, db, store).answer(
            [*thread_messages, user_message], message_usage, stop_event,
        )
        async for event in answer_stream:
            if isinstance(event, AgentMessageEvent):
                complete_answer += event.content

    except* GraphRecursionError:
        await repo.add(ThreadMessage(
            thread_id=thread.id,
            text="ERROR_RECURSION_LIMIT_EXCEEDED",
            origin=ThreadMessageOrigin.SYSTEM,
            parent_id=user_message.id,
        ))
        await UsageRepository(db).add(message_usage)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent exceeded its recursion limit while processing the message"
        )
    except* Exception:
        logger.exception(f"Error answering MCP message in thread {thread.id}")
        await repo.add(ThreadMessage(
            thread_id=thread.id,
            text="ERROR_GENERIC",
            origin=ThreadMessageOrigin.SYSTEM,
            parent_id=user_message.id,
        ))
        await UsageRepository(db).add(message_usage)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the message"
        )

    await repo.add(ThreadMessage(
        thread_id=thread.id,
        text=complete_answer,
        origin=ThreadMessageOrigin.AGENT,
        parent_id=user_message.id,
    ))
    await UsageRepository(db).add(message_usage)

    return McpMessageResponse(response=complete_answer, thread_id=thread.id)
