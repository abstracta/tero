import abc
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, List, Union

from pydantic import field_serializer
from sqlalchemy import Column, Text
from sqlmodel import SQLModel, Field, Relationship, Index, JSON

from ..agents.domain import Agent, AgentListItem
from ..core.domain import CamelCaseModel
from ..files.domain import FileMetadata, File
from ..users.domain import User


MAX_THREAD_NAME_LENGTH = 80


class BaseThread(SQLModel, abc.ABC):
    id: int = Field(primary_key=True, default=None)
    name: Optional[str] = Field(max_length=MAX_THREAD_NAME_LENGTH, default=None)
    user_id: int = Field(foreign_key="user.id", index=True)
    agent_id: int = Field(foreign_key="agent.id")
    deleted: bool = Field(default=False)


class ThreadUpdate(CamelCaseModel):
    name: str


class Thread(BaseThread, table=True):
    __table_args__ = (
        Index('ix_thread_user_id_agent_id_creation', 'user_id', 'agent_id', 'creation', 'is_test_case'),
    )
    agent: Agent = Relationship()
    creation: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user: User = Relationship()
    messages: List["ThreadMessage"] = Relationship(sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    is_test_case: bool = Field(default=False)

    def update_with(self, update: ThreadUpdate):
        update_dict = update.model_dump(exclude_unset=True)
        self.sqlmodel_update(update_dict)

    def set_default_name(self):
        self.name = f"Chat #{self.id}"

    @field_serializer('creation')
    def serialize_creation(self, value: datetime) -> str:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        naive_value = value.replace(tzinfo=None)
        return naive_value.isoformat()


class ThreadListItem(BaseThread, CamelCaseModel):
    agent: AgentListItem
    creation: Optional[datetime] = None
    last_message: Optional[datetime] = None

    @staticmethod
    def from_thread(thread: Thread, last_message: Optional[datetime] = None, creation: Optional[datetime] = None,
                    agent: Optional[AgentListItem] = None) -> 'ThreadListItem':
        return ThreadListItem.model_validate(
            {**thread.model_dump(),
             "agent": agent or AgentListItem.from_agent(thread.agent, thread.agent.is_editable_by(thread.user)),
             "creation": thread.creation if creation is None else creation,
             "last_message": last_message})


class ThreadMessageOrigin(Enum):
    USER = 'USER'
    AGENT = 'AGENT'
    SYSTEM = 'SYSTEM'


class ThreadMessageUpdate(CamelCaseModel):
    minutes_saved: Optional[int] = None
    feedback_text: Optional[str] = None
    has_positive_feedback: Optional[bool] = None


class ThreadMessage(CamelCaseModel, table=True):
    __tablename__ : Any = "thread_message"
    __table_args__ = (
        Index('ix_thread_message_thread_id_timestamp', 'thread_id', 'timestamp'),
    )
    id: int = Field(primary_key=True, default=None)
    thread_id: int = Field(foreign_key="thread.id", index=True)
    origin: ThreadMessageOrigin
    text: str = Field(sa_column=Column(Text))
    collapsed_name: Optional[str] = Field(default=None, max_length=50)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stopped: bool = Field(default=False)

    parent_id: Optional[int] = Field(
        default=None,
        foreign_key="thread_message.id",
        index=True,
    )

    minutes_saved: Optional[int] = None
    feedback_text: Optional[str] = None
    has_positive_feedback: Optional[bool] = None
    status_updates: Optional[List] = Field(default=None, sa_column=Column(JSON))
    files: List["ThreadMessageFile"] = Relationship(back_populates="thread_message")

    def update_with(self, update: ThreadMessageUpdate):
        update_dict = update.model_dump()
        self.sqlmodel_update(update_dict)


class ThreadMessageFile(CamelCaseModel, table=True):
    __tablename__ : Any = "thread_message_file"
    thread_message_id: int = Field(primary_key=True, foreign_key="thread_message.id", index=True, ondelete="CASCADE")
    file_id: int = Field(primary_key=True, foreign_key="file.id", index=True, ondelete="CASCADE")

    thread_message: "ThreadMessage" = Relationship(back_populates="files")
    file: "File" = Relationship()


# Explicitly declared fastapi response model due to "feedback" field being ignored otherwise
class ThreadMessagePublic(CamelCaseModel, table=False):
    id: int
    thread_id: int
    origin: ThreadMessageOrigin
    text: str
    collapsed_name: Optional[str] = None
    files: Optional[List[FileMetadata]] = None
    timestamp: datetime
    parent_id: Optional[int] = None
    children: List["ThreadMessagePublic"] = []
    minutes_saved: Optional[int] = None
    feedback_text: Optional[str] = None
    has_positive_feedback: Optional[bool] = None
    stopped: bool = False
    status_updates: Optional[List] = None

    @staticmethod
    def from_message(message: ThreadMessage) -> 'ThreadMessagePublic':
        return ThreadMessagePublic.model_validate(
            {**message.model_dump(),
             "files": [FileMetadata.from_file(m.file) for m in message.files if m.file] if message.files else None})


class AgentEvent(abc.ABC, CamelCaseModel):
    pass


class AgentAction(str, Enum):
    EXECUTING_TOOL = "executingTool"
    EXECUTED_TOOL = "executedTool"
    TOOL_ERROR = "toolError"
    PRE_MODEL_HOOK = "preModelHook"
    UNDEFINED = "undefined"
    PLANNING = "planning"


class AgentActionEvent(AgentEvent):
    action: AgentAction
    tool_name: Optional[str] = None
    description: Optional[str] = None
    result: Optional[Union[str, List[Any]]] = None
    args: Optional[dict[str, Any]] = None
    error_id: Optional[str] = None
    agent_name: Optional[str] = None


class AgentFileEvent(AgentEvent):
    file: FileMetadata


class AgentMessageEvent(AgentEvent):
    content: str


class ThreadTranscriptionResult(CamelCaseModel):
    transcription: str


class ModelRateLimitError(Exception):
    pass


class StatusDetailPublic(CamelCaseModel):
    value: str
