from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Optional

from ...agents.domain import AgentToolConfig
from ..auth import ToolAuthCallback
from ..core import AgentTool
from .sql_backend import SqlBackend


class SqlAuth(ABC):
    def __init__(self, tool: AgentTool):
        self._tool = tool

    @abstractmethod
    async def setup(self, prev_config: Optional[AgentToolConfig]) -> None: ...

    @abstractmethod
    async def teardown(self) -> None: ...

    @abstractmethod
    def get_backend(self) -> AbstractAsyncContextManager[SqlBackend]: ...

    @abstractmethod
    async def auth(self, auth_callback: ToolAuthCallback) -> None: ...
