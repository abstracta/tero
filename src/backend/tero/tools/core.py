import abc
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
from typing import Any, List, Optional, cast, Callable
from uuid import UUID

import jsonschema
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.config import get_stream_writer
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.domain import Agent, AgentToolConfig
from ..agents.repos import AgentToolConfigFileRepository, AgentToolConfigFile
from ..core.assets import solve_module_path
from ..core.domain import CamelCaseModel
from ..files.domain import File, FileMetadata
from ..files.repos import FileRepository
from ..threads.domain import AgentActionEvent, AgentAction
from ..tools.auth import ToolAuthCallback
from ..usage.domain import ToolUsage
from ..users.domain import User


_SECRET_MASK = "********"


def load_schema(tool_path: str) -> dict:
    with open(_build_schema_path(tool_path)) as schema_file:
        ret = json.load(schema_file)
    return _fix_core_schema_references(ret)


def _build_schema_path(tool_path: str) -> str:
    return solve_module_path(f'tool-schema.json', tool_path)


CORE_SCHEMA_PATH = _build_schema_path(__file__)


def _fix_core_schema_references(ret: dict) -> dict:
    core_schema_file_ref = "/tool-schema.json/"
    for key, value in ret.items():
        if key == "$ref" and core_schema_file_ref in value:
            ret[key] = f"file://{CORE_SCHEMA_PATH}/{value.split(core_schema_file_ref, 1)[1]}"
        elif isinstance(value, dict):
            ret[key] = _fix_core_schema_references(value)
    return ret


class StatusUpdateCallbackHandler(AsyncCallbackHandler):
    def __init__(self, tool_id: str, description: Optional[str] = "", 
                response_parser: Optional[Callable[[Any], List[str]]] = None):
        self.tool_id = tool_id
        self.description = description
        self.response_parser = response_parser

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        get_stream_writer()(
            AgentActionEvent(
                action=AgentAction.EXECUTING_TOOL,
                tool_name=self.tool_id,
                args=self._cleanup_inputs(inputs),
                description=self.description,
            )
        )

    def _cleanup_inputs(self, inputs: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return { key: value for key, value in inputs.items() if not isinstance(value, ToolRuntime) } if inputs else None

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        result = self.response_parser(output.content) if self.response_parser else output.content[0:200] + "..."
        get_stream_writer()(
            AgentActionEvent(
                action=AgentAction.EXECUTED_TOOL,
                tool_name=self.tool_id,
                result=result,
            )
        )

    async def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        get_stream_writer()(
            AgentActionEvent(
                action=AgentAction.TOOL_ERROR,
                tool_name=self.tool_id,
                result=[str(error)[:200] + "..."],
            )
        )


class AgentTool(CamelCaseModel, abc.ABC):
    id: str
    name: str
    description: str
    config_schema: dict
    _agent: Optional[Agent] = None
    _user_id: Optional[int] = None
    _config: Optional[dict] = None
    _secrets: Optional[dict] = None
    _db: Optional[AsyncSession] = None
    _thread_id: Optional[int] = None

    # this method is invoked every time the agent tool is configured or used for a given agent and user id
    def configure(self, agent: Agent, user_id: int, config: dict, db: AsyncSession, thread_id: Optional[int] = None):
        self._agent = agent
        self._user_id = user_id
        self._config = config
        self._db = db
        self._thread_id = thread_id

    @property
    def agent(self) -> Agent:
        return cast(Agent, self._agent)

    @property
    def user_id(self) -> int:
        return cast(int, self._user_id)

    @property
    def config(self) -> dict:
        return cast(dict, self._config)

    @property
    def db(self) -> AsyncSession:
        return cast(AsyncSession, self._db)

    # this method is invoked when the tool is configured or the configuration changes
    # this method will automatically extract and mask secrets in the configuration
    # use _get_secret method to get the secrets inside _setup_tool method
    async def setup(self, prev_config: Optional[AgentToolConfig]):
        try:
            # files are uploaded to endpoint and not included in file config body
            # they are included in schema so frontend knows if files can be uploaded for this tool
            jsonschema.validate(self._config, self.get_schema_without_files(self.config_schema))
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid configuration: {e.message}")
        self._secrets = self._extract_secrets()
        self._config = self._mask_secrets()
        await self._setup_tool(prev_config)

    def _extract_secrets(self) -> dict:
        return {key: value for key, value in cast(dict, self._config).items() if self.config_schema.get("properties", {}).get(key, {}).get("writeOnly")}

    def _mask_secrets(self) -> dict:
        return {key: _SECRET_MASK if key in self._secrets else value for key, value in cast(dict, self._config).items()}

    def _get_secret(self, key: str) -> Optional[str]:
        ret = cast(dict, self._secrets).get(key)
        return ret if ret != _SECRET_MASK else None

    def get_schema_without_files(self, schema: dict) -> dict:
        file_props = []
        props = {}
        ret = schema.copy()
        props = schema.get("properties")
        if not props or not isinstance(props, dict):
            return ret
        for key, value in props.items():
            if self._is_file_property(value):
                file_props.append(key)
            else:
                props[key] = value
        ret["properties"] = props
        required_fields = []
        for req_filed in schema.get("required", []):
            if req_filed not in file_props:
                required_fields.append(req_filed)
        ret["required"] = required_fields
        return ret

    @staticmethod
    def _is_file_property(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        ref = value.get("$ref")
        if ref and ref.endswith("/File"):
            return True
        items = value.get("items")
        if not isinstance(items, dict):
            return False
        ref = items.get("$ref")
        return bool(ref and ref.endswith("/File"))

    # override this method to perform any setup that is persistent across agent tool usages (eg: create tables or any resources, etc)
    # use _get_secret method to get the secrets already extracted by setup method
    @abc.abstractmethod
    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        pass

    # override this method to perform any teardown when the tool is removed from an anget (eg: drop tables or any resources, etc)
    @abc.abstractmethod
    async def teardown(self):
        pass

    async def add_file(self, file: File, user: User):
        raise NotImplementedError()

    async def update_file(self, file: File, user: User):
        raise NotImplementedError()

    async def remove_file(self, file: File):
        raise NotImplementedError()
    
    # override this method to perform some initialization that is required for using the agent tool (eg: trigger user authentication, etc)
    # this method should raise a ToolAuthRequestException if authentication is required to use the tool and the user hasn't authenticated yet
    @abc.abstractmethod
    @asynccontextmanager
    async def load(self) -> AsyncIterator['AgentTool']:
        yield self

    @abc.abstractmethod
    async def build_langchain_tools(self) -> List[BaseTool]:
        pass

    # this method is invoked when trying to complete authentication either by setup (when using oauth) or 
    # when a user first interacts with the tool and sends the authentication token
    async def auth(self, auth_callback: ToolAuthCallback):
        raise NotImplementedError()
    
    # override this method to perform any cloning that is required for using the agent tool (eg: clone embeddings, etc)
    @abc.abstractmethod
    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass


class AgentToolWithFiles(AgentTool, abc.ABC):

    @abc.abstractmethod
    async def add_file(self, file: File, user: User):
        pass

    @abc.abstractmethod
    async def update_file(self, file: File, user: User):
        pass

    @abc.abstractmethod
    async def remove_file(self, file: File):
        pass

    async def _clone_files(
        self,
        agent_id: int,
        cloned_agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> dict:
        tool_file_repo = AgentToolConfigFileRepository(db)
        files = await tool_file_repo.find_with_content_by_agent_and_tool(
            agent_id, tool_id
        )
        file_id_map = {}

        for file in files:
            new_file = file.clone(user_id=user_id)
            new_file = await FileRepository(db).add(new_file)
            await tool_file_repo.add(
                AgentToolConfigFile(
                    agent_id=cloned_agent_id, tool_id=tool_id, file_id=new_file.id
                )
            )
            file_id_map[file.id] = new_file.id
        return file_id_map


class AgentToolMetadata(CamelCaseModel):
    tool_usage: Optional[ToolUsage] = None
    file: Optional[FileMetadata] = None
