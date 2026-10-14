import copy

import aiofiles
import aiofiles.os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import re
from typing import List, Optional, Any, cast, Annotated

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, InjectedToolCallId
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from sqlmodel.ext.asyncio.session import AsyncSession
from json_schema_to_pydantic import create_model
from pydantic import BaseModel

from ...core.env import env
from ...files.domain import File, FileStatus, FileMetadata
from ...files.repos import FileRepository
from ..core import AgentTool, AgentToolConfig, load_schema, StatusUpdateCallbackHandler
from ..mcp_tool import wrap_mcp_tool_errors


BROWSER_TOOL_ID = "browser"


class ScreenshotPersistingTool(BaseTool):
    _PLAYWRIGHT_OUTPUT_DIR = "/tmp/playwright-output"

    def __init__(self, tool: BaseTool, user_id: int, thread_id: Optional[int], db: AsyncSession):
        super().__init__(
            name=tool.name,
            description=tool.description + ". IT IS IMPORTANT TO AVOID USING PROVIDED PATH TO GENERATE RESPONSES, unless the user explicitly asks for it, since the agent already provides other means to download the screenshot. IT IS VERY IMPORTANT TO NOT USE RETURNED URL AS SOURCE OF AN IMAGE.",
            args_schema=self._build_args_schema(cast(dict, tool.args_schema)),
            metadata=tool.metadata,
        )
        self._mcp_tool = tool
        self._user_id = user_id
        self._db = db
        self._thread_id = thread_id

    def _build_args_schema(self, args_schema: dict) -> type[BaseModel]:
        # Custom screenshot names from Playwright MCP may be written outside --output-dir; we only read the shared
        # output path, so callers must use the default generated filename there.
        schema = copy.deepcopy(args_schema)
        props = schema.get("properties")
        if isinstance(props, dict):
            props.pop("filename", None)
        req = schema.get("required")
        if isinstance(req, list):
            schema["required"] = [r for r in req if r != "filename"]

        OriginalModel = create_model(schema)
        class ScreenshotPersistingToolArgs(OriginalModel):
            tool_call_id: Annotated[str, InjectedToolCallId]
        return ScreenshotPersistingToolArgs

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def _arun(self, *args: Any, config: RunnableConfig,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None,
        **kwargs: Any) -> Any:

        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and k != "tool_call_id"}
        result = await self._mcp_tool._arun(*args, config=config, run_manager=run_manager, **clean_kwargs)
        path = self._extract_screenshot_path(result)
        file = await self._save_screenshot_to_database(path)
        return ToolMessage(
            tool_call_id=tool_call_id,
            name=self.name,
            content=self._replace_file_path_references(result[0], path, file),
            response_metadata={"file": file}
        )

    def _extract_screenshot_path(self, result: Any) -> str:
        # The MCP Playwright tool usually returns the full output path, but in
        # some cases it may return only the screenshot filename inside a markdown
        # link, e.g. `(01_home_loaded.png)`.
        content = result[0]

        full_path_match = re.search(
            rf"{re.escape(self._PLAYWRIGHT_OUTPUT_DIR)}/.*?\.png",
            content,
        )
        if full_path_match:
            return cast(re.Match, full_path_match).group(0)

        file_name = None
        link_target_match = re.search(r"\(([^)]+\.png)\)", content)
        if link_target_match:
            file_name = link_target_match.group(1).split("/")[-1]
        else:
            png_names = re.findall(r"([^/\s]+\.png)", content)
            if png_names:
                file_name = png_names[-1]

        if file_name:
            return f"{self._PLAYWRIGHT_OUTPUT_DIR}/{file_name}"

        raise ValueError(f"Could not extract screenshot path from: {content!r}")

    async def _save_screenshot_to_database(self, file_path: Any) -> FileMetadata:
        host_path = file_path.replace(self._PLAYWRIGHT_OUTPUT_DIR, env.browser_tool_playwright_output_dir)
        file_name = host_path.split("/")[-1]
        async with aiofiles.open(host_path, "rb") as f:
            ret = await FileRepository(self._db).add(File(name=file_name,
                content_type="image/png",
                user_id=self._user_id,
                content=await f.read(),
                status=FileStatus.PROCESSED))
            await aiofiles.os.remove(host_path)
            return FileMetadata.from_file(ret)

    def _replace_file_path_references(self, content: str, path: str, file: FileMetadata) -> str:
        final_path = f"/chat/{self._thread_id}/files/{file.id}"
        final_absolute_url = f"{env.frontend_url}{final_path}"
        ret = content.replace(f"../.." + path, final_absolute_url)
        ret = ret.replace(path, final_absolute_url)
        file_name = path.split("/")[-1]
        return ret.replace(f"({file_name})", final_path)


class BrowserTool(AgentTool):
    id: str = BROWSER_TOOL_ID
    name: str = "Browser Tools"
    description: str = "Interact with web sites and capture screenshots"
    config_schema: dict = load_schema(__file__)
    _tools: Optional[list[BaseTool]] = None

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        pass

    async def teardown(self):
        pass

    @asynccontextmanager
    async def load(self) -> AsyncIterator['BrowserTool']:
        server_name = "playwright"
        client = MultiServerMCPClient({server_name: {"transport": "streamable_http", "url": env.browser_tool_playwright_mcp_url }})
        async with client.session(server_name) as mcp_session:
            self._tools = wrap_mcp_tool_errors(await load_mcp_tools(mcp_session))
            self._tools = [ScreenshotPersistingTool(cast(StructuredTool, t), self.user_id, self._thread_id, cast(AsyncSession, self._db)) if t.name == "browser_take_screenshot" else t for t in self._tools]
            for tool in self._tools:
                tool.callbacks = [StatusUpdateCallbackHandler(tool.name, description=tool.description)]
            yield self

    async def clone(
        self,
        agent_id: int,
        cloned_agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> None:
        pass

    async def build_langchain_tools(self) -> List[BaseTool]:
        if not self._tools:
            raise RuntimeError("Browser tool has not been set up properly")
        return self._tools
