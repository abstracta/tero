from typing import List, Optional

from ..core.env import env
from .core import AgentTool
from .docs import DocsTool
from .jira import JiraTool
from .zephyr import ZephyrTool
from .redmine import RedmineTool
from .github import GitHubTool
from .youtrack import YouTrackTool
from .practitest import PractiTestTool
from .mcp import McpTool
from .web import WebTool
from .browser import BrowserTool
from .sql import SqlTool
from .azure_devops import AzureDevOpsTool

class ToolRepository:

    def __init__(self):
        self._tools: List[AgentTool] = [
            DocsTool(), McpTool(), JiraTool(), BrowserTool(), GitHubTool(), YouTrackTool(),
            ZephyrTool(), PractiTestTool(), RedmineTool(), SqlTool(), AzureDevOpsTool(),
        ]

        if env.web_tool_tavily_api_key or (env.web_tool_google_api_key and env.web_tool_google_custom_search_engine_id):
            self._tools.append(WebTool())

    def find_agent_tools(self) -> List[AgentTool]:
        return sorted(self._tools, key=lambda t: t.name.casefold())

    def find_by_id(self, tool_id: str) -> Optional[AgentTool]:
        return next((t for t in self._tools if t.id == tool_id or t.id.endswith("-*") and tool_id.startswith(t.id[:-1])), None)
