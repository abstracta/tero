from typing import cast

from ..auth import AgentToolOauth
from ..core import load_schema
from ..mcp_tool import AuthType, BaseMcpTool


PRACTITEST_TOOL_ID = "practitest"


class PractiTestTool(BaseMcpTool):
    id: str = PRACTITEST_TOOL_ID
    name: str = "PractiTest"
    description: str = "Manage tests, test sets and requirements"
    config_schema: dict = load_schema(__file__)

    def _get_auth_type(self) -> AuthType:
        return AuthType.OAUTH

    def _get_mcp_server_url(self) -> str:
        server_url = cast(str, self.config.get("serverUrl") or "")
        if not server_url:
            raise ValueError("PractiTest MCP serverUrl is not configured")
        return server_url

    def _get_oauth(self) -> AgentToolOauth:
        ret = super()._get_oauth()
        ret.context.client_metadata.client_name = "Tero"
        ret.context.client_metadata.scope = "mcp"
        return ret
