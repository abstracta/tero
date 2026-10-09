from httpx import HTTPStatusError

from ..auth import OwnerConfiguredToolAuthError, ToolAuthRepository
from ..core import load_schema
from ..openapi_tool import TokenOpenApiTool

NOTION_TOOL_ID = "notion"
NOTION_VERSION = "2026-03-11"


class NotionTool(TokenOpenApiTool):
    id: str = NOTION_TOOL_ID
    name: str = "Notion"
    description: str = "Search and retrieve pages and content from Notion"
    config_schema: dict = load_schema(__file__)
    _api_url: str = "https://api.notion.com"
    _token_secret_key = "token"
    _auth_check_path = "/v1/users/me"

    def _invalid_token_message(self) -> str:
        return "Invalid Notion token"

    def _apply_token_to_headers(self, headers: dict, token: str) -> dict:
        headers["Authorization"] = f"Bearer {token}"
        headers["Notion-Version"] = NOTION_VERSION
        return headers

    def _on_unauthorized(self, e: HTTPStatusError) -> None:
        raise OwnerConfiguredToolAuthError() from e

    async def _resolve_token(self) -> str:
        if self._token:
            return self._token.strip()
        owner_id = self.agent.user_id
        if owner_id is None:
            raise OwnerConfiguredToolAuthError()
        token = await ToolAuthRepository(self.db).find_token(owner_id, self.agent.id, self.id)
        if not token:
            raise OwnerConfiguredToolAuthError()
        self._token = token.access_token
        return self._token.strip()
