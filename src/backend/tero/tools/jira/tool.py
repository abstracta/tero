from http import HTTPMethod
import logging
from typing import Any, Optional

from sqlmodel import Field, SQLModel, and_, select, delete
from sqlmodel.ext.asyncio.session import AsyncSession

from ...core.repos import scalar
from ..auth import ToolAuthCallback
from ..core import load_schema
from ..openapi_tool import OAuthOpenApiTool, OAuthToolConfig


logger = logging.getLogger(__name__)
JIRA_TOOL_ID = "jira"
SWAGGER_URL = "https://developer.atlassian.com/cloud/jira/platform/swagger-v3.v3.json"
JIRA_BASE_API_URL = "https://api.atlassian.com"


class JiraToolConfig(SQLModel, table=True):
    __tablename__ : Any = "jira_tool_config"
    agent_id: int = Field(primary_key=True)
    cloud_id: str


class JiraToolConfigRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_agent_id(self, agent_id: int) -> Optional[JiraToolConfig]:
        stmt = select(JiraToolConfig).where(JiraToolConfig.agent_id == agent_id)
        result = await self._db.exec(stmt)
        return result.first()
    
    async def save(self, config: JiraToolConfig):
        self._db.add(config)
        await self._db.commit()

    async def delete(self, agent_id: int):
        stmt = scalar(delete(JiraToolConfig).where(and_(JiraToolConfig.agent_id == agent_id)))
        await self._db.exec(stmt)
        await self._db.commit()


class JiraTool(OAuthOpenApiTool):
    id: str = JIRA_TOOL_ID
    name: str = "Jira"
    description: str = "Manage issues and track project activity"
    config_schema: dict = load_schema(__file__)

    def _oauth_config(self) -> OAuthToolConfig:
        return OAuthToolConfig(
            authority_base_url="https://auth.atlassian.com",
            authorize_path="/authorize",
            token_path="/oauth/token",
            scope=" ".join(self.config["scope"]),
        )

    async def _resolve_api_url(self) -> str:
        cloud_id = await self._find_cloud_id()
        return f"{JIRA_BASE_API_URL}/ex/jira/{cloud_id}"

    async def teardown(self):
        await super().teardown()
        await JiraToolConfigRepository(self.db).delete(self.agent.id)

    async def auth(self, auth_callback: ToolAuthCallback):
        await super().auth(auth_callback)
        await JiraToolConfigRepository(self.db).delete(self.agent.id)

    async def _find_cloud_id(self):
        repo = JiraToolConfigRepository(self.db)
        jira_config = await repo.find_by_agent_id(self.agent.id)
        if jira_config:
            return jira_config.cloud_id
        resp = await self._invoke_rest_api(HTTPMethod.GET, f"{JIRA_BASE_API_URL}/oauth/token/accessible-resources")
        ret = next(resource["id"] for resource in resp)
        await repo.save(JiraToolConfig(agent_id=self.agent.id, cloud_id=ret))
        return ret

    async def _load_api_spec(self) -> dict:
        ret = await super()._load_api_spec()
        schemas = ret["components"]["schemas"]
        # using simplified schema instead of the original one from https://unpkg.com/@atlaskit/adf-schema@49.0.1/dist/json-schema/v1/full.json 
        # since original schema is huge, consuming time, tokens and making llm confused with so much information
        # additionally, just having version after content in doc_node makes the llm to generate a call without the version attribute, which makes the request to fail
        doc_node_schema = await self._load_json("simplified-doc-node-schema.json")
        schemas.update(doc_node_schema["definitions"])
        return ret

    # there is a limitation of up to 128 functions that can be passed to OpenAI, and JIRA API has more than 590 methods. This method filters the most common and used ones.
    def _should_include_operation(self, path: str, method: str) -> bool:
        base_path = "/rest/api/3"
        issues_path = f"{base_path}/issue"
        issue_path = f"{issues_path}/{{issueIdOrKey}}"
        comments_path = f"{issue_path}/comment"
        properties_path = f"{issue_path}/properties"
        search_path = f"{base_path}/search"
        projects_path = f"{base_path}/project"
        project_path = f"{projects_path}/{{projectIdOrKey}}"
        return path in [
            issues_path, issue_path, f"{issue_path}/assignee", f"{issue_path}/changelog", 
            comments_path, f"{comments_path}/{{id}}", properties_path, f"{properties_path}/{{propertyKey}}", f"{issue_path}/transitions", 
            f"{search_path}/approximate-count", f"{search_path}/jql", f"{projects_path}/search",
            f"{base_path}/myself", f"{base_path}/users/search"] \
                or (method == "get" and path in [f"{project_path}", f"{project_path}/statuses"])
    
    def _handle_schema_without_type(self, schema: dict, schemas: dict, refs: set) -> None:
        # Fix Jira schema which does not properly define the schema for comments.
        if "Atlassian Document Format" in schema.get("description", ""):
            self._refactor_ref(schema, "doc_node", schemas, refs)
