import asyncio

from ..core import load_schema
from ..openapi_tool import OAuthOpenApiTool, OAuthToolConfig

AZURE_DEVOPS_TOOL_ID = "azure-devops"
_SWAGGER_URLS = (
    "https://github.com/MicrosoftDocs/vsts-rest-api-specs/blob/master/specification/core/7.2/core.json",
    "https://github.com/MicrosoftDocs/vsts-rest-api-specs/blob/master/specification/testPlan/7.2/testPlan.json",
    "https://github.com/MicrosoftDocs/vsts-rest-api-specs/blob/master/specification/wit/7.2/workItemTracking.json",
)


class AzureDevOpsTool(OAuthOpenApiTool):
    id: str = AZURE_DEVOPS_TOOL_ID
    name: str = "AzureDevOps"
    description: str = "Read user stories, manage test plan cases, and link tests to stories"
    config_schema: dict = load_schema(__file__)
    _body_content_types = ("application/json-patch+json", "application/json")

    async def _load_api_spec(self) -> dict:
        specs = await asyncio.gather(
            self._load_json("ado-api-spec-work-item-tracking.json"),
            self._load_json("ado-api-spec-test-plan.json"),
            self._load_json("ado-api-spec-core.json"),
        )
        merged = specs[0].copy()
        merged["paths"] = {}
        merged["components"] = {"schemas": {}}
        for spec in specs:
            merged["paths"].update(spec.get("paths", {}))
            merged["components"]["schemas"].update(
                spec.get("components", {}).get("schemas", {})
            )
        return merged

    async def _resolve_api_url(self) -> str:
        return "https://dev.azure.com"

    def _oauth_config(self) -> OAuthToolConfig:
        tenant_id = self.config["tenantId"]
        authority_base_url = f"https://login.microsoftonline.com/{tenant_id}"
        return OAuthToolConfig(
            authority_base_url=authority_base_url,
            authorize_path="/oauth2/v2.0/authorize",
            token_path="/oauth2/v2.0/token",
            scope=" ".join(f"499b84ac-1321-427f-aa17-267ca6975798/vso.{scope}" for scope in self.config["scope"]),
        )

    def _should_include_operation(self, path: str, method: str) -> bool:
        if path in {
            "/{organization}/_apis/projects",
            "/{organization}/_apis/projects/{projectId}",
            "/{organization}/_apis/projects/{projectId}/properties",
            "/{organization}/_apis/projects/{projectId}/teams",
            "/{organization}/_apis/projects/{projectId}/teams/{teamId}",
            "/{organization}/_apis/projects/{projectId}/teams/{teamId}/members",
            "/{organization}/_apis/teams",
        }:
            return method == "get"
        return path in {
            "/{organization}/{project}/_apis/wit/workitems",
            "/{organization}/{project}/_apis/wit/workitems/${type}",
            "/{organization}/{project}/_apis/wit/workitems/{id}",
            "/{organization}/{project}/_apis/wit/workitemsbatch",
            "/{organization}/{project}/_apis/wit/workitemsdelete",
            "/{organization}/{project}/_apis/wit/workitemtypes",
            "/{organization}/{project}/_apis/wit/workitemtypes/{type}",
            "/{organization}/{project}/_apis/wit/workitemtypes/{type}/fields",
            "/{organization}/{project}/_apis/wit/workitemtypes/{type}/fields/{field}",
            "/{organization}/{project}/_apis/wit/workitemtypes/{type}/states",
            "/{organization}/{project}/_apis/wit/fields",
            "/{organization}/{project}/_apis/wit/fields/{fieldNameOrRefName}",
            "/{organization}/{project}/{team}/_apis/wit/wiql",
            "/{organization}/{project}/{team}/_apis/wit/wiql/{id}",
            "/{organization}/_apis/wit/workitemrelationtypes",
            "/{organization}/_apis/wit/workitemrelationtypes/{relation}",
            "/{organization}/_apis/wit/artifactlinktypes",
            "/{organization}/_apis/testplan/suites",
            "/{organization}/{project}/_apis/testplan/configurations",
            "/{organization}/{project}/_apis/testplan/configurations/{testConfigurationId}",
            "/{organization}/{project}/_apis/testplan/plans",
            "/{organization}/{project}/_apis/testplan/plans/{planId}",
            "/{organization}/{project}/_apis/testplan/Plans/{planId}/suites",
            "/{organization}/{project}/_apis/testplan/Plans/{planId}/suites/{suiteId}",
            "/{organization}/{project}/_apis/testplan/Plans/{planId}/Suites/{suiteId}/TestCase",
            "/{organization}/{project}/_apis/testplan/Plans/{planId}/Suites/{suiteId}/TestCase/{testCaseId}",
        }
