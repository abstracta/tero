import asyncio
from collections import Counter
from collections.abc import Awaitable, Iterator
from html import escape
from http import HTTPMethod
from typing import Any, Optional
from urllib.parse import quote

from langchain_core.callbacks import Callbacks
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field, model_validator

from ...core.env import env
from ...files.domain import File
from ...threads.repos import ThreadMessageFileRepository
from ..core import StatusUpdateCallbackHandler, UserFacingToolError, load_schema
from ..oauth_tool import OAuthToolConfig, build_azure_oauth_config
from ..openapi_tool import OAuthOpenApiTool

AZURE_DEVOPS_TOOL_ID = "azure-devops"
_TEST_SUITES_FOR_PLAN_PATH = "/{organization}/{project}/_apis/testplan/Plans/{planId}/suites"
_TEST_CASES_FOR_SUITE_PATH = "/{organization}/{project}/_apis/testplan/Plans/{planId}/Suites/{suiteId}/TestCase"
_WORK_ITEMS_DELETE_PATH = "/{organization}/{project}/_apis/wit/workitemsdelete"
_READ_POST_PATHS = frozenset({
    "/{organization}/{project}/_apis/wit/workitemsbatch",
    "/{organization}/{project}/{team}/_apis/wit/wiql",
})
# Maximum is 1000 when detailsToInclude is None and 200 otherwise
_RESULTS_WITH_WORK_ITEMS_PAGE_SIZE = 200
_WORK_ITEMS_BATCH_SIZE = 200
# WIQL query text has a size limit, so ids are split across queries
_WIQL_IDS_CHUNK_SIZE = 500
# Self-imposed cap to avoid overloading the API when a query fans out into many chunks
_MAX_CONCURRENT_REQUESTS = 8
_BASE_WORK_ITEM_FIELDS = ("System.Id", "System.Title", "System.State", "System.WorkItemType")


class AzureDevOpsTool(OAuthOpenApiTool):
    id: str = AZURE_DEVOPS_TOOL_ID
    name: str = "AzureDevOps"
    description: str = "Read user stories, manage test plan cases, link tests to stories, and get suite test results"
    config_schema: dict = load_schema(__file__)
    _body_content_types = ("application/json-patch+json", "application/json")
    _oauth_scopes_path = "security.oauth2"

    @model_validator(mode="after")
    def hide_oauth_config_when_env_set(self):
        if self._env_oauth_configured():
            for key in ("tenantId", "clientId", "clientSecret"):
                self.config_schema["properties"].pop(key, None)
                if key in self.config_schema.get("required", []):
                    self.config_schema["required"].remove(key)
        return self

    def _env_oauth_configured(self) -> bool:
        return bool(self._env_tenant_id() and super()._env_oauth_configured())

    def _env_tenant_id(self) -> Optional[str]:
        return env.azure_devops_tool_tenant_id

    def _env_client_id(self) -> Optional[str]:
        return env.azure_devops_tool_client_id

    def _env_client_secret(self) -> Optional[str]:
        return env.azure_devops_tool_client_secret.get_secret_value() if env.azure_devops_tool_client_secret else None

    async def _load_api_spec(self) -> dict:
        specs = await asyncio.gather(
            # https://github.com/MicrosoftDocs/vsts-rest-api-specs/blob/master/specification/wit/7.2/workItemTracking.json
            self._load_json("ado-api-spec-work-item-tracking.json"),
            # https://github.com/MicrosoftDocs/vsts-rest-api-specs/blob/master/specification/testPlan/7.2/testPlan.json
            self._load_json("ado-api-spec-test-plan.json"),
            # https://github.com/MicrosoftDocs/vsts-rest-api-specs/blob/master/specification/test/7.2/test.json
            self._load_json("ado-api-spec-test.json"),
            # https://github.com/MicrosoftDocs/vsts-rest-api-specs/blob/master/specification/core/7.2/core.json
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
        self._allow_untyped_json_patch_values(merged)
        return merged

    def _allow_untyped_json_patch_values(self, spec: dict) -> None:
        # spec types patch value as object; work item fields need strings
        value_schema = (
            spec.get("components", {})
            .get("schemas", {})
            .get("JsonPatchOperation", {})
            .get("properties", {})
            .get("value")
        )
        if value_schema:
            value_schema.pop("type", None)
            value_schema.pop("properties", None)

    async def _resolve_api_url(self) -> str:
        return "https://dev.azure.com"

    async def build_langchain_tools(self) -> list[BaseTool]:
        tools = await super().build_langchain_tools()
        scopes = self._configured_oauth_scopes()
        if "vso.work" in scopes:
            tools.append(AzureDevOpsRunExtendedWiqlTool(azure_devops=self))
        if {"vso.test", "vso.work"} <= scopes:
            tools.append(AzureDevOpsGetSuiteTestResultsTool(azure_devops=self))
        if "vso.work_write" in scopes:
            tools.append(AzureDevOpsAttachWorkItemFileTool(azure_devops=self))
        return tools

    def _configured_oauth_scopes(self) -> set[str]:
        scopes: set[str] = set()
        for scope in self.config.get("scope", []):
            scopes.add(f"vso.{scope}")
            if scope.endswith("_write"):
                scopes.add(f"vso.{scope.removesuffix('_write')}")
        return scopes

    def _oauth_config(self) -> OAuthToolConfig:
        tenant_id = self._env_tenant_id() or self.config["tenantId"]
        scope = " ".join(f"499b84ac-1321-427f-aa17-267ca6975798/vso.{s}" for s in self.config["scope"])
        return build_azure_oauth_config(tenant_id, scope)

    def _should_include_operation(self, path: str, method: str) -> bool:
        if path in {
            "/{organization}/_apis/projects",
            "/{organization}/_apis/projects/{projectId}",
            "/{organization}/_apis/projects/{projectId}/properties",
            "/{organization}/_apis/projects/{projectId}/teams",
            "/{organization}/_apis/projects/{projectId}/teams/{teamId}",
            "/{organization}/_apis/projects/{projectId}/teams/{teamId}/members",
            "/{organization}/_apis/teams",
            "/{organization}/{project}/_apis/test/runs",
            "/{organization}/{project}/_apis/test/runs/{runId}",
            "/{organization}/{project}/_apis/test/Runs/{runId}/results",
            "/{organization}/{project}/_apis/test/runs/{runId}/Statistics",
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
            _TEST_SUITES_FOR_PLAN_PATH,
            "/{organization}/{project}/_apis/testplan/Plans/{planId}/suites/{suiteId}",
            _TEST_CASES_FOR_SUITE_PATH,
            f"{_TEST_CASES_FOR_SUITE_PATH}/{{testCaseId}}",
            "/{organization}/{project}/_apis/testplan/testcases/{testCaseId}",
            "/{organization}/{project}/_apis/test/Plans/{planId}/Suites/{suiteId}/points",
            "/{organization}/{project}/_apis/test/Runs/{runId}/results/{testCaseResultId}",
        }

    def _synthetic_parameters(self, path: str, method: str) -> list[dict]:
        if self._is_create(path, method):
            return [self._full_output_parameter(
                "Rarely needed, leave unset. By default only the id of the created resource is returned. "
                "Set to true only if you need the full API response; it is verbose."
            )]
        if self._is_update_or_delete(path, method):
            return [self._full_output_parameter(
                "Rarely needed, leave unset. By default the response is empty. "
                "Set to true only if you need the full API response; it is verbose."
            )]
        if method != "get":
            return []
        elif path == _TEST_SUITES_FOR_PLAN_PATH:
            return [self._full_output_parameter(
                "Rarely needed, leave unset. By default each suite includes id, name, "
                "parentSuite, project, plan, hasChildren and children, which is enough for "
                "most tasks. Set to true only if you need a suite field not listed above; "
                "the raw response is verbose."
            )]
        elif path == _TEST_CASES_FOR_SUITE_PATH:
            return [
                {
                    "name": "titleContains",
                    "in": self._QUERY_LOCATION,
                    "description": "If set, only test cases whose title contains this string are returned "
                                   "(case-insensitive). Omit to return all test cases.",
                    "schema": {"type": "string"},
                }
            ]
        else:
            return []

    def _is_create(self, path: str, method: str) -> bool:
        return method == "post" and path not in _READ_POST_PATHS and path != _WORK_ITEMS_DELETE_PATH

    def _full_output_parameter(self, description: str) -> dict:
        return {
            "name": "fullOutput",
            "in": self._QUERY_LOCATION,
            "description": description,
            "schema": {"type": "boolean", "default": False},
        }

    def _is_update_or_delete(self, path: str, method: str) -> bool:
        return method in {"put", "patch", "delete"} or (method == "post" and path == _WORK_ITEMS_DELETE_PATH)

    async def _execute_operation(
        self,
        path: str,
        method: str,
        url: str,
        query: Optional[dict],
        header: Optional[dict],
        body: Optional[Any],
        body_content_type: Optional[str],
        synthetic_params: dict,
    ) -> Any:
        title_contains = synthetic_params.get("titleContains")
        if method == "get" and path == _TEST_CASES_FOR_SUITE_PATH and isinstance(title_contains, str):
            title_contains = title_contains.strip()
            if title_contains:
                return await self._list_suite_test_cases_by_title(url, dict(query or {}), header, title_contains.casefold())
        response, headers = await self._invoke_rest_api_with_headers(
            method, url, query, header, body, body_content_type
        )
        return self._with_continuation_token(
            self._transform_response(path, method, response, synthetic_params),
            headers.get("x-ms-continuationtoken"),
        )

    async def _list_suite_test_cases_by_title(
        self, url: str, query: dict, header: Optional[dict], title_contains: str
    ) -> dict:
        filtered = []
        page_size = 0
        token = None
        template = {}
        first_page = True
        while first_page or (token and (not page_size or len(filtered) < page_size)):
            first_page = False
            params = dict(query)
            if token:
                params["continuationToken"] = token
            response, headers = await self._invoke_rest_api_with_headers(
                HTTPMethod.GET, url, params, header
            )
            items = _response_values(response)
            if isinstance(response, dict):
                template = response
            # ADO does not advertise page size; use the unfiltered page we just received
            if not page_size:
                page_size = len(items)
            filtered.extend(
                test_case for test_case in items
                if title_contains in ((test_case.get("workItem") or {}).get("name") or "").casefold()
            )
            token = headers.get("x-ms-continuationtoken")
        result = {**template, "value": filtered, "count": len(filtered)}
        result.pop("continuationToken", None)
        return self._with_continuation_token(result, token)

    def _with_continuation_token(self, response: Any, token: Optional[str]) -> Any:
        if not token:
            return response
        if isinstance(response, dict):
            return {**response, "continuationToken": token}
        if isinstance(response, list):
            return {"count": len(response), "value": response, "continuationToken": token}
        return response

    def _transform_response(self, path: str, method: str, response: Any, synthetic_params: dict) -> Any:
        if not synthetic_params.get("fullOutput"):
            if self._is_create(path, method):
                return self._created_resource_id(response)
            if self._is_update_or_delete(path, method):
                return None
            if method == "get" and path == _TEST_SUITES_FOR_PLAN_PATH and isinstance(response, dict):
                return {**response, "value": [self._slim_suite(suite) for suite in response.get("value", [])]}
        return response

    def _created_resource_id(self, response: Any) -> Any:
        if isinstance(response, list):
            return [self._created_resource_id(item) for item in response]
        if not isinstance(response, dict):
            return response
        if "id" in response:
            return {"id": response["id"]}
        work_item = response.get("workItem")
        if isinstance(work_item, dict) and "id" in work_item:
            return {"id": work_item["id"]}
        if "referenceName" in response:
            return {"id": response["referenceName"]}
        return response

    def _slim_suite(self, suite: dict) -> dict:
        slim = {key: suite[key] for key in ("id", "name", "hasChildren") if key in suite}
        for ref in ("parentSuite", "project", "plan"):
            if suite.get(ref):
                slim[ref] = {key: suite[ref].get(key) for key in ("id", "name")}
        if suite.get("children"):
            slim["children"] = [self._slim_suite(child) for child in suite["children"]]
        return slim


def _response_values(response: Any) -> list:
    return response if isinstance(response, list) else (response or {}).get("value") or []


class ExtendedWiqlArgs(BaseModel):
    organization: str = Field(description="Azure DevOps organization name")
    project: str = Field(description="Project name or ID")
    queryId: str = Field(description="ID of the saved work item query to run")
    additionalFilter: Optional[str] = Field(
        default=None,
        description="WIQL condition to further filter the query results, without SELECT or WHERE. "
                    "Reference fields by reference name between brackets. Example: "
                    "\"[System.WorkItemType] = 'User Story' OR [System.WorkItemType] = 'RequestEN'\"",
    )
    lookupRelationFilter: Optional[str] = Field(
        default=None,
        description="WIQL condition selecting which related work items to include, without SELECT or "
                    "WHERE. Fields refer to the related work item, so prefix them with [Target]. Example: "
                    "\"[Target].[System.WorkItemType] = 'Bug'\". Leave unset to skip relations.",
    )
    extraFields: Optional[str] = Field(
        default=None,
        description="Comma separated field reference names to return besides id, title, state and work "
                    "item type, both for the base work items and the related ones. Example: "
                    "\"Microsoft.VSTS.Common.Severity,Custom.NewSeverity\"",
    )
    # default keeps a query without an explicit top from flooding the model context, and fits a
    # single work items batch
    top: int = Field(
        default=200,
        gt=0,
        description="Max number of work items to return, applied after additionalFilter. The response "
                    "reports totalCount and truncated, so raise it only when more items are needed",
    )


class AzureDevOpsRunExtendedWiqlTool(BaseTool):
    name: str = "AzureDevOps-runExtendedWiql"
    description: str = (
        "AzureDevOps tool that runs a saved work item query and returns its work items already resolved "
        "as id, Title, State and WorkItemType, optionally filtered, with extra fields and with their "
        "related work items. Prefer it over combining Wiql_QueryById with WorkItems_GetWorkItemsBatch."
    )
    args_schema: Optional[ArgsSchema] = ExtendedWiqlArgs
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]
    azure_devops: AzureDevOpsTool

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def _arun(self, **kwargs: Any) -> dict:
        args = ExtendedWiqlArgs.model_validate(kwargs)
        ado = self.azure_devops
        base = f"{ado._api_url}/{args.organization}/{args.project}/_apis"

        ids = await self._run_saved_query(ado, base, args.queryId)
        if args.additionalFilter:
            ids = await self._apply_filter(ado, base, ids, args.additionalFilter)
        # top is applied here, and not by the saved query, so filtering does not discard matches that
        # the query left out of its first results, and so totalCount reflects every match
        total_count = len(ids)
        ids = ids[: args.top]
        # relations of a work item, as (rel, related work item id) keyed by work item id
        relations: dict[int, list[tuple[str, int]]] = (
            await self._lookup_relations(ado, base, ids, args.lookupRelationFilter)
            if args.lookupRelationFilter else {}
        )

        fields = self._requested_fields(args.extraFields)
        aliases = self._field_aliases(fields)
        related_ids = [rel_id for rels in relations.values() for _, rel_id in rels]
        items = await _fetch_work_items(ado, base, list(dict.fromkeys([*ids, *related_ids])), fields)

        work_items = []
        for work_item_id in ids:
            item = items.get(work_item_id)
            if not item:
                continue
            projected = self._project(item, aliases)
            if args.lookupRelationFilter:
                projected["relations"] = [
                    {"rel": rel, **self._project(items[rel_id], aliases)}
                    for rel, rel_id in relations.get(work_item_id, [])
                    if rel_id in items
                ]
            work_items.append(projected)
        return {
            "count": len(work_items),
            "totalCount": total_count,
            "truncated": total_count > len(ids),
            "workItems": work_items,
        }

    async def _run_saved_query(self, ado: AzureDevOpsTool, base: str, query_id: str) -> list[int]:
        result = await ado._invoke_rest_api(
            HTTPMethod.GET, f"{base}/wit/wiql/{query_id}", params={"api-version": "7.2-preview.2"}
        )
        return self._result_ids(result)

    def _result_ids(self, result: Any) -> list[int]:
        result = result or {}
        ids = [item["id"] for item in result.get("workItems") or [] if item.get("id")]
        for link in result.get("workItemRelations") or []:
            ids.extend(ref["id"] for ref in (link.get("source"), link.get("target")) if ref and ref.get("id"))
        return list(dict.fromkeys(ids))

    async def _apply_filter(
        self, ado: AzureDevOpsTool, base: str, ids: list[int], condition: str
    ) -> list[int]:
        results = await _gather_limited([
            self._query_wiql(
                ado,
                base,
                f"SELECT [System.Id] FROM WorkItems"
                f" WHERE [System.Id] IN ({self._id_list(chunk)}) AND ({condition})",
            )
            for chunk in _chunked(ids, _WIQL_IDS_CHUNK_SIZE)
        ])
        matched = {work_item_id for result in results for work_item_id in self._result_ids(result)}
        return [work_item_id for work_item_id in ids if work_item_id in matched]

    async def _query_wiql(self, ado: AzureDevOpsTool, base: str, query: str) -> Any:
        return await ado._invoke_rest_api(
            HTTPMethod.POST,
            f"{base}/wit/wiql",
            params={"api-version": "7.2-preview.2"},
            body={"query": query},
            body_content_type="application/json",
        )

    def _id_list(self, ids: list[int]) -> str:
        return ",".join(str(work_item_id) for work_item_id in ids)

    async def _lookup_relations(
        self, ado: AzureDevOpsTool, base: str, ids: list[int], condition: str
    ) -> dict[int, list[tuple[str, int]]]:
        # MayContain keeps base work items that have no matching relation. Note WIQL link queries only
        # resolve links within the project, so relations to other projects are not returned.
        results = await _gather_limited([
            self._query_wiql(
                ado,
                base,
                f"SELECT [System.Id] FROM workitemLinks"
                f" WHERE ([Source].[System.Id] IN ({self._id_list(chunk)})) AND ({condition})"
                f" MODE (MayContain)",
            )
            for chunk in _chunked(ids, _WIQL_IDS_CHUNK_SIZE)
        ])
        base_ids = set(ids)
        ret: dict[int, list[tuple[str, int]]] = {}
        for result in results:
            for link in (result or {}).get("workItemRelations") or []:
                source, target = link.get("source"), link.get("target")
                # rows without source are the base work items themselves, not links
                if not source or not target or source.get("id") not in base_ids:
                    continue
                ret.setdefault(source["id"], []).append((link.get("rel"), target["id"]))
        return ret

    def _requested_fields(self, extra_fields: Optional[str]) -> list[str]:
        fields: list[str] = list(_BASE_WORK_ITEM_FIELDS)
        for field in (extra_fields or "").split(","):
            field = field.strip()
            if field and field not in fields:
                fields.append(field)
        return fields

    def _field_aliases(self, fields: list[str]) -> dict[str, str]:
        short_names = [field.rsplit(".", 1)[-1] for field in fields]
        return {
            field: short if short_names.count(short) == 1 else field
            for field, short in zip(fields, short_names)
        }

    def _project(self, item: dict, aliases: dict[str, str]) -> dict:
        item_fields = item.get("fields") or {}
        ret = {"id": item.get("id")}
        for field, alias in aliases.items():
            value = item_fields.get(field)
            if field != "System.Id" and value is not None:
                ret[alias] = value.get("displayName", value) if isinstance(value, dict) else value
        return ret


async def _gather_limited[T](coroutines: list[Awaitable[T]]) -> list[T]:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    async def run(coroutine: Awaitable[T]) -> T:
        async with semaphore:
            return await coroutine

    return list(await asyncio.gather(*(run(coroutine) for coroutine in coroutines)))


def _chunked[T](items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _fetch_work_items(
    ado: AzureDevOpsTool, base: str, ids: list[int], fields: list[str]
) -> dict[int, dict]:
    responses = await _gather_limited([
        ado._invoke_rest_api(
            HTTPMethod.POST,
            f"{base}/wit/workitemsbatch",
            params={"api-version": "7.2-preview.1"},
            # fields and $expand are mutually exclusive, so relations are resolved with a link query
            body={"ids": chunk, "fields": fields, "errorPolicy": "omit"},
            body_content_type="application/json",
        )
        for chunk in _chunked(ids, _WORK_ITEMS_BATCH_SIZE)
    ])
    return {
        item["id"]: item
        for response in responses
        for item in (response or {}).get("value") or []
        if item.get("id")
    }


class SuiteTestResultsArgs(BaseModel):
    organization: str = Field(description="Azure DevOps organization name")
    project: str = Field(description="Project name or ID")
    planId: int = Field(description="Test plan ID")
    suiteId: int = Field(description="Test suite ID")


class AzureDevOpsGetSuiteTestResultsTool(BaseTool):
    name: str = "AzureDevOps-getSuiteTestResults"
    description: str = (
        "AzureDevOps tool that returns the test cases of a suite and of all its child suites, as a flat "
        "list where each test case has its suiteId and suitePath, outcome, runBy, runId and associated "
        "bugs (id, name, severity), plus a byOutcome count of test results in each status."
    )
    args_schema: Optional[ArgsSchema] = SuiteTestResultsArgs
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]
    azure_devops: AzureDevOpsTool

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def _arun(self, **kwargs: Any) -> dict:
        args = SuiteTestResultsArgs.model_validate(kwargs)
        ado = self.azure_devops
        base = f"{ado._api_url}/{args.organization}/{args.project}/_apis"

        suite, points = await asyncio.gather(
            self._find_suite(ado, base, args.planId, args.suiteId),
            self._list_suite_points(ado, base, args.planId, args.suiteId),
        )
        points_by_suite: dict[int, list[dict]] = {}
        for point in points:
            if suite_id := self._ref_id(point.get("testSuite")):
                points_by_suite.setdefault(suite_id, []).append(point)

        run_ids = sorted({
            rid
            for points in points_by_suite.values()
            for point in points if (rid := self._point_results(point).get("lastTestRunId"))
        })
        # a run is shared by the suites it covers, so its results are listed once for the whole tree
        bugs_by_result: dict[tuple[int, int], list[dict]] = {}
        for run_bugs in await _gather_limited([self._list_run_bugs(ado, base, rid) for rid in run_ids]):
            bugs_by_result.update(run_bugs)

        # suites are walked in tree order, so the flat list keeps sibling and child suites together
        rows = [
            (suite_id, path, *self._point_row(point, bugs_by_result))
            for suite_id, path in self._suite_paths(suite).items()
            for point in points_by_suite.get(suite_id, [])
        ]
        bug_ids = {int(bug["id"]) for *_, bugs in rows for bug in bugs if bug.get("id")}
        severity_by_id = await self._fetch_bug_severities(ado, base, sorted(bug_ids))

        test_cases = [
            self._test_case(suite_id, path, point, run_id, bugs, severity_by_id)
            for suite_id, path, point, run_id, bugs in rows
        ]
        return {
            "count": len(test_cases),
            "byOutcome": dict(Counter(case.get("outcome") or "None" for case in test_cases)),
            "testCases": test_cases,
        }

    async def _find_suite(self, ado: AzureDevOpsTool, base: str, plan_id: int, suite_id: int) -> dict:
        # the plan wide list is used, instead of getting the suite by id, because only this endpoint
        # supports asTreeView, which nests the whole hierarchy in a single call. Getting the suite by id
        # only takes expand=children, which returns the direct children, so collecting all descendants
        # would need one request per suite with children.
        suites = await _list_paged(
            ado,
            f"{base}/testplan/Plans/{plan_id}/suites",
            {"expand": "children", "asTreeView": "true", "api-version": "7.2-preview.1"},
        )
        suite = next((s for s in self._walk_suites(suites) if s.get("id") == suite_id), None)
        if not suite:
            raise UserFacingToolError(f"Test suite {suite_id} not found in test plan {plan_id}.")
        return suite

    def _walk_suites(self, suites: list[dict]) -> Iterator[dict]:
        for suite in suites:
            yield suite
            yield from self._walk_suites(suite.get("children") or [])

    async def _list_suite_points(
        self, ado: AzureDevOpsTool, base: str, plan_id: int, suite_id: int
    ) -> list[dict]:
        # isRecursive brings the points of the whole subtree in one listing, and each point carries the
        # testSuite it belongs to, so they can be grouped back into the tree without a call per suite.
        # Only the testplan API takes isRecursive, hence the different point shape from the test API.
        return await _list_paged(
            ado,
            f"{base}/testplan/Plans/{plan_id}/Suites/{suite_id}/TestPoint",
            {"isRecursive": "true", "includePointDetails": "true", "api-version": "7.2-preview.2"},
        )

    def _ref_id(self, ref: Any) -> Optional[int]:
        return int((ref or {}).get("id") or 0) or None

    def _point_results(self, point: dict) -> dict:
        return point.get("results") or {}

    async def _list_run_bugs(
        self, ado: AzureDevOpsTool, base: str, run_id: int
    ) -> dict[tuple[int, int], list[dict]]:
        bugs: dict[tuple[int, int], list[dict]] = {}
        skip = 0
        while True:
            resp = await ado._invoke_rest_api(
                HTTPMethod.GET,
                f"{base}/test/Runs/{run_id}/results",
                params={
                    "detailsToInclude": "workItems",
                    "$skip": str(skip),
                    "$top": str(_RESULTS_WITH_WORK_ITEMS_PAGE_SIZE),
                    "api-version": "7.2-preview.6",
                },
            )
            results = resp if isinstance(resp, list) else (resp or {}).get("value", [])
            bugs.update({
                (run_id, int(r["id"])): r.get("associatedBugs") or []
                for r in results
                if r.get("id") is not None
            })
            if len(results) < _RESULTS_WITH_WORK_ITEMS_PAGE_SIZE:
                return bugs
            skip += _RESULTS_WITH_WORK_ITEMS_PAGE_SIZE

    def _point_row(
        self, point: dict, bugs_by_result: dict[tuple[int, int], list[dict]]
    ) -> tuple[dict, Optional[int], list[dict]]:
        results = self._point_results(point)
        run_id, result_id = results.get("lastTestRunId"), results.get("lastResultId")
        bugs = bugs_by_result.get((run_id, result_id), []) if run_id and result_id else []
        return point, run_id, bugs

    def _suite_paths(self, suite: dict, parents: tuple[str, ...] = ()) -> dict[int, str]:
        # each suite id mapped to its name path, so a flat test case still shows where it sits
        path = (*parents, suite.get("name") or "")
        paths = {suite["id"]: " > ".join(path)} if suite.get("id") else {}
        for child in suite.get("children") or []:
            paths.update(self._suite_paths(child, path))
        return paths

    async def _fetch_bug_severities(
        self, ado: AzureDevOpsTool, base: str, bug_ids: list[int]
    ) -> dict[int, Any]:
        items = await _fetch_work_items(
            ado, base, bug_ids, ["System.Id", "Microsoft.VSTS.Common.Severity"]
        )
        return {
            bug_id: (item.get("fields") or {}).get("Microsoft.VSTS.Common.Severity")
            for bug_id, item in items.items()
        }

    def _test_case(
        self,
        suite_id: int,
        suite_path: str,
        point: dict,
        run_id: Optional[int],
        bugs: list[dict],
        severity_by_id: dict[int, Any],
    ) -> dict:
        return {
            "suiteId": suite_id,
            "suitePath": suite_path,
            "testCaseId": (point.get("testCaseReference") or {}).get("id"),
            "testCaseName": (point.get("testCaseReference") or {}).get("name"),
            "outcome": self._point_results(point).get("outcome"),
            "runBy": ((self._point_results(point).get("lastResultDetails") or {}).get("runBy") or {})
            .get("displayName"),
            "runId": run_id,
            "bugs": [
                {
                    "id": bug.get("id"),
                    "name": bug.get("name"),
                    "severity": severity_by_id.get(int(bug["id"])) if bug.get("id") else None,
                }
                for bug in bugs
            ],
        }


async def _list_paged(ado: AzureDevOpsTool, url: str, params: dict) -> list[dict]:
    # list endpoints cap the page size and return x-ms-continuationtoken while more items remain
    values: list[dict] = []
    token: Optional[str] = None
    while True:
        response, headers = await ado._invoke_rest_api_with_headers(
            HTTPMethod.GET,
            url,
            params={**params, "continuationToken": token} if token else params,
        )
        values.extend(_response_values(response))
        token = headers.get("x-ms-continuationtoken")
        if not token:
            return values


class AttachWorkItemFileArgs(BaseModel):
    organization: str = Field(description="Azure DevOps organization name")
    project: str = Field(description="Project name or ID")
    workItemId: int = Field(description="Work item ID to add the file to")
    target: str = Field(
        description=(
            "Where to place the file. Use 'attachments' for the Attachments tab (any file). "
            "Otherwise pass a work item field referenceName (for example 'Microsoft.VSTS.TCM.ReproSteps') "
            "to also embed an image in that HTML field; only png, jpg or jpeg are allowed then. "
            "The field must have type 'html'; this can be provided by user or system prompt "
            "or looked up via Fields_List, Fields_Get or WorkItemTypesField_Get. "
        )
    )
    fileName: Optional[str] = Field(
        default=None,
        description=(
            "Name of the file uploaded in this chat. Omit to use the most recently uploaded "
            "file. Required when the chat has several files and you need a specific one."
        ),
    )


class AzureDevOpsAttachWorkItemFileTool(BaseTool):
    name: str = "AzureDevOps-attachWorkItemFile"
    description: str = (
        "AzureDevOps tool that uploads a file from this chat to a work item. "
        "Use it for the Attachments tab with any file, or to embed an image in an HTML work item field. "
        "This tool attaches the file, and when a field is specified inserts an img in that field."
    )
    args_schema: Optional[ArgsSchema] = AttachWorkItemFileArgs
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]
    azure_devops: AzureDevOpsTool

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def _arun(self, **kwargs: Any) -> dict:
        args = AttachWorkItemFileArgs.model_validate(kwargs)
        ado = self.azure_devops
        base = f"{ado._api_url}/{args.organization}/{args.project}/_apis"
        html_field = None if args.target == "attachments" else args.target
        file = await self._resolve_chat_file(args.fileName)
        if html_field and not self._is_image_file(file):
            raise UserFacingToolError(f"'{file.name}' is not a supported image (png, jpg or jpeg) and cannot be embedded in a work item field.")

        work_item_api_version = "7.2-preview.3"
        current_html = None
        if html_field:
            work_item = await ado._invoke_rest_api(
                HTTPMethod.GET,
                f"{base}/wit/workitems/{args.workItemId}",
                params={"fields": html_field, "api-version": work_item_api_version},
            )
            current_html = (work_item or {}).get("fields", {}).get(html_field)

        attachment = await ado._invoke_rest_api(
            HTTPMethod.POST,
            f"{base}/wit/attachments",
            params={"fileName": file.name, "api-version": "7.2-preview.4"},
            body=file.content,
            body_content_type="application/octet-stream",
            timeout=60.0,
        )
        attachment_url = (attachment or {}).get("url")
        if not attachment_url:
            raise UserFacingToolError("Azure DevOps did not return an attachment URL.")

        ops: list[dict] = [{
            "op": "add",
            "path": "/relations/-",
            "value": {
                "rel": "AttachedFile",
                "url": attachment_url,
                "attributes": {"comment": file.name},
            },
        }]
        if html_field:
            img_html = self._image_html(attachment_url, file.name)
            ops.insert(0, {
                "op": "replace" if current_html is not None else "add",
                "path": f"/fields/{html_field}",
                "value": f"{current_html}{img_html}" if current_html else img_html,
            })

        await ado._invoke_rest_api(
            HTTPMethod.PATCH,
            f"{base}/wit/workitems/{args.workItemId}",
            params={"api-version": work_item_api_version},
            body=ops,
            body_content_type="application/json-patch+json",
        )
        return {
            "workItemId": args.workItemId,
            "fileName": file.name,
            "attachmentId": attachment.get("id"),
            "attachmentUrl": attachment_url,
            "target": "attachments" if html_field is None else html_field,
        }

    async def _resolve_chat_file(self, file_name: Optional[str]) -> File:
        thread_id = self.azure_devops._thread_id
        if not thread_id:
            raise UserFacingToolError("No chat is associated with this request, so no uploaded file is available.")
        files = await ThreadMessageFileRepository(self.azure_devops.db).find_with_content_by_thread_id(thread_id)
        if file_name:
            match = next((f for f in files if f.name.casefold() == file_name.casefold()), None)
            if match:
                return match
            available = ", ".join(f.name for f in files) or "none"
            raise UserFacingToolError(f"File '{file_name}' was not found in this chat. Available files: {available}")
        if not files:
            raise UserFacingToolError("No file has been uploaded in this chat.")
        return files[0]

    def _is_image_file(self, file: File) -> bool:
        # Azure devops does not support all image formats (e.g. SVG)
        return file.content_type.startswith("image/") and file.name.lower().endswith((".png", ".jpg", ".jpeg"))

    def _image_html(self, attachment_url: str, file_name: str) -> str:
        separator = "&" if "?" in attachment_url else "?"
        src = f"{attachment_url}{separator}fileName={quote(file_name)}"
        return f'<div><img src="{escape(src, quote=True)}" alt="{escape(file_name)}"></div>'
