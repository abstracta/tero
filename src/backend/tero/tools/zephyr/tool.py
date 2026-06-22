from http import HTTPMethod
from typing import Any, Optional

from langchain_core.callbacks import Callbacks
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core import StatusUpdateCallbackHandler, load_schema
from ..openapi_tool import TokenOpenApiTool

ZEPHYR_TOOL_ID = "zephyr"

PRODUCT_URLS = {
    "cloud": "https://api.zephyrscale.smartbear.com/v2",
    "essential": "https://prod-api.zephyr4jiracloud.com/v2",
}


class ZephyrTool(TokenOpenApiTool):
    id: str = ZEPHYR_TOOL_ID
    name: str = "Zephyr"
    description: str = "Manage test cases, plans, cycles, and executions"
    config_schema: dict = load_schema(__file__)
    _api_url: str = ""
    _token_secret_key = "accessToken"
    _auth_check_path = "/healthcheck"

    def configure(self, agent, user_id: int, config: dict, db: AsyncSession, thread_id: Optional[int] = None):
        super().configure(agent, user_id, config, db, thread_id)
        self._api_url = PRODUCT_URLS[self.config["product"]].rstrip("/")

    async def _load_api_spec(self) -> dict:
        return await self._load_json(f"zephyr-{self.config['product']}-api-spec.json")

    async def build_langchain_tools(self) -> list[BaseTool]:
        tools = await super().build_langchain_tools()
        tools.append(ZephyrSearchTestCasesTool(zephyr=self))
        tools.append(ZephyrCloneTestCaseTool(zephyr=self))
        return tools

    def _invalid_token_message(self) -> str:
        return "Invalid Zephyr API access token"

    def _apply_token_to_headers(self, headers: dict, token: str) -> dict:
        headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _fetch_paged_values(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict]:
        values: list[dict] = []
        start_at = 0
        while True:
            query: dict[str, Any] = {"maxResults": 1000, "startAt": start_at}
            if params is not None:
                query.update(params)
            page = await self._invoke_rest_api(HTTPMethod.GET, url, params=query)
            batch = page.get("values") or []
            values.extend(batch)
            if not batch or page["isLast"]:
                break
            start_at += len(batch)
        return values


class SearchTestCasesArgs(BaseModel):
    projectKey: str = Field(description="Jira project key filter")
    folderId: Optional[int] = Field(default=None, description="Folder ID filter")
    statusNames: Optional[list[str]] = Field(default=None, description="Status names filter")
    labelNames: Optional[list[str]] = Field(
        default=None,
        description="Label names filter. Test case must include every listed label.",
    )
    nameContains: Optional[str] = Field(default=None, description="Test case name substring filter")
    # Copied from other maxResults fields in the API spec
    maxResults: int = Field(
        default=10,
        ge=1,
        le=1000,
        description=(
            "Specifies the maximum number of results to return in a single call. "
            "The default value is 10, and the maximum value that can be requested is 1000."
        ),
    )
    # Copied from other startAt fields in the API spec
    startAt: int = Field(
        default=0,
        ge=0,
        le=1_000_000,
        description="Zero-indexed starting position. Should be a multiple of maxResults.",
    )


class ZephyrSearchTestCasesTool(BaseTool):
    name: str = "Zephyr-searchTestCases"
    description: str = "Zephyr tool that searches test cases. Query parameters can be used to filter the results."
    args_schema: Optional[ArgsSchema] = SearchTestCasesArgs
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]
    zephyr: ZephyrTool

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def _arun(self, **kwargs: Any) -> dict[str, Any]:
        args = SearchTestCasesArgs.model_validate(kwargs)
        params: dict[str, Any] = {"projectKey": args.projectKey}
        if args.folderId is not None:
            params["folderId"] = args.folderId
        cases = await self.zephyr._fetch_paged_values(f"{self.zephyr._api_url}/testcases", params)
        allowed_status_ids: Optional[set[int]] = None
        if args.statusNames:
            allowed_status_ids = await self._find_status_ids_with_names(args.projectKey, args.statusNames)
        label_filter = {label.casefold() for label in args.labelNames} if args.labelNames else None
        name_filter = args.nameContains.casefold() if args.nameContains else None
        matched = []
        for case in cases:
            if allowed_status_ids is not None and (case.get("status") or {}).get("id") not in allowed_status_ids:
                continue
            if label_filter:
                case_labels = {label.casefold() for label in case.get("labels") or []}
                if not label_filter.issubset(case_labels):
                    continue
            if name_filter and name_filter not in (case.get("name") or "").casefold():
                continue
            matched.append(case)
        page = matched[args.startAt : args.startAt + args.maxResults]
        return {
            "startAt": args.startAt,
            "maxResults": len(page),
            "total": len(matched),
            "isLast": args.startAt + len(page) >= len(matched),
            "values": page,
        }
    
    async def _find_status_ids_with_names(self, project_key: str, status_names: list[str]) -> set[int]:
        statuses = await self.zephyr._fetch_paged_values(
            f"{self.zephyr._api_url}/statuses",
            {"projectKey": project_key, "statusType": "TEST_CASE"},
        )
        return {
            status["id"]
            for status in statuses
            if (status.get("name") or "").casefold() in {name.casefold() for name in status_names}
        }


class CloneTestCaseArgs(BaseModel):
    sourceTestCaseKey: str = Field(description="The key of the test case to clone.")


class ZephyrCloneTestCaseTool(BaseTool):
    name: str = "Zephyr-cloneTestCase"
    description: str = (
        "Zephyr tool that clones a test case: copies metadata, custom fields, test steps or test script, "
        "and Jira/weblink traceability links. Attachments and comments are not copied. "
        "The new case name is the source name plus '(cloned)'."
    )
    args_schema: Optional[ArgsSchema] = CloneTestCaseArgs
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description)]
    zephyr: ZephyrTool

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def _arun(self, **kwargs: Any) -> dict[str, Any]:
        args = CloneTestCaseArgs.model_validate(kwargs)
        base = f"{self.zephyr._api_url}/testcases"
        source = await self.zephyr._invoke_rest_api(HTTPMethod.GET, f"{base}/{args.sourceTestCaseKey}")
        project_key, _, _ = source["key"].rpartition("-T")
        body: dict[str, Any] = {
            "projectKey": project_key,
            "name": f"{source.get('name') or ''} (cloned)",
            "objective": source.get("objective"),
            "precondition": source.get("precondition"),
            "estimatedTime": source.get("estimatedTime"),
            "labels": source.get("labels"),
            "customFields": source.get("customFields"),
        }
        if component := source.get("component"):
            body["componentId"] = component.get("id")
        if folder := source.get("folder"):
            body["folderId"] = folder.get("id")
        if owner := source.get("owner"):
            body["ownerId"] = owner.get("accountId")
        if priority_name := await self._linked_resource_name("priorities", source.get("priority")):
            body["priorityName"] = priority_name
        if status_name := await self._linked_resource_name("statuses", source.get("status")):
            body["statusName"] = status_name
        create_body = {k: v for k, v in body.items() if v is not None}
        created = await self.zephyr._invoke_rest_api(HTTPMethod.POST, base, body=create_body)
        new_key = created["key"]
        if (source.get("testScript") or {}).get("self", "").rstrip("/").casefold().endswith("/testscript"):
            script = await self.zephyr._invoke_rest_api(
                HTTPMethod.GET, f"{base}/{args.sourceTestCaseKey}/testscript"
            )
            await self.zephyr._invoke_rest_api(
                HTTPMethod.POST,
                f"{base}/{new_key}/testscript",
                body={"type": script["type"], "text": script["text"]},
            )
        else:
            items = []
            for raw in await self.zephyr._fetch_paged_values(f"{base}/{args.sourceTestCaseKey}/teststeps"):
                if inline := raw.get("inline"):
                    if self._inline_has_content(inline):
                        payload = {k: v for k, v in inline.items() if v is not None}
                        items.append({"inline": payload})
                elif tc := raw.get("testCase"):
                    step: dict[str, Any] = {"testCaseKey": tc["testCaseKey"]}
                    if tc.get("parameters"):
                        step["parameters"] = tc["parameters"]
                    items.append({"testCase": step})
            for i in range(0, len(items), 100):
                await self.zephyr._invoke_rest_api(
                    HTTPMethod.POST,
                    f"{base}/{new_key}/teststeps",
                    body={"mode": "OVERWRITE" if i == 0 else "APPEND", "items": items[i : i + 100]},
                )
        links = source.get("links") or {}
        for issue in links.get("issues") or []:
            await self.zephyr._invoke_rest_api(
                HTTPMethod.POST,
                f"{base}/{new_key}/links/issues",
                body={"issueId": issue["issueId"]},
            )
        for web in links.get("webLinks") or []:
            await self.zephyr._invoke_rest_api(
                HTTPMethod.POST,
                f"{base}/{new_key}/links/weblinks",
                body={k: web[k] for k in ("url", "description") if web.get(k)},
            )
        return created
    
    @staticmethod
    def _inline_has_content(inline: dict) -> bool:
        return any((inline.get(k) or "").strip() for k in ("description", "testData", "expectedResult"))


    async def _linked_resource_name(self, resource: str, link: Optional[dict]) -> str:
        if link is None or (resource_id := link.get("id")) is None:
            return ""
        entity = await self.zephyr._invoke_rest_api(
            HTTPMethod.GET,
            f"{self.zephyr._api_url}/{resource}/{resource_id}",
        )
        return entity.get("name") or ""
