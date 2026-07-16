from enum import Enum
import json
from typing import Optional, cast, Any, Callable, Awaitable

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool, StructuredTool
from mcp.client.session import ClientSession
from pydantic import BaseModel, Field

from ...ai_models import ai_factory
from ..core import load_schema
from ..mcp_tool import AuthType, BaseMcpTool


GITHUB_TOOL_ID = "github"
GITHUB_MCP_SERVER_URL = "https://api.githubcopilot.com/mcp/"
ANALYZE_LOG_PROMPT = """
You are a helpful assistant that analyzes GitHub Actions workflow run logs.
You are given a GitHub Actions workflow run log and you need to analyze it and provide a summary of the log.
The log is in raw format and depends on the job that failed.
- You need to provide a summary of the log in a way that is easy to understand for a human.
- The summary should provide a possible cause and solution for the failure.
- If the log is in any test framework format you should provide a summary of the test results including the number of tests passed and failed,
  the list of tests passed and failed, and when possible the categorization of the tests.
"""


class CreateReportArgs(BaseModel):
    id: str = Field(description="GitHub Actions workflow run id")
    owner: str = Field(description="Repository owner (user or organization)")
    repo: str = Field(description="Repository name")


class WorkflowRunsEventType(str, Enum):
    branch_protection_rule = "branch_protection_rule"
    check_run = "check_run"
    check_suite = "check_suite"
    create = "create"
    delete = "delete"
    deployment = "deployment"
    deployment_status = "deployment_status"
    discussion = "discussion"
    discussion_comment = "discussion_comment"
    fork = "fork"
    gollum = "gollum"
    issue_comment = "issue_comment"
    issues = "issues"
    label = "label"
    merge_group = "merge_group"
    milestone = "milestone"
    page_build = "page_build"
    public = "public"
    pull_request = "pull_request"
    pull_request_review = "pull_request_review"
    pull_request_review_comment = "pull_request_review_comment"
    pull_request_target = "pull_request_target"
    push = "push"
    registry_package = "registry_package"
    release = "release"
    repository_dispatch = "repository_dispatch"
    schedule = "schedule"
    status = "status"
    watch = "watch"
    workflow_call = "workflow_call"
    workflow_dispatch = "workflow_dispatch"
    workflow_run = "workflow_run"


class WorkflowRunsStatusFilter(str, Enum):
    queued = "queued"
    in_progress = "in_progress"
    completed = "completed"
    requested = "requested"
    waiting = "waiting"


class WorkflowRunsFilter(BaseModel):
    actor: Optional[str] = Field(None, description="Filter to a specific GitHub user's workflow runs")
    branch: Optional[str] = Field(None, description="Filter workflow runs to a specific Git branch")
    event: Optional[WorkflowRunsEventType] = Field(None, description="Filter workflow runs to a specific event type")
    status: Optional[WorkflowRunsStatusFilter] = Field(None, description="Filter workflow runs to only runs with a specific status")


class WorkflowRunsReducedArgs(BaseModel):
    owner: str = Field(description="Repository owner")
    repo: str = Field(description="Repository name")
    resource_id: Optional[str] = Field(None, description="Workflow ID or workflow file name (e.g. ci.yml) to filter runs for a specific workflow. Omit to list all workflow runs in the repository.")
    per_page: Optional[int] = Field(30, ge=30, le=100, description="Results per page (default: 30, min: 30, max: 100)")
    page: Optional[int] = Field(1, ge=1, description="Page number for pagination (default: 1)")
    workflow_runs_filter: Optional[WorkflowRunsFilter] = Field(None,description="Optional filters for workflow runs")


class GitHubTool(BaseMcpTool):
    id: str = GITHUB_TOOL_ID
    name: str = "GitHub"
    description: str = "Search code, manage repos, review PRs and track issues"
    config_schema: dict = load_schema(__file__)

    def _get_auth_type(self) -> AuthType:
        return AuthType.BEARER_TOKEN

    def _get_mcp_server_url(self) -> str:
        return GITHUB_MCP_SERVER_URL

    def _build_headers(self, auth_headers: dict[str, str]) -> dict[str, str]:
        headers = dict(auth_headers)
        headers["X-MCP-Toolsets"] = "default,actions"
        return headers

    async def _build_tools(self, mcp_session: ClientSession) -> list[BaseTool]:
        ret = await super()._build_tools(mcp_session)
        tools_by_name = {t.name: t for t in ret}
        ret.append(self._create_report_tool(tools_by_name))
        ret.append(self._create_workflow_runs_reduced(tools_by_name))
        return ret

    def _create_report_tool(self, tools_by_name: dict[str, BaseTool]) -> BaseTool:
        actions_list_fn = self._get_structured_tool("actions_list", tools_by_name)
        actions_get_fn = self._get_structured_tool("actions_get", tools_by_name)
        get_job_logs_fn = self._get_structured_tool("get_job_logs", tools_by_name)

        async def create_report(owner: str, repo: str, id: str) -> str:
            if not actions_get_fn or not get_job_logs_fn or not actions_list_fn:
                raise ValueError("Missing tools")

            workflow_run_result = await actions_get_fn(
                method="get_workflow_run", owner=owner, repo=repo, resource_id=id
            )
            workflow_run = json.loads(workflow_run_result[0])
            if not workflow_run:
                raise ValueError("No workflow run found")

            workflow_run_id = str(workflow_run.get("id"))
            jobs_result = await actions_list_fn(
                method="list_workflow_jobs",
                owner=owner,
                repo=repo,
                resource_id=workflow_run_id,
            )
            jobs = json.loads(jobs_result[0]).get("jobs", {}).get("jobs", [])

            jobs_logs_result = await get_job_logs_fn(
                failed_only=True,
                owner=owner,
                repo=repo,
                return_content=True,
                run_id=workflow_run_id,
                tail_lines=100,
            )
            failure_jobs_logs = json.loads(jobs_logs_result[0]).get("logs", [])
            jobs_logs_by_id = {}
            for failure_job_log in failure_jobs_logs:
                job_id = failure_job_log.get("job_id")
                job_log_content = failure_job_log.get("logs_content")
                jobs_logs_by_id[job_id] = await self._analyze_job_log(job_id, job_log_content)

            jobs_summary = [{
                "id": job.get("id"),
                "name": job.get("name"),
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "steps": job.get("steps", []),
                "logs": jobs_logs_by_id.get(job.get("id"), ""),
            } for job in jobs ]

            return json.dumps({
                "id": workflow_run_id,
                "run_number": workflow_run.get("run_number"),
                "name": workflow_run.get("name"),
                "conclusion": workflow_run.get("conclusion"),
                "status": workflow_run.get("status"),
                "head_branch": workflow_run.get("head_branch"),
                "event": workflow_run.get("event"),
                "created_at": workflow_run.get("created_at"),
                "updated_at": workflow_run.get("updated_at"),
                "run_started_at": workflow_run.get("run_started_at"),
                "html_url": workflow_run.get("html_url"),
                "commit_message": (workflow_run.get("head_commit") or {}).get("message"),
                "actor": (workflow_run.get("actor") or {}).get("login"),
                "jobs": jobs_summary,
            })

        return StructuredTool.from_function(
            coroutine=create_report,
            name="create_report",
            description="Create a GitHub Actions workflow run report for a repository",
            args_schema=CreateReportArgs,
        )

    def _get_structured_tool(self, tool_name: str, tools_by_name: dict[str, BaseTool]) -> Optional[Callable[..., Awaitable[Any]]]:
        ret = tools_by_name.get(tool_name)
        # coroutine is used to avoid callbacks triggering in internal calls to the MCP server
        return cast(StructuredTool, ret).coroutine if ret else None

    async def _analyze_job_log(self, job_id: str, job_log_content: str) -> str:
        llm = ai_factory.build_chat_model(
            self.agent.model.id,
            self.agent.model_temperature,
            self.agent.model_reasoning_effort,
        )
        ret = await llm.ainvoke(
            [
                HumanMessage(
                    ANALYZE_LOG_PROMPT
                    + "\n\n This is the log:\n\n"
                    + job_log_content
                )
            ]
        )
        return cast(str, cast(AIMessage, ret).content)

    def _create_workflow_runs_reduced(self, tools_by_name: dict[str, BaseTool]) -> BaseTool:
        actions_list_fn = self._get_structured_tool("actions_list", tools_by_name)

        async def workflow_runs_reduced(
            owner: str,
            repo: str,
            resource_id: Optional[str] = None,
            per_page: Optional[int] = 30,
            page: Optional[int] = 1,
            workflow_runs_filter: Optional[WorkflowRunsFilter] = None,
        ) -> str:
            if not actions_list_fn:
                raise ValueError("Missing tools")

            kwargs: dict[str, Any] = {
                "method": "list_workflow_runs",
                "owner": owner,
                "repo": repo,
                "per_page": per_page,
                "page": page,
            }
            if resource_id:
                kwargs["resource_id"] = resource_id
            if workflow_runs_filter:
                kwargs["workflow_runs_filter"] = workflow_runs_filter.model_dump(exclude_none=True)
            workflows_runs_result = await actions_list_fn(**kwargs)
            workflows_runs = json.loads(workflows_runs_result[0])
            if not workflows_runs:
                return json.dumps({"total_count": 0, "workflow_runs": []})
            else:
                return json.dumps({
                    "total_count": workflows_runs.get("total_count"),
                    "workflow_runs": [{
                        "id": workflow_run.get("id"),
                        "name": workflow_run.get("name"),
                        "status": workflow_run.get("status"),
                        "conclusion": workflow_run.get("conclusion"),
                        "html_url": workflow_run.get("html_url"),
                        "repository_name": workflow_run.get("repository").get("name"),
                        "actor_login": workflow_run.get("actor").get("login"),
                        "run_started_at": workflow_run.get("run_started_at"),
                    } for workflow_run in workflows_runs.get("workflow_runs", [])]
                })

        return StructuredTool.from_function(
            coroutine=workflow_runs_reduced,
            name="workflow_runs_reduced",
            description="List GitHub Actions workflow runs for a repository",
            args_schema=WorkflowRunsReducedArgs,
        )
