import logging
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from sqlalchemy import text
from sqlmodel import select
from testcontainers.generic import ServerContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from .common import *

from tero.agents.api import AGENT_TOOL_FILE_PATH
from tero.tools.browser import BrowserTool, BROWSER_TOOL_ID
from tero.tools.codebase import CodebaseTool
from tero.tools.docs import DocsTool, DOCS_TOOL_ID
from tero.tools.docs.tool import DocsExecutionStep
from tero.files.domain import File
from tero.tools.jira import JiraTool
from tero.tools.azure_devops import AzureDevOpsTool
from tero.tools.zephyr import ZephyrTool
from tero.tools.redmine import RedmineTool
from tero.tools.github import GitHubTool
from tero.tools.youtrack import YouTrackTool
from tero.tools.practitest import PractiTestTool
from tero.tools.mcp import McpTool
from tero.tools.notion import NotionTool
from tero.tools.subagent import SubagentTool
from tero.tools.sql import SqlTool
from tero.threads.engine import AgentEngine
from tero.tools.web import WebTool, WEB_TOOL_ID
from tero.usage.domain import Usage, UsageType


logger = logging.getLogger(__name__)


@pytest.fixture
def stub_web_tool_tavily_ainvoke():
    search = '{"results": [{"url": "https://example.com", "content": "Stub search result."}]}'
    extract = [{"url": "https://example.com", "raw_content": "Stub extracted body."}]
    with (
        patch("tero.tools.web.tool.TavilySearch.ainvoke", new=AsyncMock(return_value=search)),
        patch("tero.tools.web.tool.TavilyExtract.ainvoke", new=AsyncMock(return_value=extract)),
    ):
        yield


async def test_find_tools(client: AsyncClient, session: AsyncSession):
    resp = await client.get(f"{BASE_PATH}/tools")
    expected_tools = [DocsTool(), McpTool(), JiraTool(), BrowserTool(), GitHubTool(), YouTrackTool(), ZephyrTool(), PractiTestTool(), RedmineTool(), SqlTool(), AzureDevOpsTool(), CodebaseTool(), NotionTool(), SubagentTool()]
    if env.web_tool_tavily_api_key or (env.web_tool_google_api_key and env.web_tool_google_custom_search_engine_id):
        expected_tools.append(WebTool())
    expected_tools.sort(key=lambda t: t.name.casefold())
    assert_response(resp, expected_tools)


@docs_ingestion_timeout
async def test_docs_tool(client: AsyncClient):
    await _configure_docs_tool_with_file("Emma's routine.pdf", client)
    answer = await _answer_question("What time does Emma wake up according to the document? Output only the time in H:MM format. Don't use clock tool.", client)
    assert "7:35" in answer


_WIDE_MARKDOWN_TABLE = """
| Key | Name | Status | Precondition | Objective | Folder | Priority | Component | Labels | Owner | Estimated Time | Coverage (Issues) | Coverage (Pages) | Country | Material | QA Reviewer | Reviewer | Test Type | Test Script (Step-by-Step) - Step | Test Script (Step-by-Step) - Test Data | Test Script (Step-by-Step) - Expected Result | Test Script (Plain Text) | Test Script (BDD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | ABC/Folder - Create a user with a valid email | Draft | User has access to the application | Verify that an administrator can create a user with a valid email | /ABC/Folder | Normal | CodeCrunchers | CloudInterface | owner-id | | ABC-123 | | USA | Wood | | | Feature Test | Open the user management page | | The user management page is displayed | | |
"""


@docs_ingestion_timeout
async def test_docs_tool_markdown_table_query(client: AsyncClient):
    await _configure_docs_tool_with_file("Emma's routine.pdf", client)
    answer = await _answer_question(
        f"{_WIDE_MARKDOWN_TABLE}\n"
        "Use the docs tool with this entire message as user_query. "
        "What time does Emma wake up according to the document? "
        "Output only the time in H:MM format. Don't use clock tool.",
        client,
    )
    assert "7:35" in answer


async def test_chat_persists_after_tool_sql_failure(client: AsyncClient):
    original_build_runtime = AgentEngine._build_runtime

    def build_runtime_with_failing_query(self, llm, tools, messages, subagents=None):
        async def fail_db_query() -> str:
            """Run a database query. Call this tool whenever the user mentions fail_db_query."""
            await self._tool_db.execute(text("SELECT * FROM __tero_repro_nonexistent__"))
            return "ok"

        failing_tool = StructuredTool.from_function(
            coroutine=fail_db_query,
            name="fail_db_query",
            description="Runs a database query. You must call this when the user mentions fail_db_query.",
        )
        return original_build_runtime(self, llm, [*tools, failing_tool], messages, subagents=subagents)

    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    thread_id = resp.json()["id"]
    with patch.object(AgentEngine, "_build_runtime", build_runtime_with_failing_query):
        async with add_message_to_thread(
            client,
            thread_id,
            "Call the fail_db_query tool, then reply with only the word recovered.",
        ) as stream:
            stream.raise_for_status()
            response = ""
            async for event in stream.aiter_text():
                response += _parse_event(event)

    assert "recovered" in response.lower()
    messages = await client.get(THREAD_MESSAGES_PATH.format(thread_id=thread_id))
    messages.raise_for_status()
    nodes = list(_iter_messages(messages.json()))
    texts = [node["text"] for node in nodes]
    assert any("recovered" in text.lower() for text in texts)
    assert "ERROR_GENERIC" not in texts
    tool_errors = [
        update
        for node in nodes
        for update in (node.get("statusUpdates") or [])
        if update.get("action") == "toolError" and update.get("toolName") == "fail_db_query"
    ]
    assert tool_errors
    assert all(update.get("errorId") and not update.get("result") for update in tool_errors)


def _iter_messages(nodes: list) -> Generator[dict, None, None]:
    for node in nodes:
        yield node
        yield from _iter_messages(node.get("children") or [])


async def _configure_docs_tool_with_file(file_path: str, client: AsyncClient, config: Optional[dict] = None) -> int:
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID, {"advancedFileProcessing": False, **(config or {})}, client)
    content = await find_asset_bytes(file_path)
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client, filename=os.path.basename(file_path),
                                               content=content)
    await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id, client)
    return file_id


async def _answer_question(question: str, client: AsyncClient) -> str:
    answer, _ = await _answer_question_with_thought_process(question, client)
    return answer


def _parse_event(event_str: str) -> str:
    event = None
    data = ""
    event_marker = "event: "
    data_marker = "data: "
    for line in event_str.splitlines():
        if line.startswith(data_marker):
            part = line[len(data_marker):]
            data += (part if part else "\n")
        elif line.startswith(event_marker):
            event = line[len(event_marker):]
    if event == "error":
        raise Exception(data)
    return data


@docs_ingestion_timeout
async def test_docs_tool_with_removed_file(client: AsyncClient):
    file_id = await _configure_docs_tool_with_file("Emma's routine.pdf", client)
    resp = await client.delete(AGENT_TOOL_FILE_PATH.format(agent_id=1, tool_id=DOCS_TOOL_ID, file_id=file_id))
    resp.raise_for_status()
    answer = await _answer_question("What time does Emma wake up according to the document? Output only the time in H:MM format. Don't use clock tool.", client)
    assert "7:35" not in answer


async def test_resolve_doc_files_case_insensitive():
    tool = DocsTool()
    available_files = [
        File(id=1, name="manual.pdf", content_type="application/pdf", user_id=1, content=b""),
        File(id=2, name="Guide.TXT", content_type="text/plain", user_id=1, content=b""),
    ]
    assert tool._resolve_file_ids(["Manual.PDF", "guide.txt"], available_files) == ["1", "2"]


async def test_run_returns_message_when_files_not_found():
    tool = DocsTool()
    agent = MagicMock()
    agent.id = 1
    tool.configure(agent=agent, user_id=1, config={}, db=MagicMock())
    tool._find_available_files = AsyncMock(return_value=[
        File(id=1, name="manual.pdf", content_type="application/pdf", user_id=1, content=b""),
    ])
    result = await tool._run("what does it say?", files=["nonexistent.pdf"])
    assert "nonexistent.pdf" in result
    assert "manual.pdf" in result


async def test_run_keeps_braces_in_system_prompt():
    tool = DocsTool()
    agent = MagicMock()
    agent.id = 1
    agent.system_prompt = "Refer to the record {id} in the field {texto}."
    tool.configure(agent=agent, user_id=1, config={"responseGrounding": False}, db=MagicMock())
    tool._find_available_files = AsyncMock(return_value=[
        File(id=1, name="manual.pdf", content_type="application/pdf", user_id=1, content=b""),
    ])
    tool._build_retriever = MagicMock(return_value=RunnableLambda(lambda _: "retrieved context"))
    seen_prompt = {}

    async def fake_llm(prompt_value):
        seen_prompt["text"] = prompt_value.to_string()
        return AIMessage(content="the answer")

    usage_repo = MagicMock()
    usage_repo.add = AsyncMock()
    with (
        patch("tero.tools.docs.tool.ensure_config", return_value={}),
        patch("tero.tools.docs.tool.ai_factory.build_chat_model", return_value=RunnableLambda(fake_llm)),
        patch("tero.tools.docs.tool.get_stream_writer", return_value=MagicMock()),
        patch("tero.tools.docs.tool.UsageRepository", return_value=usage_repo),
    ):
        result = await tool._run("what does the manual say?")

    assert result == "the answer"
    assert "{id}" in seen_prompt["text"]
    assert "{texto}" in seen_prompt["text"]
    assert "what does the manual say?" in seen_prompt["text"]


_DOCS_WAKE_UP_QUESTION = (
    "What time does Emma wake up according to the document? "
    "Output only the time in H:MM format. Don't use clock tool."
)
_GROUNDING_STEPS = {DocsExecutionStep.GROUNDING_RESPONSE.value, DocsExecutionStep.GROUNDED_RESPONSE.value}


@docs_ingestion_timeout
async def test_docs_tool_response_grounding(client: AsyncClient):
    await _configure_docs_tool_with_file("Emma's routine.pdf", client)
    answer, thought_process = await _answer_question_with_thought_process(_DOCS_WAKE_UP_QUESTION, client)
    assert "7:35" in answer
    assert _GROUNDING_STEPS <= _thought_process_steps(thought_process)

    await configure_agent_tool(
        AGENT_ID, DOCS_TOOL_ID, {"advancedFileProcessing": False, "responseGrounding": False}, client
    )
    answer, thought_process = await _answer_question_with_thought_process(_DOCS_WAKE_UP_QUESTION, client)
    assert "7:35" in answer
    assert _thought_process_steps(thought_process).isdisjoint(_GROUNDING_STEPS)


async def _answer_question_with_thought_process(question: str, client: AsyncClient) -> tuple[str, list]:
    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    thread_id = resp.json()["id"]
    async with add_message_to_thread(client, thread_id, question) as stream:
        stream.raise_for_status()
        answer = ""
        async for event in stream.aiter_text():
            answer += _parse_event(event)
    resp = await client.get(THREAD_MESSAGES_PATH.format(thread_id=thread_id))
    resp.raise_for_status()
    return answer, _first_agent_message(resp.json()).get("statusUpdates") or []


def _first_agent_message(messages: list) -> dict:
    for message in messages:
        if message["origin"] == "AGENT":
            return message
        children = message.get("children") or []
        if children:
            return _first_agent_message(children)
    raise AssertionError("No agent message found")


def _thought_process_steps(thought_process: list) -> set:
    return {event.get("step") for event in thought_process if event.get("step")}


@docs_ingestion_timeout
async def test_docs_tool_filters_by_file(client: AsyncClient):
    await _configure_docs_tool_with_file("Emma's routine.pdf", client)
    await _configure_docs_tool_with_file("sample.txt", client)
    answer = await _answer_question(
        'Use the docs tool with files=["Emma\'s routine.pdf"]. What time does Emma wake up? Output only the time in H:MM format.',
        client,
    )
    assert "7:35" in answer
    assert "Sample test" not in answer


@pytest.mark.usefixtures("stub_web_tool_tavily_ainvoke")
async def test_web_tool_search_usage(client: AsyncClient, session: AsyncSession):
    await configure_agent_tool(AGENT_ID, WEB_TOOL_ID, {}, client)

    initial_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_SEARCH))
    initial_count = len(initial_usage.all())

    await _answer_question("Search for the latest news about AI", client)

    final_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_SEARCH))
    final_count = len(final_usage.all())

    assert final_count > initial_count


@pytest.mark.usefixtures("stub_web_tool_tavily_ainvoke")
async def test_web_tool_extract_usage(client: AsyncClient, session: AsyncSession):
    await configure_agent_tool(AGENT_ID, WEB_TOOL_ID, {}, client)

    initial_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_EXTRACT))
    initial_count = len(initial_usage.all())

    await _answer_question("What is the first paragraph of this url: https://modelcontextprotocol.io/", client)

    final_usage = await session.exec(select(Usage).where(Usage.type == UsageType.WEB_EXTRACT))
    final_count = len(final_usage.all())

    assert final_count > initial_count


@pytest.fixture(scope="function")
def containers_network() -> Generator[Network, None, None]:
    with Network() as network:
        yield network


@pytest.fixture(scope="function")
def playwright_container_url(containers_network: Network) -> Generator[str, None, None]:
    port = 8931
    output_dir = "/tmp/share/playwright-output"
    env.browser_tool_playwright_output_dir = output_dir
    with DockerContainer("mcp/playwright")\
            .with_exposed_ports(port)\
            .with_network(containers_network)\
            .with_kwargs(user="0:0")\
            .with_command([f"--port={port}", "--allowed-hosts=*", "--host=0.0.0.0", "--output-dir=/tmp/playwright-output"]) \
            .with_volume_mapping(output_dir, "/tmp/playwright-output", "rw") \
            .waiting_for(LogMessageWaitStrategy(f"Listening on http://localhost:{port}")) \
            as container:
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(port)}"


@pytest.fixture(scope="function")
def nginx_container_url(containers_network: Network) -> Generator[str, None, None]:
    nginx_port = 80
    network_alias = "nginx"
    with ServerContainer(nginx_port, "nginx:1.29") \
            .with_network(containers_network) \
            .with_network_aliases(network_alias):
        yield f"http://{network_alias}:{nginx_port}"


async def test_browser_tool(client: AsyncClient, playwright_container_url: str, nginx_container_url: str):
    await _configure_browser_tool(playwright_container_url, client)
    answer = await _answer_question(f"Navigate to {nginx_container_url} and get body of the page", client)
    assert "Welcome to nginx!" in answer


async def _configure_browser_tool(playwright_container_url: str, client: AsyncClient):
    env.browser_tool_playwright_mcp_url = playwright_container_url + "/mcp"
    await configure_agent_tool(AGENT_ID, BROWSER_TOOL_ID, {}, client)


async def test_browser_tool_screenshot(client: AsyncClient, playwright_container_url: str, nginx_container_url: str):
    await _configure_browser_tool(playwright_container_url, client)
    answer = await _answer_question(f"Navigate to {nginx_container_url} and take a screenshot of the page", client)
    assert '"files": [{' in answer
