from io import BytesIO
from typing import cast
from zipfile import ZipFile

import aiofiles
from sqlmodel import col

from .common import *

from tero.agents.api import AGENTS_PATH, AGENT_PATH, AGENT_TOOLS_PATH, AGENT_TOOL_FILE_PATH, DEFAULT_SYSTEM_PROMPT
from tero.agents.domain import PublicAgent, AgentToolConfig, AgentType, LlmTemperature, ReasoningEffort, AgentUpdate
from tero.agents.prompts.api import AGENT_PROMPTS_PATH
from tero.agents.prompts.domain import AgentPromptCreate, AgentPromptPublic, AgentPrompt
from tero.agents.test_cases.api import TEST_CASES_PATH, TEST_CASE_MESSAGES_PATH, TEST_CASE_PATH
from tero.agents.test_cases.domain import NewTestCaseMessage, PublicTestCase
from tero.files.domain import FileMetadata, FileStatus, File, FileProcessor
from tero.threads.domain import ThreadMessageOrigin, Thread, ThreadMessagePublic
from tero.tools.browser import BROWSER_TOOL_ID
from tero.tools.docs import DOCS_TOOL_ID
from tero.tools.web import WEB_TOOL_ID
from tero.users.domain import UserListItem


@freeze_time(CURRENT_TIME)
async def test_import_exported_minimal_agent(users: List[UserListItem], client: AsyncClient):
    source_agent_id = await _create_agent(client)
    zip_file_content = await _export_agent(source_agent_id, client)
    target_agent_id = await _create_agent(client)
    result = await _import_agent(target_agent_id, zip_file_content, client)
    assert result == {"unavailableTools": [], "toolsRequiringAuthentication": [], "unavailableModel": None, "defaultModel": None}
    resp = await _find_agent(target_agent_id, client)
    assert_response(resp, PublicAgent(
        id=target_agent_id, name=f"Agent #{source_agent_id}", description="", last_update=CURRENT_TIME, user_id=USER_ID,
        model_id=cast(str, env.agent_default_model), system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=LlmTemperature.NEUTRAL, reasoning_effort=ReasoningEffort.LOW,
        recursion_limit=20, icon=None, can_edit=True, user=users[0]))


@freeze_time(CURRENT_TIME)
async def test_import_exported_agent_preserves_advanced_settings(users: List[UserListItem], client: AsyncClient):
    source_agent_id = await _create_agent(client)
    await _update_agent(source_agent_id, AgentUpdate(
        recursion_limit=50, agent_type=AgentType.DEEP_AGENT, is_protected=True), client)
    zip_file_content = await _export_agent(source_agent_id, client)
    target_agent_id = await _create_agent(client)
    await _import_agent(target_agent_id, zip_file_content, client)
    resp = await _find_agent(target_agent_id, client)
    assert_response(resp, PublicAgent(
        id=target_agent_id, name=f"Agent #{source_agent_id}", description="", last_update=CURRENT_TIME, user_id=USER_ID,
        model_id=cast(str, env.agent_default_model), system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=LlmTemperature.NEUTRAL,
        reasoning_effort=ReasoningEffort.LOW, recursion_limit=50, icon=None, can_edit=True, user=users[0],
        is_protected=True, agent_type=AgentType.DEEP_AGENT))


@freeze_time(CURRENT_TIME)
async def test_import_old_export_applies_default_advanced_settings(users: List[UserListItem], client: AsyncClient):
    agent_id = await _create_agent(client)
    await _update_agent(agent_id, AgentUpdate(recursion_limit=40, agent_type=AgentType.DEEP_AGENT), client)
    await _import_agent(agent_id, await _zip_markdown("minimal_agent"), client)
    resp = await _find_agent(agent_id, client)
    assert_response(resp, PublicAgent(
        id=agent_id, name="Test Agent", description="A Test Agent", last_update=CURRENT_TIME, user_id=USER_ID,
        model_id="gpt-5-mini", system_prompt="Test instructions", temperature=LlmTemperature.NEUTRAL,
        reasoning_effort=ReasoningEffort.MEDIUM, recursion_limit=20, icon=None, can_edit=True, user=users[0],
        agent_type=AgentType.REACT_AGENT))


async def _create_agent(client: AsyncClient) -> int:
    resp = await client.post(AGENTS_PATH)
    resp.raise_for_status()
    return resp.json()["id"]


async def _export_agent(agent_id: int, client: AsyncClient) -> bytes:
    resp = await _try_export_agent(agent_id, client)
    resp.raise_for_status()
    return resp.content


async def _try_export_agent(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(f"{AGENT_PATH.format(agent_id=agent_id)}/dist")


async def _import_agent(agent_id: int, file: bytes, client: AsyncClient) -> dict:
    resp = await _try_import_agent(agent_id, file, client)
    resp.raise_for_status()
    return resp.json()


async def _try_import_agent(agent_id: int, file: bytes, client: AsyncClient) -> Response:
    return await client.put(f"{AGENT_PATH.format(agent_id=agent_id)}/dist", files={"file": ("test.zip", file)})


async def _find_agent(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(AGENT_PATH.format(agent_id=agent_id))


@pytest.fixture(name="last_prompt_id")
async def last_prompt_id_fixture(session: AsyncSession) -> int:
    return await find_last_id(col(AgentPrompt.id), session)


@pytest.fixture(name="last_file_id")
async def fixture_last_file_id(session: AsyncSession) -> int:
    return await find_last_id(col(File.id), session)


@freeze_time(CURRENT_TIME)
@pytest.mark.usefixtures("stub_docs_tool_generate_description")
async def test_import_exported_agent_with_all_tools_and_configs(
    users: List[UserListItem], last_prompt_id: int, last_file_id: int, last_thread_id: int, last_message_id: int, client: AsyncClient):
    agent_update = AgentUpdate(name="Test Agent 1", description="Test description", system_prompt="Test system prompt",
            model_id="gpt-5-mini", icon=TEST_ICON, reasoning_effort= ReasoningEffort.MEDIUM)
    prompts = [
        AgentPromptCreate(name="Starter", content="Starter Text", shared=True, starter=True),
        AgentPromptCreate(name="Private", content="Private Text", shared=False),
        AgentPromptCreate(name="Public", content="Public Text", shared=True)
    ]
    advanced_file_processing = False
    file_name = "sample.txt"
    file_content = await find_asset_bytes(file_name)
    test_messages = [
        NewTestCaseMessage(text="Test user message", origin=ThreadMessageOrigin.USER),
        NewTestCaseMessage(text="Test agent message", origin=ThreadMessageOrigin.AGENT),
    ]
    source_agent_id, test_case_name = await _create_agent_with_all_tools_and_configs(
        agent_update, prompts, advanced_file_processing, file_name, file_content, test_messages, client)
    zip_file_content = await _export_agent(source_agent_id, client)
    target_agent_id = await _create_agent(client)
    result = await _import_agent(target_agent_id, zip_file_content, client)
    assert result == {"unavailableTools": [], "toolsRequiringAuthentication": [], "unavailableModel": None, "defaultModel": None}
    await _assert_imported_agent(target_agent_id, agent_update, prompts, advanced_file_processing, file_name, file_content, test_messages,
        last_prompt_id + len(prompts), last_file_id + 1, last_thread_id + 1, last_message_id + len(test_messages), test_case_name, users, client)


async def _create_agent_with_all_tools_and_configs(
        agent_update: AgentUpdate, prompts: list[AgentPromptCreate], advanced_file_processing: bool, file_name: str, file_content: bytes,
        test_messages: list[NewTestCaseMessage], client: AsyncClient) -> tuple[int, str]:
    agent_id = await _create_agent(client)
    await _update_agent(agent_id, agent_update, client)
    for prompt in prompts:
        await _add_agent_prompt(agent_id, prompt, client)
    await configure_agent_tool(agent_id, DOCS_TOOL_ID, {"advancedFileProcessing": advanced_file_processing}, client)
    file_id = await upload_agent_tool_config_file(agent_id, DOCS_TOOL_ID, client, filename=file_name, content=file_content)
    await await_files_processed(agent_id, DOCS_TOOL_ID, file_id, client)
    await configure_agent_tool(agent_id, WEB_TOOL_ID, {}, client)
    await configure_agent_tool(agent_id, BROWSER_TOOL_ID, {}, client)
    # this test does not add mcp and jira tools since they require a mock and tests for those tools are not implemented yet
    # add test case
    test_id = await _add_test(agent_id, client)
    for message in test_messages:
        await _add_test_message(agent_id, test_id, message, client)
    # refresh test case to get updated name after adding messages
    test_case = await _find_test_case(agent_id, test_id, client)
    return agent_id, test_case["thread"]["name"]


async def _update_agent(agent_id: int, update: AgentUpdate, client: AsyncClient):
    resp = await client.put(AGENT_PATH.format(agent_id=agent_id), json=update.model_dump(mode="json", by_alias=True))
    resp.raise_for_status()


async def _add_agent_prompt(agent_id: int, prompt: AgentPromptCreate, client: AsyncClient):
    resp = await client.post(AGENT_PROMPTS_PATH.format(agent_id=agent_id), json=prompt.model_dump(mode="json", by_alias=True))
    resp.raise_for_status()


async def _add_test(agent_id: int, client: AsyncClient) -> int:
    resp = await client.post(TEST_CASES_PATH.format(agent_id=agent_id))
    resp.raise_for_status()
    return resp.json()["thread"]["id"]


async def _add_test_message(agent_id: int, test_case_id: int, message: NewTestCaseMessage, client: AsyncClient):
    resp = await client.post(TEST_CASE_MESSAGES_PATH.format(agent_id=agent_id, test_case_id=test_case_id),
            json=message.model_dump(mode="json", by_alias=True))
    resp.raise_for_status()


async def _assert_imported_agent(agent_id: int, agent_update: AgentUpdate, prompts: list[AgentPromptCreate],
        advanced_file_processing: bool, file_name: str, file_content: bytes, test_messages: list[NewTestCaseMessage],
        last_prompt_id: int, last_file_id: int, last_thread_id: int, last_message_id: int, test_case_name: str, users: List[UserListItem], client: AsyncClient):
    resp = await _find_agent(agent_id, client)
    resp.raise_for_status()
    assert_response(resp, PublicAgent(
        id=agent_id, name=agent_update.name, description=agent_update.description, last_update=CURRENT_TIME,
        user_id=USER_ID, model_id=cast(str, agent_update.model_id), system_prompt=cast(str, agent_update.system_prompt),
        temperature=LlmTemperature.NEUTRAL,  reasoning_effort=cast(ReasoningEffort, agent_update.reasoning_effort),
        recursion_limit=20, icon=agent_update.icon, can_edit=True, user=users[0]))

    agent_prompts = await _find_agent_prompts(agent_id, client)
    expected_prompts = [
        AgentPromptPublic(
            id=last_prompt_id + 1 + i, name=p.name, content=p.content, shared=p.shared,
            last_update=CURRENT_TIME, user_id=USER_ID, can_edit=True, starter=p.starter)
        for i, p in enumerate(prompts)]
    expected_prompts.sort(key=lambda x: x.name or "")
    assert_response(agent_prompts, expected_prompts)

    resp = await _find_agent_tools(agent_id, client)
    assert_response(resp, [
        AgentToolConfig(agent_id=agent_id, tool_id=DOCS_TOOL_ID, config={"advancedFileProcessing": advanced_file_processing}),
        AgentToolConfig(agent_id=agent_id, tool_id=WEB_TOOL_ID, config={}),
        AgentToolConfig(agent_id=agent_id, tool_id=BROWSER_TOOL_ID, config={}),
    ])
    resp = await find_agent_tool_config_files(agent_id, DOCS_TOOL_ID, client)
    file_id = last_file_id + 1
    assert_response(resp, [FileMetadata(
        id=file_id, name=file_name, status=FileStatus.PROCESSED, file_processor=FileProcessor.BASIC,
        content_type="text/plain; charset=ascii", user_id=USER_ID, timestamp=CURRENT_TIME)])
    resp = await _find_agent_tool_config_file_content(agent_id, DOCS_TOOL_ID, file_id, client)
    resp.raise_for_status()
    assert resp.content == file_content

    resp = await _find_agent_tests(agent_id, client)
    test_thread_id = last_thread_id + 1
    assert_response(resp, [PublicTestCase(
        agent_id=agent_id, thread=Thread(id=test_thread_id, name=test_case_name, user_id=USER_ID, agent_id=agent_id, creation=CURRENT_TIME, is_test_case=True),
        last_update=CURRENT_TIME)])
    resp = await _find_test_case_messages(agent_id, test_thread_id, client)
    assert_response(resp, [ThreadMessagePublic(
        id=last_message_id + 1 + i, thread_id=test_thread_id, origin=message.origin, text=message.text, files=[], timestamp=CURRENT_TIME)
        for i, message in enumerate(test_messages)])


async def _find_agent_prompts(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(AGENT_PROMPTS_PATH.format(agent_id=agent_id))


async def _find_agent_tools(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(AGENT_TOOLS_PATH.format(agent_id=agent_id))


async def _find_agent_tool_config_file_content(agent_id: int, tool_id: str, file_id: int,
        client: AsyncClient) -> Response:
    return await client.get(AGENT_TOOL_FILE_PATH.format(agent_id=agent_id, tool_id=tool_id, file_id=file_id) + "/content")


async def _find_agent_tests(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(TEST_CASES_PATH.format(agent_id=agent_id))


async def _find_test_case_messages(agent_id: int, test_case_id: int, client: AsyncClient) -> Response:
    return await client.get(TEST_CASE_MESSAGES_PATH.format(agent_id=agent_id, test_case_id=test_case_id))


async def _find_test_case(agent_id: int, test_case_id: int, client: AsyncClient) -> dict:
    resp = await client.get(TEST_CASE_PATH.format(agent_id=agent_id, test_case_id=test_case_id))
    resp.raise_for_status()
    return resp.json()


async def test_export_non_visible_agent(client: AsyncClient):
    resp = await _try_export_agent(NON_VISIBLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_export_protected_agent_as_non_editor(client: AsyncClient):
    resp = await _try_export_agent(NON_EDITABLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_403_FORBIDDEN
    assert resp.json()["detail"] == "Editors access required to export this protected agent"


async def test_import_non_editable_agent(client: AsyncClient):
    resp = await _try_import_agent(NON_EDITABLE_AGENT_ID, await _zip_markdown("minimal_agent"), client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def _zip_markdown(markdown_filename: str) -> bytes:
    markdown_path = solve_asset_path(f"agent_distribution/{markdown_filename}.md", __file__)
    async with aiofiles.open(markdown_path) as markdown:
        return _zip_file("agent.md", await markdown.read())


def _zip_file(filename: str, content: bytes | str) -> bytes:
    zip_bytes = BytesIO()
    with ZipFile(zip_bytes, 'w') as zip_file:
        zip_file.writestr(filename, content)
    return zip_bytes.getvalue()


async def test_import_with_invalid_zip_file(client: AsyncClient):
    resp = await _try_import_agent(AGENT_ID, b"invalid", client)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert resp.json()["detail"] == "invalidFileFormat"


async def test_import_with_missing_markdown(client: AsyncClient):
    resp = await _try_import_agent(AGENT_ID, _zip_file("sample.txt", "Hello"), client)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert resp.json()["detail"] == "unsupportedFileStructure"


async def test_import_with_invalid_markdown(client: AsyncClient):
    resp = await _try_import_agent(AGENT_ID, await _zip_markdown("invalid_agent"), client)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert resp.json()["detail"] == "missingRequiredConfiguration"


async def test_import_with_unavailable_tool(client: AsyncClient):
    agent_id = await _create_agent(client)
    result = await _import_agent(agent_id, await _zip_markdown("invalid_tool_agent"), client)
    assert "test" in result["unavailableTools"]
    assert result["toolsRequiringAuthentication"] == []


@freeze_time(CURRENT_TIME)
async def test_import_with_unavailable_tool_still_imports_agent_data(client: AsyncClient):
    agent_id = await _create_agent(client)
    await _import_agent(agent_id, await _zip_markdown("invalid_tool_agent"), client)
    resp = await _find_agent(agent_id, client)
    resp.raise_for_status()
    agent = resp.json()
    assert agent["name"] == "Test Agent"
    assert agent["description"] == "A Test Agent"
    assert agent["systemPrompt"] == "Test instructions"
