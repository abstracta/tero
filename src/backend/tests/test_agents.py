from typing import Any, cast

from sqlmodel import col

from .common import *

from tero.agents.api import AGENTS_PATH, AGENT_PIN_PATH, AGENT_PATH, AGENT_TOOL_PATH, AGENT_TOOLS_PATH, \
    AGENT_TOOL_FILE_PATH
from tero.agents.domain import PublicAgent, AgentToolConfig, AutomaticAgentField, LlmTemperature, ReasoningEffort, AgentUpdate, AgentListItem, AgentType
from tero.agents.prompts.api import AGENT_PROMPTS_PATH
from tero.agents.prompts.domain import AgentPromptPublic, AgentPrompt
from tero.files.domain import FileMetadata, FileStatus, FileProcessor
from tero.teams.domain import Team, Role
from tero.tools.docs import DOCS_TOOL_ID
from tero.users.domain import UserListItem


@freeze_time(CURRENT_TIME)
async def test_find_all_user_agents(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_user_agents(client)
    assert_response(resp, _clear_active_users([agents[1], agents[0]]))


async def _find_user_agents(client: AsyncClient, params: Optional[dict[str, Any]] = None) -> Response:
    params = params or {}
    params["pinned"] = True
    return await _find_agents(client, params)


async def _find_agents(client: AsyncClient, params: dict[str, Any]) -> Response:
    return await client.get(AGENTS_PATH, params=params)


def _clear_active_users(agents: List[AgentListItem]) -> List[BaseModel]:
    for agent in agents:
        agent.active_users = 0
    return cast(List[BaseModel], agents)


@freeze_time(CURRENT_TIME)
async def test_find_user_agents_by_text(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_user_agents(client, {"text": "first"})
    assert_response(resp, _clear_active_users([agents[0]]))


@freeze_time(CURRENT_TIME)
async def test_add_user_agent(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _add_user_agent(NON_EDITABLE_AGENT_ID, client)
    resp.raise_for_status()
    resp = await _find_user_agents(client)
    assert_response(resp, _clear_active_users([agents[2], agents[1], agents[0]]))


async def _add_user_agent(agent_id: int, client: AsyncClient) -> Response:
    return await client.post(AGENT_PIN_PATH.format(agent_id=agent_id))


async def test_add_non_visible_agent(client: AsyncClient):
    resp = await _add_user_agent(NON_VISIBLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@freeze_time(CURRENT_TIME)
async def test_remove_user_agent(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _remove_user_agent(AGENT_ID, client)
    resp.raise_for_status()
    resp = await _find_user_agents(client)
    assert_response(resp, _clear_active_users([agents[1]]))


async def _remove_user_agent(agent_id: int, client: AsyncClient) -> Response:
    return await client.delete(AGENT_PIN_PATH.format(agent_id=agent_id))


async def test_remove_non_pinned_agent(client: AsyncClient):
    resp = await _remove_user_agent(NON_EDITABLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


# we freeze time to get expected counts and order of agents
@freeze_time(CURRENT_TIME)
async def test_top_used_agents(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_agents(client, {"sort": "ACTIVE_USERS", "limit": 3})
    assert_response(resp, [agents[1], agents[3], agents[2]])


# we freeze time to get expected counts and order of agents
@freeze_time(CURRENT_TIME)
async def test_top_used_agents_limit(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_agents(client, {"sort": "ACTIVE_USERS", "limit": 1})
    assert_response(resp, [agents[1]])


# we freeze time to get expected counts and order of agents
@freeze_time(CURRENT_TIME)
async def test_newest_agents(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_agents(client, {"sort": "LAST_UPDATE", "limit": 3})
    assert_response(resp, [agents[3], agents[2], agents[1]])


# we freeze time to get expected counts and order of agents
@freeze_time(CURRENT_TIME)
async def test_last_update_agents_limit(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_agents(client, {"sort": "LAST_UPDATE", "limit": 1})
    assert_response(resp, [agents[3]])


# we freeze time to get expected counts and order of agents
@freeze_time(CURRENT_TIME)
async def test_find_agents_by_text(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_agents(client, {"text": "second"})
    assert_response(resp, [agents[1]])


# we freeze time to get expected counts and order of agents
@freeze_time(CURRENT_TIME)
async def test_find_agents_by_text_limit(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_agents(client, {"text": "agent", "limit": 1})
    assert_response(resp, [agents[1]])

# we freeze time to get expected counts and order of agents
@freeze_time(CURRENT_TIME)
async def test_find_own_agents(client: AsyncClient, agents: List[AgentListItem]):
    resp = await _find_agents(client, {"own": True})
    assert_response(resp, [agents[1], agents[0]])


@freeze_time(CURRENT_TIME)
async def test_create_agent(agents: List[AgentListItem], users: dict[int, UserListItem], last_agent_id: int, client: AsyncClient, session: AsyncSession):
    resp = await client.post(AGENTS_PATH)
    resp.raise_for_status()
    resp = await _find_user_agents(client)
    agent_id = last_agent_id + 1
    new_agent = AgentListItem(id=agent_id, name=f"Agent #{agent_id}", last_update=CURRENT_TIME, user=users[0], can_edit=True, active_users=0)
    assert_response(resp, _clear_active_users([new_agent, agents[1], agents[0]]))


@freeze_time(CURRENT_TIME)
async def test_update_agent(client: AsyncClient, users: List[UserListItem], teams: List[Team]):
    update = AgentUpdate(name="Updated Agent", description="Updated description", model_id="gpt-5", icon=TEST_ICON,
                temperature=LlmTemperature.PRECISE, reasoning_effort=ReasoningEffort.MEDIUM, team_id=GLOBAL_TEAM_ID, system_prompt="Updated prompt")
    resp = await _update_agent(AGENT_ID, update, client)
    resp.raise_for_status()
    resp = await _find_agent(AGENT_ID, client)
    assert_response(
        resp,
        PublicAgent(id=AGENT_ID, name=update.name, description=update.description, last_update=CURRENT_TIME, team=teams[0], user_id=USER_ID,
              model_id=cast(str, update.model_id), system_prompt=cast(str, update.system_prompt), temperature=cast(LlmTemperature, update.temperature),
              reasoning_effort=cast(ReasoningEffort, update.reasoning_effort), recursion_limit=20, icon=update.icon, can_edit=True, user=users[0]))


async def _update_agent(agent_id: int, update: AgentUpdate, client: AsyncClient) -> Response:
    return await client.put(AGENT_PATH.format(agent_id=agent_id), json=update.model_dump(mode="json", by_alias=True))


async def _find_agent(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(AGENT_PATH.format(agent_id=agent_id))


async def test_update_agent_type(client: AsyncClient):
    resp = await _update_agent(AGENT_ID, AgentUpdate(agent_type=AgentType.DEEP_AGENT), client)
    resp.raise_for_status()
    assert resp.json()["agentType"] == AgentType.DEEP_AGENT.value

    resp = await _update_agent(AGENT_ID, AgentUpdate(agent_type=AgentType.REACT_AGENT), client)
    resp.raise_for_status()
    assert resp.json()["agentType"] == AgentType.REACT_AGENT.value


async def test_update_non_editable_agent(client: AsyncClient):
    resp = await _update_agent(NON_EDITABLE_AGENT_ID, AgentUpdate(name="test"), client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_agent_protected_as_global_owner(client: AsyncClient):
    resp = await _update_agent(AGENT_ID, AgentUpdate(is_protected=True), client)
    resp.raise_for_status()
    assert resp.json()["isProtected"] is True


async def test_update_agent_protected_as_non_global_owner(client: AsyncClient, override_user_role):
    override_user_role(Role.TEAM_EDITOR)
    resp = await _update_agent(AGENT_ID, AgentUpdate(is_protected=True), client)
    assert resp.status_code == status.HTTP_403_FORBIDDEN
    assert resp.json()["detail"] == "Global owner access required to manage protected configuration"


async def test_find_non_visible_agent(client: AsyncClient):
    resp = await _find_agent(NON_VISIBLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_configue_agent_tool(client: AsyncClient):
    await _configure_docs_tool(client)
    resp = await _find_agent_tools(AGENT_ID, client)
    assert_response(resp, [AgentToolConfig(agent_id=AGENT_ID, tool_id=DOCS_TOOL_ID, config={"advancedFileProcessing": False})])


async def _configure_docs_tool(client: AsyncClient, file_processor: FileProcessor = FileProcessor.BASIC):
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID, {"advancedFileProcessing": file_processor == FileProcessor.ENHANCED}, client)


async def _find_agent_tools(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(AGENT_TOOLS_PATH.format(agent_id=agent_id))


async def test_configure_non_editable_agent_tool(client: AsyncClient):
    resp = await try_configure_agent_tool(NON_EDITABLE_AGENT_ID, DOCS_TOOL_ID, {}, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_configure_invalid_tool(client: AsyncClient):
    resp = await try_configure_agent_tool(AGENT_ID, "invalid", {}, client)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST


async def test_configure_tool_with_invalid_config(client: AsyncClient):
    resp = await try_configure_agent_tool(AGENT_ID, DOCS_TOOL_ID, {"test": "val"}, client)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST


async def test_get_non_visible_agent_tools(client: AsyncClient):
    resp = await _find_agent_tools(NON_VISIBLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_get_protected_agent_tools_as_non_editor(client: AsyncClient):
    resp = await _find_agent_tools(NON_EDITABLE_AGENT_ID, client)
    assert resp.status_code == status.HTTP_403_FORBIDDEN
    assert resp.json()["detail"] == "Editors access required to view this agent's tool configs"


async def test_remove_agent_tool_config(client: AsyncClient):
    await _configure_docs_tool(client)
    resp = await _remove_agent_tool_config(AGENT_ID, DOCS_TOOL_ID, client)
    resp.raise_for_status()
    resp = await _find_agent_tools(AGENT_ID, client)
    assert resp.json() == []


async def _remove_agent_tool_config(agent_id: int, tool_id: str, client: AsyncClient) -> Response:
    return await client.delete(AGENT_TOOL_PATH.format(agent_id=agent_id, tool_id=tool_id))


async def test_remove_non_editable_agent_tool_config(client: AsyncClient):
    resp = await _remove_agent_tool_config(NON_EDITABLE_AGENT_ID, DOCS_TOOL_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_remove_unconfigured_tool(client: AsyncClient):
    resp = await _remove_agent_tool_config(AGENT_ID, DOCS_TOOL_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@freeze_time(CURRENT_TIME)
@pytest.mark.usefixtures("stub_docs_tool_generate_description")
async def test_upload_agent_tool_file(client: AsyncClient):
    await _configure_docs_tool(client)
    filename = "test.txt"
    file_content = b"Hello"
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client, filename, file_content)
    resp = await _await_docs_tool_file_processed(file_id, client)
    assert_response(resp, [_build_uploaded_file_metadata(file_id, filename)])
    resp = await _find_agent_tool_config_file_content(AGENT_ID, DOCS_TOOL_ID, file_id, client)
    resp.raise_for_status()
    assert resp.content == file_content


async def _await_docs_tool_file_processed(file_id: int, client: AsyncClient) -> Response:
    return await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id, client)


def _build_uploaded_file_metadata(file_id: int, filename: str, content_type: str = "text/plain; charset=ascii") -> FileMetadata:
    return FileMetadata(id=file_id, status=FileStatus.PROCESSED, name=filename, content_type=content_type, user_id=USER_ID, timestamp=CURRENT_TIME, file_processor=FileProcessor.BASIC)


async def _find_agent_tool_config_file_content(agent_id: int, tool_id: str, file_id: int,
        client: AsyncClient) -> Response:
    return await client.get(
        AGENT_TOOL_FILE_PATH.format(agent_id=agent_id, tool_id=tool_id, file_id=file_id) + "/content")


async def test_upload_file_to_non_editable_agent_tool(client: AsyncClient):
    resp = await try_upload_agent_tool_config_file(NON_EDITABLE_AGENT_ID, DOCS_TOOL_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_upload_file_to_unconfigured_tool(client: AsyncClient):
    resp = await try_upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_find_non_visible_tool_files(client: AsyncClient):
    resp = await find_agent_tool_config_files(NON_VISIBLE_AGENT_ID, DOCS_TOOL_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_find_unconfigured_tool_files(client: AsyncClient):
    resp = await find_agent_tool_config_files(AGENT_ID, DOCS_TOOL_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_find_non_visible_tool_file_content(client: AsyncClient):
    resp = await _find_agent_tool_config_file_content(NON_VISIBLE_AGENT_ID, DOCS_TOOL_ID, 1, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_find_unconfigured_tool_file_content(client: AsyncClient):
    resp = await _find_agent_tool_config_file_content(AGENT_ID, DOCS_TOOL_ID, 2, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_find_non_existent_file_content(client: AsyncClient):
    await _configure_docs_tool(client)
    resp = await _find_agent_tool_config_file_content(AGENT_ID, DOCS_TOOL_ID, 2, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@freeze_time(CURRENT_TIME)
@pytest.mark.usefixtures("stub_docs_tool_generate_description")
async def test_update_agent_tool_file(client: AsyncClient):
    await _configure_docs_tool(client)
    filename = "test.txt"
    file_content = b"Hello"
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client, filename, file_content)
    new_content = b"World"
    resp = await _update_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, file_id, client, filename, new_content)
    resp.raise_for_status()
    resp = await _await_docs_tool_file_processed(file_id, client)
    assert_response(resp, [_build_uploaded_file_metadata(file_id, filename)])
    resp = await _find_agent_tool_config_file_content(AGENT_ID, DOCS_TOOL_ID, file_id, client)
    resp.raise_for_status()
    assert resp.content == new_content


async def _update_agent_tool_config_file(agent_id: int, tool_id: str, file_id: int, client: AsyncClient,
        filename: str = "test2.txt", content: bytes = b"World"):
    return await client.put(AGENT_TOOL_FILE_PATH.format(agent_id=agent_id, tool_id=tool_id, file_id=file_id),
                            files={"file": (filename, content)})


async def test_update_non_editable_agent_tool_file(client: AsyncClient):
    resp = await _update_agent_tool_config_file(NON_EDITABLE_AGENT_ID, DOCS_TOOL_ID, 1, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_unconfigured_tool_file(client: AsyncClient):
    resp = await _update_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, 2, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_non_existent_file(client: AsyncClient):
    await _configure_docs_tool(client)
    resp = await _update_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, 2, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.usefixtures("stub_docs_tool_generate_description")
async def test_delete_agent_tool_file(client: AsyncClient):
    await _configure_docs_tool(client)
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client)
    resp = await _delete_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, file_id, client)
    resp.raise_for_status()
    resp = await find_agent_tool_config_files(AGENT_ID, DOCS_TOOL_ID, client)
    assert_response(resp, [])


async def _delete_agent_tool_config_file(agent_id: int, tool_id: str, file_id: int, client: AsyncClient) -> Response:
    return await client.delete(AGENT_TOOL_FILE_PATH.format(agent_id=agent_id, tool_id=tool_id, file_id=file_id))


async def test_delete_non_editable_agent_tool_file(client: AsyncClient):
    resp = await _delete_agent_tool_config_file(NON_EDITABLE_AGENT_ID, DOCS_TOOL_ID, 1, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_delete_unconfigured_tool_file(client: AsyncClient):
    resp = await _delete_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, 2, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_delete_non_existent_file(client: AsyncClient):
    await _configure_docs_tool(client)
    resp = await _delete_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, 2, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND

async def _reprocess_agent_tool_file(client: AsyncClient, file_processor: FileProcessor):
    await _configure_docs_tool(client, FileProcessor.BASIC if file_processor == FileProcessor.ENHANCED else FileProcessor.ENHANCED)

    file_content = await find_asset_bytes("Emma's routine.pdf")

    filename = "Emma's routine.pdf"
    file_id = await upload_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, client, filename, file_content)
    await _await_docs_tool_file_processed(file_id, client)

    await _configure_docs_tool(client, file_processor)

    resp = await _update_agent_tool_config_file(AGENT_ID, DOCS_TOOL_ID, file_id, client, filename, file_content)
    resp.raise_for_status()

    await _await_docs_tool_file_processed(file_id, client)

    resp = await client.get(AGENT_TOOL_FILE_PATH.format(agent_id=AGENT_ID, tool_id=DOCS_TOOL_ID, file_id=file_id))
    resp.raise_for_status()
    toolFile = resp.json()

    assert toolFile["fileProcessor"] == file_processor.value
    content = toolFile["processedContent"]

    expected_content = await find_asset_text(f"pdf_{file_processor.value.lower()}_content.txt")
    assert content.strip() == expected_content.strip()

    return file_id


@freeze_time(CURRENT_TIME)
@pytest.mark.usefixtures("stub_docs_tool_generate_description")
async def test_reprocess_agent_tool_file_basic(client: AsyncClient):
    await _reprocess_agent_tool_file(client, FileProcessor.BASIC)


@pytest.mark.skipif(not env.azure_doc_intelligence_key, reason="Azure Doc Intelligence not configured")
@freeze_time(CURRENT_TIME)
@pytest.mark.usefixtures("stub_docs_tool_generate_description")
async def test_reprocess_agent_tool_file_enhanced(client: AsyncClient):
    await _reprocess_agent_tool_file(client, FileProcessor.ENHANCED)


async def test_generate_agent_name(client: AsyncClient):
    resp = await _generate_agent_field(AGENT_ID, AutomaticAgentField.NAME, client)
    resp.raise_for_status()
    assert resp.text is not None


async def _generate_agent_field(agent_id: int, field: AutomaticAgentField, client: AsyncClient) -> Response:
    return await client.post(AGENT_PATH.format(agent_id=agent_id) + f"/fields/{field.name}")


async def test_generate_non_editable_agent_field(client: AsyncClient):
    resp = await _generate_agent_field(NON_EDITABLE_AGENT_ID, AutomaticAgentField.NAME, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_generate_agent_description(client: AsyncClient):
    resp = await _generate_agent_field(AGENT_ID, AutomaticAgentField.DESCRIPTION, client)
    resp.raise_for_status()
    assert resp.text is not None


async def test_generate_agent_system_prompt(client: AsyncClient):
    resp = await _generate_agent_field(AGENT_ID, AutomaticAgentField.SYSTEM_PROMPT, client)
    resp.raise_for_status()
    assert resp.text is not None


@freeze_time(CURRENT_TIME)
async def test_clone_agent(users: dict[int, UserListItem], last_agent_id: int,client: AsyncClient):
    public_agent_id = 2
    cloned_agent_id = await _clone_agent(public_agent_id, client)
    resp = await _find_agent(cloned_agent_id, client)
    assert_response(resp, PublicAgent(
        id=last_agent_id + 1,
        name="Agent 2 (copy)",
        description="This is the second agent",
        last_update=CURRENT_TIME,
        team=None,
        icon=None,
        user_id=USER_ID,
        can_edit=True,
        model_id="gpt-5-mini",
        system_prompt="You are a helpful AI agent.",
        temperature=LlmTemperature.CREATIVE,
        reasoning_effort=ReasoningEffort.LOW,
        recursion_limit=20,
        user=users[0],
        agent_type=AgentType.DEEP_AGENT
    ))


async def _clone_agent(agent_id: int, client: AsyncClient) -> int:
    resp = await client.post(f"{AGENT_PATH.format(agent_id=agent_id)}/clone")
    resp.raise_for_status()
    return resp.json()["id"]


async def test_clone_protected_agent_as_non_editor(client: AsyncClient):
    resp = await client.post(f"{AGENT_PATH.format(agent_id=NON_EDITABLE_AGENT_ID)}/clone")
    assert resp.status_code == status.HTTP_403_FORBIDDEN
    assert resp.json()["detail"] == "Editors access required to clone this protected agent"


@pytest.fixture(name="last_prompt_id")
async def last_prompt_id_fixture(session: AsyncSession) -> int:
    return await find_last_id(col(AgentPrompt.id), session)


@freeze_time(CURRENT_TIME)
async def test_clone_agent_prompts(last_prompt_id: int,client: AsyncClient):
    cloned_agent_id = await _clone_agent(AGENT_ID, client)
    resp = await _find_agent_prompts(cloned_agent_id, client)
    assert_response(resp, [
        AgentPromptPublic(id=last_prompt_id + 1, name="Test prompt private 1", content="Test prompt content", shared=False,
            last_update=CURRENT_TIME, user_id=USER_ID, can_edit=True, starter=False),
        AgentPromptPublic(id=last_prompt_id + 2, name="Test prompt shared", content="Test shared prompt content", shared=True,
            last_update=CURRENT_TIME, user_id=USER_ID, can_edit=True, starter=False)])


async def _find_agent_prompts(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(AGENT_PROMPTS_PATH.format(agent_id=agent_id))


async def test_find_default_agent(client: AsyncClient, teams: List[Team]):
    resp = await client.get(AGENTS_PATH + "/default")
    assert_response(resp, PublicAgent(id=6, name="GPT-5 Nano", description="This is the default agent", last_update=PAST_TIME, team=teams[0], user_id=None,
              model_id="gpt-5-nano", system_prompt="You are a helpful AI agent.", temperature=LlmTemperature.NEUTRAL, reasoning_effort=ReasoningEffort.LOW, recursion_limit=20, icon=None, can_edit=True, user=None))
