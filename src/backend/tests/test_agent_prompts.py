from typing import Any, Callable

from fastapi import status
from freezegun import freeze_time
from sqlmodel import col

from .common import *

from tero.agents.api import AGENT_PATH
from tero.agents.domain import AgentUpdate
from tero.agents.prompts.api import AGENT_PROMPTS_PATH, AGENT_PROMPT_PATH
from tero.agents.prompts.domain import AgentPromptPublic, AgentPrompt


NON_VISIBLE_AGENT_PROMPT_ID = 3
SHARED_AGENT_ID = 2
INACCESSIBLE_TEAM_ID = 4


@pytest.fixture(name="prompts")
def agent_prompts_fixture() -> List[AgentPromptPublic]:
    return [
        AgentPromptPublic(id=1, name="Test prompt private 1", content="Test prompt content",
                          last_update=CURRENT_TIME, shared=False,
                          user_id=USER_ID, can_edit=True, starter=False),
        AgentPromptPublic(id=2, name="Test prompt shared", content="Test shared prompt content",
                          last_update=CURRENT_TIME, shared=True,
                          user_id=OTHER_USER_ID, can_edit=True, starter=False),
        AgentPromptPublic(id=NON_VISIBLE_AGENT_PROMPT_ID, name="Test prompt private 2", content="Test prompt content 2",
                          last_update=CURRENT_TIME, shared=False, user_id=OTHER_USER_ID,
                          can_edit=False, starter=False),
    ]


@pytest.fixture(name="last_prompt_id")
async def last_prompt_id_fixture(session: AsyncSession) -> int:
    return await find_last_id(col(AgentPrompt.id), session)


async def test_find_agent_prompts(client: AsyncClient, prompts: List[AgentPromptPublic]):
    resp = await _find_agents_prompts(AGENT_ID, client)
    assert_response(resp, [prompts[0], prompts[1]])


@pytest.mark.parametrize("agent_id", [NON_VISIBLE_AGENT_ID, 99999])
async def test_find_prompts_without_readable_content_returns_empty_list(client: AsyncClient, agent_id: int):
    assert_response(await _find_agents_prompts(agent_id, client), [])


@pytest.mark.parametrize("team_id", [None, INACCESSIBLE_TEAM_ID], ids=["private", "other-team"])
async def test_inaccessible_agent_without_private_prompts_returns_empty_list(client: AsyncClient, override_user: Callable[[int], None], team_id: Optional[int]):
    for shared in [False, True]:
        resp = await add_agent_prompt(SHARED_AGENT_ID, client, name="Owner prompt", content="Owner content", shared=shared)
        resp.raise_for_status()
    resp = await client.put(AGENT_PATH.format(agent_id=SHARED_AGENT_ID), json=AgentUpdate(team_id=team_id).model_dump(mode="json", by_alias=True))
    resp.raise_for_status()

    override_user(OTHER_USER_ID)
    assert_response(await _find_agents_prompts(SHARED_AGENT_ID, client), [])


@pytest.mark.parametrize("team_id", [None, INACCESSIBLE_TEAM_ID], ids=["private", "other-team"])
async def test_inaccessible_agent_only_returns_private_prompts(client: AsyncClient, override_user: Callable[[int], None], team_id: Optional[int]):
    for shared in [False, True]:
        resp = await add_agent_prompt(SHARED_AGENT_ID, client, name="Owner prompt", content="Owner content", shared=shared)
        resp.raise_for_status()

    override_user(OTHER_USER_ID)
    resp = await add_agent_prompt(SHARED_AGENT_ID, client, name="Private prompt", content="Private content", shared=False)
    resp.raise_for_status()
    private_prompt = AgentPromptPublic.model_validate(resp.json())

    override_user(USER_ID)
    resp = await client.put(AGENT_PATH.format(agent_id=SHARED_AGENT_ID), json=AgentUpdate(team_id=team_id).model_dump(mode="json", by_alias=True))
    resp.raise_for_status()

    override_user(OTHER_USER_ID)
    private_prompt.can_edit = False
    assert_response(await _find_agents_prompts(SHARED_AGENT_ID, client), [private_prompt])


@pytest.mark.parametrize("team_id", [None, INACCESSIBLE_TEAM_ID], ids=["private", "other-team"])
@pytest.mark.parametrize("method, path, payload", [
    ("POST", AGENT_PROMPTS_PATH, {"name": "New prompt", "content": "New content"}),
    ("PUT", AGENT_PROMPT_PATH, {"name": "Changed prompt", "content": "Changed content"}),
    ("DELETE", AGENT_PROMPT_PATH, None),
], ids=["create", "update", "delete"])
async def test_inaccessible_agent_rejects_prompt_writes(
        client: AsyncClient, override_user: Callable[[int], None], team_id: Optional[int],
        method: str, path: str, payload: Optional[dict[str, str]]):
    override_user(OTHER_USER_ID)
    resp = await add_agent_prompt(SHARED_AGENT_ID, client, name="Private prompt", content="Private content", shared=False)
    resp.raise_for_status()
    private_prompt = AgentPromptPublic.model_validate(resp.json())

    override_user(USER_ID)
    resp = await client.put(AGENT_PATH.format(agent_id=SHARED_AGENT_ID), json=AgentUpdate(team_id=team_id).model_dump(mode="json", by_alias=True))
    resp.raise_for_status()

    override_user(OTHER_USER_ID)
    resp = await client.request(method, path.format(agent_id=SHARED_AGENT_ID, prompt_id=private_prompt.id), json=payload)
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    private_prompt.can_edit = False
    assert_response(await _find_agents_prompts(SHARED_AGENT_ID, client), [private_prompt])


async def _find_agents_prompts(agent_id: int, client: AsyncClient) -> Response:
    return await client.get(AGENT_PROMPTS_PATH.format(agent_id=agent_id))


@freeze_time(CURRENT_TIME)
async def test_add_agent_prompt(client: AsyncClient, prompts: List[AgentPromptPublic], last_prompt_id: int):
    name = "Test added prompt"
    content = "Test added prompt content"
    shared = True
    resp = await add_agent_prompt(AGENT_ID, name=name, content=content, shared=shared, client=client)
    resp.raise_for_status()
    new_prompt = AgentPromptPublic(id=last_prompt_id + 1, name=name, content=content, last_update=CURRENT_TIME, shared=shared, user_id=USER_ID, can_edit=True, starter=False)
    assert_response(resp, new_prompt)
    prompts_res = await _find_agents_prompts(AGENT_ID, client)
    assert_response(prompts_res, [new_prompt, prompts[0], prompts[1]])


async def add_agent_prompt(agent_id: int, client: AsyncClient, name: Optional[str] = None, content: Optional[str] = None, shared: Optional[bool] = None, starter: Optional[bool] = None) -> Response:
    agent_prompt_dict = {"name": name, "content": content, "shared": shared, "starter": starter}
    payload = {key: value for key, value in agent_prompt_dict.items() if value is not None}
    return await client.post(AGENT_PROMPTS_PATH.format(agent_id=agent_id), json=payload)


async def test_remove_agent_prompt(client: AsyncClient, prompts: List[AgentPromptPublic]):
    resp = await _remove_agent_prompt(AGENT_ID, 1, client)
    resp.raise_for_status()
    resp_prompts = await _find_agents_prompts(AGENT_ID, client)
    assert_response(resp_prompts, [prompts[1]])


async def _remove_agent_prompt(agent_id: int, prompt_id: int, client: AsyncClient) -> Response:
    return await client.delete(AGENT_PROMPT_PATH.format(agent_id=agent_id, prompt_id=prompt_id))


async def test_remove_non_visible_agent_prompt(client: AsyncClient):
    resp = await _remove_agent_prompt(AGENT_ID, NON_VISIBLE_AGENT_PROMPT_ID, client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@freeze_time(CURRENT_TIME)
async def test_update_agent_prompt(client: AsyncClient, prompts: List[AgentPromptPublic]):
    name = "Updated Prompt"
    content = "Updated content"
    shared = True
    resp = await _update_agent_prompt(
        AGENT_ID, 1, 
        {"name": name, "content": content, "shared": shared},
        client)
    resp.raise_for_status()
    resp = await _find_agents_prompts(AGENT_ID, client)
    assert_response(
        resp,
        [
            prompts[1],
            AgentPromptPublic(id=1, name=name, content=content, last_update=CURRENT_TIME, shared=shared, user_id=USER_ID, can_edit=True, starter=False),
        ]
    )


async def _update_agent_prompt(agent_id: int, prompt_id: int, body: dict[str, Any], client: AsyncClient) -> Response:
    return await client.put(AGENT_PROMPT_PATH.format(agent_id=agent_id, prompt_id=prompt_id),
                            json=body)


async def test_update_non_visible_agent_prompt(client: AsyncClient):
    name = "Updated Prompt"
    content = "Updated content"
    shared = True
    resp = await _update_agent_prompt(
        AGENT_ID, NON_VISIBLE_AGENT_PROMPT_ID, 
        {"name": name, "content": content, "shared": shared},
        client)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


@freeze_time(CURRENT_TIME)
async def test_add_starter_prompt(client: AsyncClient, prompts: List[AgentPromptPublic], last_prompt_id: int):
    name = "Starter prompt"
    content = "Initial suggestion"
    shared = False
    starter = True
    resp = await add_agent_prompt(AGENT_ID, name=name, content=content, shared=shared, starter=starter, client=client)
    resp.raise_for_status()

    new_prompt = AgentPromptPublic(
        id=last_prompt_id + 1,
        name=name,
        content=content,
        last_update=CURRENT_TIME,
        shared=shared,
        user_id=USER_ID,
        can_edit=True,
        starter=starter,
    )

    assert_response(resp, new_prompt)

    prompts_res = await _find_agents_prompts(AGENT_ID, client)
    assert_response(prompts_res, [new_prompt, prompts[0], prompts[1]])
