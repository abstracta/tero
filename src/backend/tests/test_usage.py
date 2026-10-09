from datetime import timedelta
from typing import Callable

from httpx import AsyncClient

from .common import *

from tero.agents.domain import AgentListItem
from tero.external_agents.domain import PublicExternalAgent
from tero.teams.domain import Role, Team, MY_TEAM_ID
from tero.usage.api import IMPACT_PATH, USAGE_PATH
from tero.usage.domain import AgentImpactItem, UserImpactItem, ImpactSummary, UsageSummary, AgentUsageItem, UserUsageItem, PRIVATE_AGENT_ID
from tero.users.domain import UserListItem


async def test_user_budget(client: AsyncClient):
    assert await _find_user_budget(client) == {"usagePercent": 0.0}
    async with add_message_to_thread(client, THREAD_ID, "Hello") as resp:
        resp.raise_for_status()
    budget = await _find_user_budget(client)
    usage = budget.get("usagePercent")
    assert usage and usage > 0.0


async def _find_user_budget(client: AsyncClient) -> dict:
    resp = await client.get(f"{BASE_PATH}/budget")
    resp.raise_for_status()
    return resp.json()


@freeze_time(CURRENT_TIME)
async def test_impact_summary_metrics(client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_summary_metrics(from_date, to_date, 1, client)

    assert_response(resp, ImpactSummary(human_hours=640, ai_hours=6, previous_human_hours=800, previous_ai_hours=2))


async def _get_impact_summary_metrics(from_date: datetime, to_date: datetime, team_id: int, client: AsyncClient) -> Response:
    return await client.get(
        f"{IMPACT_PATH}/summary",
        params={"from_date": from_date.isoformat(), "to_date": to_date.isoformat(), "team_id": team_id},
    )


@freeze_time(CURRENT_TIME)
async def test_impact_summary_metrics_team(client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_summary_metrics(from_date, to_date, 2, client)

    assert_response(resp, ImpactSummary(human_hours=320, ai_hours=5, previous_human_hours=320, previous_ai_hours=2))


@freeze_time(CURRENT_TIME)
async def test_impact_summary_metrics_me(client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_summary_metrics(from_date, to_date, MY_TEAM_ID, client)

    assert_response(resp, ImpactSummary(human_hours=160, ai_hours=4, previous_human_hours=160, previous_ai_hours=1))


@freeze_time(CURRENT_TIME)
async def test_impact_summary_metrics_team_no_owner_access(override_user: Callable[[int], None], client: AsyncClient):
    override_user(OTHER_USER_ID)
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_summary_metrics(from_date, to_date, 3, client)
    assert resp.status_code == 403


async def test_summary_metrics_date_validation(client: AsyncClient):
    from_date = datetime.now()
    to_date = (from_date - timedelta(days=1))
    resp = await _get_impact_summary_metrics(from_date, to_date, 1, client)
    assert resp.status_code == 400


@pytest.fixture(name="external_agents")
def fixture_external_agents() -> List[PublicExternalAgent]:
    return [
        PublicExternalAgent(id=1, name="ChatGPT"),
        PublicExternalAgent(id=2, name="Cursor"),
        PublicExternalAgent(id=3, name="Claude")
    ]


@freeze_time(CURRENT_TIME)
async def test_impact_top_agents(teams: List[Team], users: List[UserListItem], agents: List[AgentListItem], external_agents: List[PublicExternalAgent], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_agents(from_date, to_date, 1, 10, client)
    assert_response(resp, [
        AgentImpactItem(agent_id=PRIVATE_AGENT_ID, agent_name=None, minutes_saved=145, active_users=3, icon_bytes=None, team=None, previous_minutes_saved=0, previous_active_users=0, author_name=None),
        AgentImpactItem(agent_id=external_agents[1].id, agent_name=external_agents[1].name, minutes_saved=120, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=0, previous_active_users=0, author_name=None, is_external_agent=True),
        AgentImpactItem(agent_id=external_agents[0].id, agent_name=external_agents[0].name, minutes_saved=60, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=0, previous_active_users=0, author_name=None, is_external_agent=True),
        AgentImpactItem(agent_id=agents[1].id, agent_name=agents[1].name, minutes_saved=35, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=60, previous_active_users=1, author_name=users[0].name),
        AgentImpactItem(agent_id=external_agents[2].id, agent_name=external_agents[2].name, minutes_saved=0, active_users=0, icon_bytes=None, team=teams[0], previous_minutes_saved=60, previous_active_users=1, author_name=None, is_external_agent=True),
    ])


async def _get_impact_top_agents(from_date: datetime, to_date: datetime, team_id: int, limit: int, client: AsyncClient) -> Response:
    return await client.get(
        f"{IMPACT_PATH}/agents",
        params={"from_date": from_date.isoformat(), "to_date": to_date.isoformat(), "team_id": team_id, "limit": limit}
    )


@freeze_time(CURRENT_TIME)
async def test_impact_top_agents_team(teams: List[Team], users: List[UserListItem], agents: List[AgentListItem], external_agents: List[PublicExternalAgent], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_agents(from_date, to_date, 2, 10, client)
    assert_response(resp, [
        AgentImpactItem(agent_id=external_agents[1].id, agent_name=external_agents[1].name, minutes_saved=120, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=0, previous_active_users=0, author_name=None, is_external_agent=True),
        AgentImpactItem(agent_id=agents[3].id, agent_name=agents[3].name, minutes_saved=60, active_users=1, icon_bytes=None, team=teams[1], previous_minutes_saved=0, previous_active_users=0, author_name=users[1].name),
        AgentImpactItem(agent_id=external_agents[0].id, agent_name=external_agents[0].name, minutes_saved=60, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=0, previous_active_users=0, author_name=None, is_external_agent=True),
        AgentImpactItem(agent_id=PRIVATE_AGENT_ID, agent_name=None, minutes_saved=35, active_users=1, icon_bytes=None, team=None, previous_minutes_saved=0, previous_active_users=0, author_name=None),
        AgentImpactItem(agent_id=agents[1].id, agent_name=agents[1].name, minutes_saved=35, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=60, previous_active_users=1, author_name=users[0].name),
        AgentImpactItem(agent_id=external_agents[2].id, agent_name=external_agents[2].name, minutes_saved=0, active_users=0, icon_bytes=None, team=teams[0], previous_minutes_saved=60, previous_active_users=1, author_name=None, is_external_agent=True),
    ])


@freeze_time(CURRENT_TIME)
async def test_impact_top_agents_me(teams: List[Team], users: List[UserListItem], agents: List[AgentListItem], external_agents: List[PublicExternalAgent], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_agents(from_date, to_date, MY_TEAM_ID, 10, client)
    assert_response(resp, [
        AgentImpactItem(agent_id=external_agents[1].id, agent_name=external_agents[1].name, minutes_saved=120, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=0, previous_active_users=0, author_name=None, is_external_agent=True),
        AgentImpactItem(agent_id=external_agents[0].id, agent_name=external_agents[0].name, minutes_saved=60, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=0, previous_active_users=0, author_name=None, is_external_agent=True),
        AgentImpactItem(agent_id=agents[0].id, agent_name=agents[0].name, minutes_saved=35, active_users=1, icon_bytes=None, team=None, previous_minutes_saved=0, previous_active_users=0, author_name=users[0].name),
        AgentImpactItem(agent_id=agents[1].id, agent_name=agents[1].name, minutes_saved=35, active_users=1, icon_bytes=None, team=teams[0], previous_minutes_saved=0, previous_active_users=0, author_name=users[0].name),
        AgentImpactItem(agent_id=external_agents[2].id, agent_name=external_agents[2].name, minutes_saved=0, active_users=0, icon_bytes=None, team=teams[0], previous_minutes_saved=60, previous_active_users=1, author_name=None, is_external_agent=True),
    ])


@freeze_time(CURRENT_TIME)
async def test_impact_top_agents_team_no_owner_access(override_user: Callable[[int], None], client: AsyncClient):
    override_user(OTHER_USER_ID)
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_agents(from_date, to_date, 3, 10, client)
    assert resp.status_code == 403


async def test_impact_top_agents_date_validation(client: AsyncClient):
    from_date = datetime.now()
    to_date = (from_date - timedelta(days=1))
    resp = await _get_impact_top_agents(from_date, to_date, 1, 10, client)
    assert resp.status_code == 400


@freeze_time(CURRENT_TIME)
async def test_impact_top_users(users: List[UserListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_users(from_date, to_date, 1, 10, client)

    assert_response(resp, [UserImpactItem(user_id=users[0].id, user_name=users[0].name, minutes_saved=250, monthly_hours=160, previous_minutes_saved=60),
                           UserImpactItem(user_id=users[1].id, user_name=users[1].name, minutes_saved=60, monthly_hours=160, previous_minutes_saved=60),
                           UserImpactItem(user_id=users[2].id, user_name=users[2].name, minutes_saved=50, monthly_hours=160, previous_minutes_saved=0),
                           UserImpactItem(user_id=users[3].id, user_name=users[3].name, minutes_saved=0, monthly_hours=160, previous_minutes_saved=0)])


async def _get_impact_top_users(from_date: datetime, to_date: datetime, team_id: int, limit: int, client: AsyncClient) -> Response:
    return await client.get(
        f"{IMPACT_PATH}/users",
        params={"from_date": from_date.isoformat(), "to_date": to_date.isoformat(), "team_id": team_id, "limit": limit}
    )


@freeze_time(CURRENT_TIME)
async def test_impact_top_users_team(users: List[UserListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_users(from_date, to_date, 2, 10, client)

    assert_response(resp, [UserImpactItem(user_id=users[0].id, user_name=users[0].name, minutes_saved=250, monthly_hours=160, previous_minutes_saved=60),
                           UserImpactItem(user_id=users[1].id, user_name=users[1].name, minutes_saved=60, monthly_hours=160, previous_minutes_saved=60)])


@freeze_time(CURRENT_TIME)
async def test_impact_top_users_team_no_owner_access(override_user: Callable[[int], None], client: AsyncClient):
    override_user(OTHER_USER_ID)
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_users(from_date, to_date, 3, 10, client)
    assert resp.status_code == 403

@freeze_time(CURRENT_TIME)
async def test_impact_top_users_team_with_pending_and_rejected_status(users: List[UserListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_impact_top_users(from_date, to_date, 4, 10, client)

    assert_response(resp, [UserImpactItem(user_id=users[0].id, user_name=users[0].name, minutes_saved=250, monthly_hours=160, previous_minutes_saved=60),
                           UserImpactItem(user_id=users[1].id, user_name=users[1].name, minutes_saved=60, monthly_hours=160, previous_minutes_saved=60),
                           UserImpactItem(user_id=users[3].id, user_name=users[3].name, minutes_saved=0, monthly_hours=160, previous_minutes_saved=0)])


async def test_impact_top_users_date_validation(client: AsyncClient):
    from_date = datetime.now()
    to_date = (from_date - timedelta(days=1))

    resp = await _get_impact_top_users(from_date, to_date, 1, 10, client)
    assert resp.status_code == 400


TODAY = datetime.now()
PARAMS = {
    "from_date": (TODAY - timedelta(days=30)).isoformat(),
    "to_date": TODAY.isoformat(),
    "team_id": 1,
}


async def test_get_impact_summary_forbidden(override_user_role: Callable[[Role], None], client: AsyncClient):
    override_user_role(Role.TEAM_MEMBER)
    resp = await client.get(f"{IMPACT_PATH}/summary", params=PARAMS)
    assert resp.status_code == status.HTTP_403_FORBIDDEN


async def test_get_impact_agents_forbidden(override_user_role: Callable[[Role], None], client: AsyncClient):
    override_user_role(Role.TEAM_MEMBER)
    resp = await client.get(f"{IMPACT_PATH}/agents", params=PARAMS)
    assert resp.status_code == status.HTTP_403_FORBIDDEN


async def test_get_impact_users_forbidden(override_user_role: Callable[[Role], None], client: AsyncClient):
    override_user_role(Role.TEAM_MEMBER)
    resp = await client.get(f"{IMPACT_PATH}/users", params=PARAMS)
    assert resp.status_code == status.HTTP_403_FORBIDDEN


@freeze_time(CURRENT_TIME)
async def test_usage_summary_metrics(client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_summary_metrics(from_date, to_date, 1, client)

    assert_response(resp, UsageSummary(active_users=3, total_threads=4, previous_active_users=1, previous_total_threads=1))


@freeze_time(CURRENT_TIME)
async def test_usage_summary_metrics_team(client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_summary_metrics(from_date, to_date, 2, client)

    assert_response(resp, UsageSummary(active_users=2, total_threads=3, previous_active_users=1, previous_total_threads=1))


@freeze_time(CURRENT_TIME)
async def test_usage_summary_metrics_me(client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_summary_metrics(from_date, to_date, MY_TEAM_ID, client)

    assert_response(resp, UsageSummary(active_users=1, total_threads=2, previous_active_users=0, previous_total_threads=0))


@freeze_time(CURRENT_TIME)
async def test_usage_summary_metrics_team_no_owner_access(override_user: Callable[[int], None], client: AsyncClient):
    override_user(OTHER_USER_ID)
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_summary_metrics(from_date, to_date, 3, client)
    assert resp.status_code == 403


async def test_usage_summary_date_validation(client: AsyncClient):
    from_date = datetime.now()
    to_date = (from_date - timedelta(days=1))
    resp = await _get_usage_summary_metrics(from_date, to_date, 1, client)
    assert resp.status_code == 400


async def _get_usage_summary_metrics(from_date: datetime, to_date: datetime, team_id: int, client: AsyncClient) -> Response:
    return await client.get(
        f"{USAGE_PATH}/summary",
        params={"from_date": from_date.isoformat(), "to_date": to_date.isoformat(), "team_id": team_id},
    )


@freeze_time(CURRENT_TIME)
async def test_usage_top_agents(teams: List[Team], users: List[UserListItem], agents: List[AgentListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_agents(from_date, to_date, 1, 10, client)

    assert_response(resp, [
        AgentUsageItem(agent_id=PRIVATE_AGENT_ID, agent_name=None, icon_bytes=None, team=None, author_name=None, active_users=3, total_threads=3, previous_active_users=0, previous_total_threads=0),
        AgentUsageItem(agent_id=agents[1].id, agent_name=agents[1].name, icon_bytes=None, team=teams[0], author_name=users[0].name, active_users=1, total_threads=1, previous_active_users=1, previous_total_threads=1)
    ])


@freeze_time(CURRENT_TIME)
async def test_usage_top_agents_team(teams: List[Team], users: List[UserListItem], agents: List[AgentListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_agents(from_date, to_date, 2, 10, client)

    assert_response(resp, [
        AgentUsageItem(agent_id=PRIVATE_AGENT_ID, agent_name=None, icon_bytes=None, team=None, author_name=None, active_users=1, total_threads=1, previous_active_users=0, previous_total_threads=0),
        AgentUsageItem(agent_id=agents[1].id, agent_name=agents[1].name, icon_bytes=None, team=teams[0], author_name=users[0].name, active_users=1, total_threads=1, previous_active_users=1, previous_total_threads=1),
        AgentUsageItem(agent_id=agents[3].id, agent_name=agents[3].name, icon_bytes=None, team=teams[1], author_name=users[1].name, active_users=1, total_threads=1, previous_active_users=0, previous_total_threads=0),
    ])


@freeze_time(CURRENT_TIME)
async def test_usage_top_agents_me(teams: List[Team], users: List[UserListItem], agents: List[AgentListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_agents(from_date, to_date, MY_TEAM_ID, 10, client)

    assert_response(resp, [
        AgentUsageItem(agent_id=agents[0].id, agent_name=agents[0].name, icon_bytes=None, team=None, author_name=users[0].name, active_users=1, total_threads=1, previous_active_users=0, previous_total_threads=0),
        AgentUsageItem(agent_id=agents[1].id, agent_name=agents[1].name, icon_bytes=None, team=teams[0], author_name=users[0].name, active_users=1, total_threads=1, previous_active_users=0, previous_total_threads=0),
    ])


@freeze_time(CURRENT_TIME)
async def test_usage_top_agents_team_no_owner_access(override_user: Callable[[int], None], client: AsyncClient):
    override_user(OTHER_USER_ID)
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_agents(from_date, to_date, 3, 10, client)
    assert resp.status_code == 403


async def test_usage_top_agents_date_validation(client: AsyncClient):
    from_date = datetime.now()
    to_date = (from_date - timedelta(days=1))
    resp = await _get_usage_top_agents(from_date, to_date, 1, 10, client)
    assert resp.status_code == 400


async def _get_usage_top_agents(from_date: datetime, to_date: datetime, team_id: int, limit: int,client: AsyncClient) -> Response:
    return await client.get(
        f"{USAGE_PATH}/agents",
        params={"from_date": from_date.isoformat(), "to_date": to_date.isoformat(), "team_id": team_id, "limit": limit}
    )


@freeze_time(CURRENT_TIME)
async def test_usage_top_users(users: List[UserListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_users(from_date, to_date, 1, 10, client)

    assert_response(resp, [
        UserUsageItem(user_id=users[0].id, user_name=users[0].name, total_threads=2, previous_total_threads=0),
        UserUsageItem(user_id=users[1].id, user_name=users[1].name, total_threads=1, previous_total_threads=1),
        UserUsageItem(user_id=users[2].id, user_name=users[2].name, total_threads=1, previous_total_threads=0),
    ])


@freeze_time(CURRENT_TIME)
async def test_usage_top_users_team(users: List[UserListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_users(from_date, to_date, 2, 10, client)

    assert_response(resp, [
        UserUsageItem(user_id=users[0].id, user_name=users[0].name, total_threads=2, previous_total_threads=0),
        UserUsageItem(user_id=users[1].id, user_name=users[1].name, total_threads=1, previous_total_threads=1),
    ])


@freeze_time(CURRENT_TIME)
async def test_usage_top_users_team_no_owner_access(override_user: Callable[[int], None], client: AsyncClient):
    override_user(OTHER_USER_ID)
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_users(from_date, to_date, 3, 10, client)
    assert resp.status_code == 403


@freeze_time(CURRENT_TIME)
async def test_usage_top_users_team_with_pending_and_rejected_status(users: List[UserListItem], client: AsyncClient):
    to_date = datetime.now()
    from_date = (to_date - timedelta(days=30))
    resp = await _get_usage_top_users(from_date, to_date, 4, 10, client)

    assert_response(resp, [
        UserUsageItem(user_id=users[0].id, user_name=users[0].name, total_threads=2, previous_total_threads=0),
        UserUsageItem(user_id=users[1].id, user_name=users[1].name, total_threads=1, previous_total_threads=1),
    ])


async def test_usage_top_users_date_validation(client: AsyncClient):
    from_date = datetime.now()
    to_date = (from_date - timedelta(days=1))
    resp = await _get_usage_top_users(from_date, to_date, 1, 10, client)
    assert resp.status_code == 400


async def _get_usage_top_users(from_date: datetime, to_date: datetime, team_id: int, limit: int, client: AsyncClient) -> Response:
    return await client.get(
        f"{USAGE_PATH}/users",
        params={"from_date": from_date.isoformat(), "to_date": to_date.isoformat(), "team_id": team_id, "limit": limit}
    )


async def test_get_usage_summary_forbidden(override_user_role: Callable[[Role], None], client: AsyncClient):
    override_user_role(Role.TEAM_MEMBER)
    resp = await client.get(f"{USAGE_PATH}/summary", params=PARAMS)
    assert resp.status_code == status.HTTP_403_FORBIDDEN


async def test_get_usage_agents_forbidden(override_user_role: Callable[[Role], None], client: AsyncClient):
    override_user_role(Role.TEAM_MEMBER)
    resp = await client.get(f"{USAGE_PATH}/agents", params=PARAMS)
    assert resp.status_code == status.HTTP_403_FORBIDDEN


async def test_get_usage_users_forbidden(override_user_role: Callable[[Role], None], client: AsyncClient):
    override_user_role(Role.TEAM_MEMBER)
    resp = await client.get(f"{USAGE_PATH}/users", params=PARAMS)
    assert resp.status_code == status.HTTP_403_FORBIDDEN


@freeze_time(CURRENT_TIME)
async def test_exclude_user_from_metrics(client: AsyncClient):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=30)

    resp = await _get_impact_summary_metrics(from_date, to_date, 1, client)
    baseline_hours = resp.json()["humanHours"]

    await client.put(f"{BASE_PATH}/users/{OTHER_USER_ID}", json={"excludeFromMetrics": True})

    resp = await _get_impact_summary_metrics(from_date, to_date, 1, client)
    excluded_hours = resp.json()["humanHours"]

    assert baseline_hours - excluded_hours == 160
