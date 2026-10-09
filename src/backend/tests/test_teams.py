from typing import Callable

from .common import *

from tero.agents.api import AGENTS_PATH
from tero.teams.api import TEAM_USER_PATH, TEAM_USERS_PATH, TEAMS_PATH, TEAM_PATH
from tero.teams.domain import Team, Role, TeamCreate, TeamRole, TeamRoleStatus, TeamUpdate, TeamUser
from tero.users.api import CURRENT_USER_PATH, CURRENT_USER_TEAM_PATH
from tero.users.domain import UserProfile, PublicTeamRole


async def test_get_users_from_team(client: AsyncClient):
    users = await _get_team_users(client, 2)
    assert_response(users, [TeamUser(id=OTHER_USER_ID, name="Jane Doe", username="test2", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                           TeamUser(id=USER_ID, name="John Doe", username="test", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True)])


async def _get_team_users(client: AsyncClient, team_id: int) -> Response:
    return await client.get(TEAM_USERS_PATH.format(team_id=team_id))


async def test_get_users_from_team_not_found(client: AsyncClient):
    resp = await client.get(TEAM_USERS_PATH.format(team_id=5))
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_add_users_to_team(client: AsyncClient, teams: list[Team], override_user: Callable[[int], None]):
    user_id = 5
    user_role = Role.TEAM_OWNER
    resp = await _send_invitation(client, 2, user_id, user_role)
    resp.raise_for_status()
    override_user(user_id)
    user_profile = await _get_user_profile(client)
    assert_response(user_profile, UserProfile(teams=[PublicTeamRole(id=2, name=teams[1].name, role=user_role, status=TeamRoleStatus.PENDING)]))


async def _send_invitation(client: AsyncClient, team_id: int, user_id: int, user_role: Role):
    return await client.post(TEAM_USERS_PATH.format(team_id=team_id), json=[{"username": f"test{user_id}", "role": user_role}])


async def _get_user_profile(client: AsyncClient) -> Response:
    return await client.get(CURRENT_USER_PATH)


async def test_add_non_existing_user_to_team(client: AsyncClient):
    resp = await client.post(TEAM_USERS_PATH.format(team_id=2), json=[{"username": "testUser", "role": Role.TEAM_OWNER}])
    assert resp.status_code == status.HTTP_200_OK


async def test_add_users_to_team_not_found(client: AsyncClient):
    resp = await client.post(TEAM_USERS_PATH.format(team_id=5), json=[{"username": "testUser", "role": Role.TEAM_MEMBER}])
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_add_users_to_team_not_authorized(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(OTHER_USER_ID)
    resp = await client.post(TEAM_USERS_PATH.format(team_id=3), json=[{"username": "testUser", "role": Role.TEAM_MEMBER}])
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_accept_team_invitation(client: AsyncClient, teams: list[Team], override_user: Callable[[int], None]):
    user_id = 5
    user_role = Role.TEAM_OWNER
    team_id = teams[1].id
    await send_invitation_and_switch_user(client, override_user, team_id, user_id, user_role)
    resp_accept = await client.put(CURRENT_USER_TEAM_PATH.format(team_id=team_id), json={"teamId": team_id})
    resp_accept.raise_for_status()
    user_profile = await _get_user_profile(client)
    assert_response(user_profile, UserProfile(teams=[PublicTeamRole(id=team_id, name=teams[1].name, role=Role.TEAM_OWNER, status=TeamRoleStatus.ACCEPTED)]))


async def send_invitation_and_switch_user(client: AsyncClient, override_user: Callable[[int], None], team_id: int, user_id: int, user_role: Role):
    resp = await _send_invitation(client, team_id, user_id, user_role)
    resp.raise_for_status()
    override_user(user_id)


async def test_accept_non_existent_team_invitation(client: AsyncClient, teams:list[Team]):
    resp = await client.put(CURRENT_USER_TEAM_PATH.format(team_id=3))
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    user_profile = await _get_user_profile(client)
    assert_response(user_profile, UserProfile(teams=[PublicTeamRole(id=teams[0].id, name=teams[0].name, role=Role.TEAM_OWNER, status=TeamRoleStatus.ACCEPTED),
                                                     PublicTeamRole(id=teams[1].id, name=teams[1].name, role=Role.TEAM_OWNER, status=TeamRoleStatus.ACCEPTED),
                                                     PublicTeamRole(id=teams[2].id, name=teams[2].name, role=Role.TEAM_MEMBER, status=TeamRoleStatus.ACCEPTED)]))


async def test_reject_team_invitation(client: AsyncClient, teams: list[Team], override_user: Callable[[int], None]):
    user_id = 5
    user_role = Role.TEAM_OWNER
    team_id = teams[1].id
    await send_invitation_and_reject(client, override_user, team_id, user_id, user_role)
    user_profile = await _get_user_profile(client)
    assert_response(user_profile, UserProfile(teams=[]))


async def send_invitation_and_reject(client: AsyncClient, override_user: Callable[[int], None], team_id: int, user_id: int, user_role: Role):
    await send_invitation_and_switch_user(client, override_user, team_id, user_id, user_role)
    resp_reject = await client.delete(CURRENT_USER_TEAM_PATH.format(team_id=team_id))
    resp_reject.raise_for_status()


async def test_reject_team_invitation_user_not_in_team(client: AsyncClient, teams: list[Team], override_user: Callable[[int], None]):
    user_id = 5
    user_role = Role.TEAM_OWNER
    team_id = teams[1].id
    await send_invitation_and_reject(client, override_user, team_id, user_id, user_role)
    override_user(USER_ID)
    resp = await _get_team_users(client, team_id)
    assert_response(resp, [TeamUser(id=OTHER_USER_ID, name="Jane Doe", username="test2", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                           TeamUser(id=USER_ID, name="John Doe", username="test", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                           TeamUser(id=5, name="John Doe 5", username="test5", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.REJECTED, verified=True)])


async def test_pending_team_invitation_user_not_in_team(client: AsyncClient, teams: list[Team], override_user: Callable[[int], None]):
    user_id = 5
    user_role = Role.TEAM_OWNER
    team_id = teams[1].id
    await _send_invitation(client, team_id, user_id, user_role)
    resp = await _get_team_users(client, team_id)
    assert_response(resp, [TeamUser(id=OTHER_USER_ID, name="Jane Doe", username="test2", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                           TeamUser(id=USER_ID, name="John Doe", username="test", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                           TeamUser(id=5, name="John Doe 5", username="test5", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.PENDING, verified=True)])


async def test_pending_invitation_agent_list(client: AsyncClient, teams: list[Team], override_user: Callable[[int], None]):
    user_id = 5
    user_role = Role.TEAM_OWNER
    team_id = teams[1].id
    await send_invitation_and_switch_user(client, override_user, team_id, user_id, user_role)
    resp = await client.get(AGENTS_PATH, params={"team_id": team_id})
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_user_role_in_team(client: AsyncClient):
    resp = await client.put(TEAM_USER_PATH.format(team_id=2, user_id=OTHER_USER_ID), json={"role": Role.TEAM_MEMBER})
    resp.raise_for_status()
    user_resp = await _get_team_users(client, 2)
    assert_response(user_resp, [TeamUser(id=OTHER_USER_ID, name="Jane Doe", username="test2", role=Role.TEAM_MEMBER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                                TeamUser(id=USER_ID, name="John Doe", username="test", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True)])


async def test_update_user_role_in_team_not_found(client: AsyncClient):
    resp = await client.put(TEAM_USER_PATH.format(team_id=5, user_id=OTHER_USER_ID), json={"role": Role.TEAM_MEMBER})
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_user_role_in_team_user_not_found(client: AsyncClient):
    resp = await client.put(TEAM_USER_PATH.format(team_id=2, user_id=10), json={"role": Role.TEAM_MEMBER})
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_user_role_in_team_not_authorized(client: AsyncClient):
    resp = await client.put(TEAM_USER_PATH.format(team_id=3, user_id=OTHER_USER_ID), json={"role": Role.TEAM_MEMBER})
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_user_role_in_global_team(client: AsyncClient):
    resp = await client.put(TEAM_USER_PATH.format(team_id=GLOBAL_TEAM_ID, user_id=5), json={"role": Role.TEAM_OWNER})
    resp.raise_for_status()
    user_resp = await _get_team_users(client, GLOBAL_TEAM_ID)
    assert_response(user_resp, [TeamUser(id=OTHER_USER_ID, name="Jane Doe", username="test2", role=Role.TEAM_MEMBER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                                TeamUser(id=USER_ID, name="John Doe", username="test", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True),
                                TeamUser(id=5, name="John Doe 5", username="test5", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True)])


async def test_assign_budget_owner_not_authorized(client: AsyncClient):
    resp = await _send_invitation(client, GLOBAL_TEAM_ID, 5, Role.BUDGET_OWNER)
    assert resp.status_code == status.HTTP_403_FORBIDDEN


async def test_assign_budget_owner_to_non_global_team(client: AsyncClient, override_user_role: Callable[[Role], None]):
    override_user_role(Role.BUDGET_OWNER)
    resp = await _send_invitation(client, 2, 5, Role.BUDGET_OWNER)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST


async def test_assign_budget_owner_to_global_team(client: AsyncClient, override_user_role: Callable[[Role], None]):
    override_user_role(Role.BUDGET_OWNER)
    resp = await _send_invitation(client, GLOBAL_TEAM_ID, 5, Role.BUDGET_OWNER)
    assert resp.status_code == status.HTTP_200_OK


async def test_update_budget_owner_role_not_authorized(client: AsyncClient, session: AsyncSession):
    session.add(TeamRole(user_id=5, team_id=GLOBAL_TEAM_ID, role=Role.BUDGET_OWNER, status=TeamRoleStatus.ACCEPTED))
    await session.commit()
    resp = await client.put(TEAM_USER_PATH.format(team_id=GLOBAL_TEAM_ID, user_id=5), json={"role": Role.TEAM_OWNER})
    assert resp.status_code == status.HTTP_403_FORBIDDEN


async def test_delete_user_from_team(client: AsyncClient):
    resp = await client.delete(TEAM_USER_PATH.format(team_id=2, user_id=OTHER_USER_ID))
    resp.raise_for_status()
    user_resp = await _get_team_users(client, 2)
    assert_response(user_resp, [TeamUser(id=USER_ID, name="John Doe", username="test", role=Role.TEAM_OWNER, role_status=TeamRoleStatus.ACCEPTED, verified=True)])


async def test_delete_user_from_team_not_found(client: AsyncClient):
    resp = await client.delete(TEAM_USER_PATH.format(team_id=5, user_id=OTHER_USER_ID))
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_delete_user_from_team_user_not_found(client: AsyncClient):
    resp = await client.delete(TEAM_USER_PATH.format(team_id=2, user_id=10))
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_delete_user_from_team_not_authorized(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(OTHER_USER_ID)
    resp = await client.delete(TEAM_USER_PATH.format(team_id=3, user_id=OTHER_USER_ID))
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_delete_user_from_global_team(client: AsyncClient):
    resp = await client.delete(TEAM_USER_PATH.format(team_id=GLOBAL_TEAM_ID, user_id=USER_ID))
    resp.raise_for_status()
    user_resp = await _get_team_users(client, GLOBAL_TEAM_ID)
    assert user_resp.status_code == status.HTTP_404_NOT_FOUND


async def test_find_teams(client: AsyncClient):
    resp = await client.get(TEAMS_PATH)
    resp.raise_for_status()
    assert_response(resp, [Team(id=2, name="Another Team"),
                           Team(id=4, name="Fourth Team"),
                           Team(id=1, name="Test Team"),
                           Team(id=3, name="Third Team")])


async def test_find_teams_unauthorized(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(5)
    resp = await client.get(TEAMS_PATH)
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_create_team(client: AsyncClient, teams: list[Team]):
    new_team = await client.post(TEAMS_PATH, json=TeamCreate(name="Team").model_dump())
    new_team.raise_for_status()
    resp = await client.get(TEAMS_PATH)
    assert_response(resp, [Team(id=2, name="Another Team"),
                           Team(id=4, name="Fourth Team"),
                           Team(**new_team.json()),
                           Team(id=1, name="Test Team"),
                           Team(id=3, name="Third Team")])


async def test_create_team_unauthorized(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(5)
    resp = await client.post(TEAMS_PATH, json=TeamCreate(name="Unauthorized Team").model_dump())
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_team(client: AsyncClient):
    resp = await client.put(TEAM_PATH.format(team_id=2), json=TeamUpdate(name="Updated Team Name").model_dump())
    resp.raise_for_status()
    resp = await client.get(TEAMS_PATH)
    assert_response(resp, [Team(id=4, name="Fourth Team"),
                           Team(id=1, name="Test Team"),
                           Team(id=3, name="Third Team"),
                           Team(id=2, name="Updated Team Name")])


async def test_update_team_unauthorized(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(5)
    resp = await client.put(TEAM_PATH.format(team_id=2), json=TeamUpdate(name="Unauthorized Update").model_dump())
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_update_team_not_found(client: AsyncClient):
    resp = await client.put(TEAM_PATH.format(team_id=999), json=TeamUpdate(name="Non-existent Team").model_dump())
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_remove_team(client: AsyncClient):
    resp = await client.delete(TEAM_PATH.format(team_id=3))
    resp.raise_for_status()
    resp = await client.get(TEAMS_PATH)
    assert_response(resp, [Team(id=2, name="Another Team"), Team(id=4, name="Fourth Team"), Team(id=1, name="Test Team")])


async def test_remove_team_unauthorized(client: AsyncClient, override_user: Callable[[int], None]):
    override_user(5)
    resp = await client.delete(TEAM_PATH.format(team_id=3))
    assert resp.status_code == status.HTTP_404_NOT_FOUND


async def test_remove_team_not_found(client: AsyncClient):
    resp = await client.delete(TEAM_PATH.format(team_id=999))
    assert resp.status_code == status.HTTP_404_NOT_FOUND
