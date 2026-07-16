import pytest
# Register assertion rewriting for helper modules BEFORE importing them
# This enables pytest's assertion introspection (showing diffs) for assertions in these modules
pytest.register_assert_rewrite('tests.common')

import asyncio
import contextlib
import logging
import os
from typing import Generator, AsyncGenerator, List
from unittest.mock import AsyncMock, patch

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

from fastapi import Depends
import freezegun
from httpx import AsyncClient, ASGITransport
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel, col
import sqlparse
from testcontainers.postgres import PostgresContainer

# avoid any authentication requirements
os.environ['OPENID_URL'] = ''

from tero.agents.domain import AgentListItem, Agent
from tero.agents.test_cases.domain import TestSuiteRun, TestCaseResult
from tero.api import app
from tero.core import repos as repos_module, auth
from tero.core.env import env
from tero.core.repos import get_db
from tero.teams.domain import Role, Team, TeamRole, TeamRoleStatus
from tero.threads.domain import Thread, ThreadMessage
from tero.threads.agents_store import get_store
from tero.users.domain import User, UserListItem

from .common import find_last_id, USER_ID, OTHER_USER_ID, AGENT_ID, NON_EDITABLE_AGENT_ID, GLOBAL_TEAM_ID, parse_date, find_asset_text

# avoid transformers module giving erros when using freeze_time due to torch not being installed (torch gives problems when installed on x86_64 macos)
freezegun.configure(extend_ignore_list=["transformers"])

# Fix for pydantic datetime schema generation with freezegun
# Based on: https://github.com/pydantic/pydantic/discussions/9343
from pydantic._internal._generate_schema import GenerateSchema
from pydantic_core import core_schema

initial_match_type = GenerateSchema.match_type

def patched_match_type(self, obj):
    if getattr(obj, "__name__", None) == "datetime":
        return core_schema.datetime_schema()
    return initial_match_type(self, obj)

GenerateSchema.match_type = patched_match_type

logger = logging.getLogger(__name__)


class SafeEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()
        _orig_call_soon = loop.call_soon

        def safe_call_soon(callback, *args, **kwargs):
            # drop any callbacks scheduled after the loop is closed
            if loop.is_closed():
                return
            return _orig_call_soon(callback, *args, **kwargs)

        loop.call_soon = safe_call_soon # type: ignore
        return loop


@pytest.fixture(scope="session")
def event_loop_policy():
    # Avoid "Event loop is closed" error log generated aclose in httpx
    return SafeEventLoopPolicy()


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    with PostgresContainer("pgvector/pgvector:pg17", driver="psycopg") as postgres:
        yield postgres


@pytest_asyncio.fixture(name="session")
async def session_fixture(postgres_container: PostgresContainer) -> AsyncGenerator[AsyncSession, None]:
    test_engine = create_async_engine(postgres_container.get_connection_url())
    
    original_engine = repos_module.engine
    repos_module.engine = test_engine
    
    try:
        async with test_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
            await conn.run_sync(SQLModel.metadata.create_all)
            await _init_db_data(conn)
        async with AsyncSession(test_engine, expire_on_commit=False) as ret:
            yield ret
    finally:
        repos_module.engine = original_engine


async def _init_db_data(conn: AsyncConnection) -> None:
    sql_script = await find_asset_text('init_db.sql')
    statements = sqlparse.split(sql_script)
    for statement in statements:
        if statement.strip():
            await conn.exec_driver_sql(statement)


@pytest_asyncio.fixture(name="client")
async def client_fixture(session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def get_db_override() -> AsyncGenerator[AsyncSession, None]:
        yield session

    async def get_current_user_override(db: AsyncSession = Depends(get_db)):
        from tero.users.repos import UserRepository
        user = await UserRepository(db).find_by_id(USER_ID)
        if user is None:
            raise ValueError(f"User with ID {USER_ID} not found")
        return user

    test_store = InMemoryStore()

    def get_store_override() -> BaseStore:
        return test_store

    app.dependency_overrides[get_db] = get_db_override
    app.dependency_overrides[auth.get_current_user] = get_current_user_override
    app.dependency_overrides[get_store] = get_store_override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture(name="users")
def users_fixture() -> List[UserListItem]:
    return [UserListItem(id=USER_ID, username="test", name="John Doe"),
            UserListItem(id=OTHER_USER_ID, username="test2", name="Jane Doe"),
            UserListItem(id=3, username="test3", name="John Doe 3"),
            UserListItem(id=5, username="test5", name="John Doe 5")]


@pytest.fixture(name="teams")
def teams_fixture() -> List[Team]:
    return [Team(id=GLOBAL_TEAM_ID, name="Test Team"),
            Team(id=2, name="Another Team"),
            Team(id=4, name="Fourth Team")]


@pytest.fixture(name="agents")
def agents_fixture(users: dict[int, UserListItem], teams: List[Team]) -> List[AgentListItem]:
    return [
        AgentListItem(id=AGENT_ID, name="Agent 1", description="This is the first agent",
                      last_update=parse_date("2025-02-21T12:00"), team=None,
                      user=users[0], active_users=1, can_edit=True, is_protected=False),
        AgentListItem(id=2, name="Agent 2", description="This is the second agent",
                      last_update=parse_date("2025-02-21T12:01"), team=teams[0],
                      user=users[0], active_users=2, can_edit=True, is_protected=False),
        AgentListItem(id=NON_EDITABLE_AGENT_ID, name="Agent 3", description="This is the third agent",
                      last_update=parse_date("2025-02-21T12:02"), team=teams[2],
                      user=users[1], active_users=1, can_edit=False, is_protected=True),
        AgentListItem(id=5, name="Agent 5", description="This is the fifth agent",
                      last_update=parse_date("2025-02-21T12:04"), team=teams[1],
                      user=users[1], active_users=1, can_edit=True, is_protected=False)]


@pytest.fixture
def override_user_role():
    def _override(role: Role):
        async def _fake_get_current_user():
            return _build_mock_user_with_role(role)
        app.dependency_overrides[auth.get_current_user] = _fake_get_current_user
    yield _override
    app.dependency_overrides.clear()


def _build_mock_user_with_role(role: Role) -> User:
    user = User(id=USER_ID, username='test', name="John Doe", monthly_usd_limit=env.monthly_usd_limit_default)
    team = Team(id=1, name="Test Team")
    team_role = TeamRole(team_id=team.id, user_id=user.id, role=role, status=TeamRoleStatus.ACCEPTED)
    team_role.team = team
    user.team_roles = [team_role]
    return user


@pytest.fixture
def override_user():
    def _override(user_id: int):
        async def _fake_get_current_user(db: AsyncSession = Depends(get_db)):
            from tero.users.repos import UserRepository
            user = await UserRepository(db).find_by_id(user_id)
            if user is None:
                raise ValueError(f"User with ID {user_id} not found")
            return user
        app.dependency_overrides[auth.get_current_user] = _fake_get_current_user
    yield _override
    app.dependency_overrides.clear()


@pytest.fixture(name="last_agent_id")
async def last_agent_id_fixture(session: AsyncSession) -> int:
    return await find_last_id(col(Agent.id), session)


@pytest.fixture(name="last_thread_id")
async def fixture_last_thread_id(session: AsyncSession) -> int:
    return await find_last_id(col(Thread.id), session)


@pytest.fixture(name="last_message_id")
async def fixture_last_message_id(session: AsyncSession) -> int:
    return await find_last_id(col(ThreadMessage.id), session)


@pytest.fixture(name="last_suite_run_id")
async def fixture_last_suite_run_id(session: AsyncSession) -> int:
    return await find_last_id(col(TestSuiteRun.id), session)


@pytest.fixture(name="last_result_id")
async def fixture_last_result_id(session: AsyncSession) -> int:
    return await find_last_id(col(TestCaseResult.id), session)


@pytest.fixture(autouse=True)
def _autouse_thread_api_stubs(request):
    with contextlib.ExitStack() as stack:
        if not request.node.get_closest_marker("no_stub_estimate_minutes_saved"):
            stack.enter_context(patch("tero.threads.api.estimate_minutes_saved", new=AsyncMock(return_value=0)))
        if not request.node.get_closest_marker("no_stub_build_thread_name"):
            stack.enter_context(patch("tero.threads.api.build_thread_name", new=AsyncMock(return_value="stub chat title")))
        yield
