from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import List, Optional

from langchain.tools import BaseTool
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import Agent, AgentToolConfig, AgentType
from ...agents.repos import AgentRepository
from ...teams.domain import GLOBAL_TEAM_ID
from ...users.repos import UserRepository
from ..core import AgentTool, load_schema

SUBAGENT_TOOL_ID = "subagent"


class SubagentTool(AgentTool):
    id: str = f"{SUBAGENT_TOOL_ID}-*"
    name: str = "Subagent"
    description: str = "Delegate work to another Tero agent"
    config_schema: dict = load_schema(__file__)
    _subagent: Optional[Agent] = None

    def configure(self, agent: Agent, user_id: int, config: dict, db: AsyncSession, thread_id: Optional[int] = None):
        super().configure(agent, user_id, config, db, thread_id)
        if "agentId" in config:
            self.id = self.tool_id_for(self._configured_subagent_id())

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        if self.agent.agent_type != AgentType.DEEP_AGENT:
            raise ValueError("Subagents can only be configured on advanced agents")

        if self._configured_subagent_id() == self.agent.id:
            raise ValueError("An agent cannot use itself as a subagent")

        self._subagent = await self._find_usable_subagent()

    async def teardown(self):
        pass

    @asynccontextmanager
    async def load(self) -> AsyncIterator["SubagentTool"]:
        self._subagent = await self._find_usable_subagent()
        yield self

    async def _find_usable_subagent(self) -> Agent:
        subagent = await AgentRepository(self.db).find_by_id(self._configured_subagent_id())
        user = await UserRepository(self.db).find_by_id(self.user_id)
        if not subagent or not user or not subagent.is_visible_by(user):
            raise ValueError("Subagent not found")
        if self.agent.team_id is not None and subagent.team_id not in (self.agent.team_id, GLOBAL_TEAM_ID):
            raise ValueError(
                f"Subagent '{subagent.name}' must be shared with the same team before it can be added"
            )
        return subagent

    async def clone(
        self,
        agent_id: int,
        cloned_agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> None:
        pass

    async def build_langchain_tools(self) -> List[BaseTool]:
        # Subagents are exposed through deepagents' native task tool, not as langchain tools.
        return []

    @property
    def subagent(self) -> Agent:
        if self._subagent is None:
            raise ValueError("Subagent has not been loaded")
        return self._subagent

    @classmethod
    def tool_id_for(cls, agent_id: int) -> str:
        return f"{SUBAGENT_TOOL_ID}-{agent_id}"

    @classmethod
    def parse_subagent_id(cls, tool_id: str) -> Optional[int]:
        prefix = f"{SUBAGENT_TOOL_ID}-"
        if not tool_id.startswith(prefix):
            return None
        suffix = tool_id[len(prefix):]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _configured_subagent_id(self) -> int:
        return int(self.config["agentId"])
