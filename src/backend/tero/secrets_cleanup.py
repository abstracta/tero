import asyncio
import logging

from .core.env import env
from .core.repos import get_db
from .tools.auth import ToolAuthRepository, ToolOAuthClientInfoRepository
from .tools.mcp import MCP_TOOL_ID

logging.basicConfig(level=logging.INFO)

async def main():
    async for db in get_db():
        await ToolAuthRepository(db).cleanup()
        await ToolOAuthClientInfoRepository(db).cleanup(f"{MCP_TOOL_ID}-*", env.mcp_tool_oauth_client_registration_ttl_minutes)

if __name__ == "__main__":
    asyncio.run(main())
