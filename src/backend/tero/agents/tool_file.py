import logging
from typing import cast

from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi.background import BackgroundTasks

from ..core import repos as repos_module
from ..files.core import add_encoding_to_content_type, QuotaExceededError
from ..files.domain import File, FileStatus, FileMetadata
from ..files.repos import FileRepository
from ..threads.domain import ModelRateLimitError
from ..tools.core import AgentTool
from ..tools.repos import ToolRepository
from ..users.domain import User
from ..users.repos import UserRepository
from .domain import AgentToolConfigFile, Agent
from .repos import AgentToolConfigFileRepository, AgentRepository


logger = logging.getLogger(__name__)


async def upload_tool_file(file: File, tool: AgentTool, agent_id: int, user: User, db: AsyncSession, background_tasks: BackgroundTasks) -> FileMetadata:
    file.content_type = add_encoding_to_content_type(file.content_type, file.content)
    file = await FileRepository(db).add(file)
    await AgentToolConfigFileRepository(db).add(AgentToolConfigFile(agent_id=agent_id, tool_id=tool.id, file_id=file.id))
    # Pass file_id instead of file object to avoid session conflicts
    # The background task will create its own session and re-fetch the file
    background_tasks.add_task(_add_tool_file, file.id, user.id, tool.id, agent_id, tool.config)
    return FileMetadata.from_file(file)


async def _add_tool_file(file_id: int, user_id: int, tool_id: str, agent_id: int, tool_config: dict):
    async with AsyncSession(repos_module.engine, expire_on_commit=False) as db:
        f = cast(File, await FileRepository(db).find_by_id(file_id))
        user = cast(User, await UserRepository(db).find_by_id(user_id))
        agent = cast(Agent, await AgentRepository(db).find_by_id(agent_id))
        tool = cast(AgentTool, ToolRepository().find_by_id(tool_id))
        tool.configure(agent, user_id, tool_config, db)

        try:
            await tool.add_file(f, user)
            f.status = FileStatus.PROCESSED
        except QuotaExceededError:
            f.status = FileStatus.QUOTA_EXCEEDED
            logger.error(f"Quota exceeded for user {user_id} when adding tool file {file_id} {f.name}")
        except ModelRateLimitError:
            f.status = FileStatus.RATE_LIMITED
            logger.warning(f"Rate limited by AI provider when adding tool file {file_id} {f.name}")
        except Exception as e:
            f.status = FileStatus.ERROR
            logger.error(f"Error adding tool file {file_id} {f.name} {e}", exc_info=True)
        finally:
            await FileRepository(db).update(f)
