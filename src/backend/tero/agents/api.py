import logging
from enum import Enum
from typing import Annotated, Optional, List, cast
from zipfile import BadZipFile

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core import repos as repos_module
from ..core.api import BASE_PATH
from ..core.auth import get_current_user
from ..core.domain import CamelCaseModel
from ..core.env import env
from ..core.repos import get_db
from ..files.api import build_file_download_response
from ..files.core import QuotaExceededError, add_encoding_to_content_type
from ..files.domain import File, FileStatus, FileUpdate, FileMetadata, FileMetadataWithContent
from ..files.repos import FileRepository
from ..teams.domain import GLOBAL_TEAM_ID, Role
from ..tools.auth import ToolAuthRequestException, build_tool_auth_request_http_exception
from ..tools.core import AgentTool
from ..tools.repos import ToolRepository
from ..users.domain import User
from ..users.repos import UserRepository
from . import field_generation, distribution
from .distribution import AgentImportResult, UnsupportedFileStructureError, MissingRequiredConfigurationError
from .domain import AgentListItem, Agent, AgentUpdate, AgentToolConfig, AutomaticAgentField, PublicAgent
from .evaluators.repos import EvaluatorRepository
from .prompts.repos import AgentPromptRepository
from .repos import AgentRepository, AgentToolConfigRepository, AgentToolConfigFileRepository
from .test_cases.clone import clone_test_case
from .test_cases.repos import TestCaseRepository
from .tool_file import upload_tool_file


logger = logging.getLogger(__name__)
router = APIRouter()
AGENTS_PATH = f"{BASE_PATH}/agents"
_DEFAULT_FILE_NAME = "uploaded-file"
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant.
Use provided tools and information provided in context to answer user questions.
Avoid generating responses that are not based on tools or previous context.
Answer in the same language as the user.
Use markdown to format your responses. You can include code blocks, tables, plantuml diagrams code blocks, echarts configuration code blocks and any standard markdown format"""


class AgentSort(Enum):
    LAST_UPDATE = "LAST_UPDATE"
    ACTIVE_USERS = "ACTIVE_USERS"


@router.get(AGENTS_PATH)
async def find_agents(
        user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)],
        own: bool = False, pinned: bool = False, text: Optional[str] = None, sort: AgentSort = AgentSort.LAST_UPDATE,
        limit: int = 8, offset: int = 0, team_id: Optional[int] = None) -> List[AgentListItem]:
    agents_repo = AgentRepository(db)
    if(team_id and not user.is_member_of(team_id)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if own:
        return await agents_repo.find_own_agents(user, limit, offset)
    if pinned:
        return await agents_repo.find_user_agents(text, user)
    if text:
        return await agents_repo.find_by_text(text, user, limit, offset)
    if sort == AgentSort.ACTIVE_USERS:
        ret = await agents_repo.find_top_used(user, limit, offset, team_id)
        return ret
    else:
        return await agents_repo.find_newest(user, limit, offset, team_id)


@router.get(f"{AGENTS_PATH}/default")
async def find_default_agent(user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> PublicAgent:
    agent = await AgentRepository(db).find_default_agent()
    if not agent or not agent.is_visible_by(user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Default agent not found")
    return PublicAgent.from_agent(agent, agent.is_editable_by(user))


AGENT_PATH = f"{AGENTS_PATH}/{{agent_id}}"
AGENT_PIN_PATH = f"{AGENT_PATH}/pin"


@router.post(AGENT_PIN_PATH, status_code=status.HTTP_201_CREATED)
async def add_user_agent(agent_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]):
    await find_agent_by_id(agent_id, user, db)
    await AgentRepository(db).add_user_agent(agent_id, user.id)


async def find_agent_by_id(agent_id: int, user: User, db: AsyncSession) -> Agent:
    ret = await AgentRepository(db).find_by_id(agent_id)
    if not ret or not ret.is_visible_by(user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    return ret


def _require_editor_for_protected(agent: Agent, user: User, detail: str) -> None:
    if agent.is_protected and not agent.is_editable_by(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Editors access required to {detail}")


@router.delete(AGENT_PIN_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def remove_user_agent(agent_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]):
    repo = AgentRepository(db)
    agent = await repo.find_user_agent(agent_id, user.id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    await repo.remove_user_agent(agent_id, user.id)


@router.post(AGENTS_PATH, status_code=status.HTTP_201_CREATED)
async def new_agent(user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> PublicAgent:
    ret = await AgentRepository(db).add(Agent(
        user_id=user.id,
        model_id=cast(str, env.agent_default_model),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ))
    return PublicAgent.from_agent(ret, True)


@router.put(AGENT_PATH)
async def update_agent(agent_id: int, updated: AgentUpdate, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> PublicAgent:
    agent = await find_editable_agent(agent_id, user, db)

    if updated.is_protected is not None and updated.is_protected != agent.is_protected and not user.is_global_owner():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Global owner access required to manage protected configuration"
        )

    if updated.team_id == GLOBAL_TEAM_ID and env.disable_publish_global and not any(tr.role in [Role.TEAM_OWNER, Role.TEAM_EDITOR] and tr.team_id == GLOBAL_TEAM_ID for tr in user.team_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Global team members cannot publish to global team"
        )

    agent.update_with(updated)
    ret = await AgentRepository(db).update(agent)
    if updated.publish_prompts:
        await AgentPromptRepository(db).publish_all_private_prompts(agent_id)
    return PublicAgent.from_agent(ret, ret.user_id == user.id)


async def find_editable_agent(agent_id: int, user: User, db: AsyncSession) -> Agent:
    ret = await find_agent_by_id(agent_id, user, db)
    if not ret.is_editable_by(user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    return ret


@router.post(f"{AGENT_PATH}/fields/{{field}}")
async def generate_agent_field(agent_id: int, field: AutomaticAgentField, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> str:
    agent = await find_editable_agent(agent_id, user, db)
    return await field_generation.generate_agent_field(agent, field, user.id, db)


@router.get(AGENT_PATH)
async def find_agent(agent_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> PublicAgent:
    ret = await find_agent_by_id(agent_id, user, db)
    return PublicAgent.from_agent(ret, ret.is_editable_by(user))

AGENT_TOOLS_PATH = f"{AGENT_PATH}/tools"
AGENT_TOOL_PATH = f"{AGENT_TOOLS_PATH}/{{tool_id}}"


class PublicAgentTool(CamelCaseModel):
    tool_id: str
    config: dict


@router.post(AGENT_TOOLS_PATH)
async def configure_agent_tool(agent_id: int, tool_config: PublicAgentTool,
        user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> PublicAgentTool:
    agent = await find_editable_agent(agent_id, user, db)
    tool = _find_agent_tool(tool_config.tool_id)
    if not tool:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tool id")
    tool.configure(agent, user.id, tool_config.config, db)
    try:
        prev_config = await AgentToolConfigRepository(db).find_by_ids(agent_id, tool.id, include_drafts=True)
        await tool.setup(prev_config)
        await _save_tool_config(agent_id, tool, tool_config.tool_id, draft=False, db=db)
        return tool_config
    except ToolAuthRequestException as e:
        # To be able to complete authentication we need to save the tool config
        await _save_tool_config(agent_id, tool, tool_config.tool_id, draft=True, db=db)
        raise build_tool_auth_request_http_exception(e.request)
    except Exception:
        logger.error(f"Invalid tool configuration {agent_id} {tool_config.tool_id} {tool_config.config}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tool configuration")


def _find_agent_tool(tool_id: str) -> Optional[AgentTool]:
    return ToolRepository().find_by_id(tool_id)


async def _save_tool_config(agent_id: int, tool: AgentTool, prev_tool_id: str, draft: bool, db: AsyncSession):
    repo = AgentToolConfigRepository(db)
    await repo.delete_drafts(agent_id)
    # this happens when tool id has a wildcard and the definitive tool id is defined when tool.configure is called
    # we need to delete the old tool config since the tool id may have changed when tool.configure was invoked
    if tool.id != prev_tool_id:
        await repo.delete(agent_id, prev_tool_id)
    await repo.add(AgentToolConfig(agent_id=agent_id, tool_id=tool.id, config=tool.config, draft=draft))


@router.get(AGENT_TOOLS_PATH)
async def find_agent_tools_configs(agent_id: int, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> List[AgentToolConfig]:
    agent = await find_agent_by_id(agent_id, user, db)
    _require_editor_for_protected(agent, user, "view this agent's tool configs")
    return await AgentToolConfigRepository(db).find_by_agent_id(agent_id)


@router.delete(AGENT_TOOL_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_tool(agent_id: int, tool_id: str, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]):
    tool = await _find_editable_configured_agent_tool(agent_id, tool_id, user, db)
    await tool.teardown()
    await AgentToolConfigFileRepository(db).delete_by_agent_id_and_tool_id(agent_id, tool_id)
    await AgentToolConfigRepository(db).delete(agent_id, tool_id)


async def _find_editable_configured_agent_tool(agent_id: int, tool_id: str, user: User,
        db: AsyncSession) -> AgentTool:
    ret = await _find_configured_agent_tool(agent_id, tool_id, user, db)
    if not ret.agent.is_editable_by(user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    return ret


async def _find_configured_agent_tool(agent_id: int, tool_id: str, user: User, db: AsyncSession, include_drafts: bool = False) -> AgentTool:
    config = await AgentToolConfigRepository(db).find_by_ids(agent_id, tool_id, include_drafts)
    if not config or not config.agent.is_visible_by(user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    ret = _find_agent_tool(tool_id)
    if not ret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool not found")
    ret.configure(config.agent, user.id, config.config, db)
    return ret


AGENT_TOOL_FILES_PATH = f"{AGENT_TOOL_PATH}/files"

@router.post(AGENT_TOOL_FILES_PATH, status_code=status.HTTP_202_ACCEPTED)
async def upload_agent_tool_file(agent_id: int, tool_id: str, file: UploadFile,
        user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)],
        background_tasks: BackgroundTasks) -> FileMetadata:
        tool = await _find_editable_configured_agent_tool(agent_id, tool_id, user, db)
        f = File(
            user_id=user.id,
            name=file.filename or _DEFAULT_FILE_NAME,
            content_type=file.content_type or "",
            content=await file.read(),
            status=FileStatus.PENDING
        )
        return await upload_tool_file(f, tool, agent_id, user, db, background_tasks)


@router.get(AGENT_TOOL_FILES_PATH)
async def find_agent_tool_files(agent_id: int, tool_id: str, user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> List[FileMetadata]:
    await _find_configured_agent_tool(agent_id, tool_id, user, db)
    return [FileMetadata.from_file(f) for f in
            await AgentToolConfigFileRepository(db).find_by_agent_id_and_tool_id(agent_id, tool_id)]

AGENT_TOOL_FILE_PATH = f"{AGENT_TOOL_FILES_PATH}/{{file_id}}"
AGENT_TOOL_FILE_CONTENT_PATH = f"{AGENT_TOOL_FILE_PATH}/content"

@router.get(AGENT_TOOL_FILE_CONTENT_PATH)
async def download_agent_tool_file(agent_id: int, tool_id: str, file_id: int,
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> StreamingResponse:
    await _find_configured_agent_tool(agent_id, tool_id, user, db)
    ret = await AgentToolConfigFileRepository(db).find_with_content_by_ids(agent_id, tool_id, file_id)
    return build_file_download_response(ret)


@router.get(AGENT_TOOL_FILE_PATH)
async def find_agent_doc_tool_file(agent_id: int, tool_id: str, file_id: int,
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)]) -> FileMetadataWithContent:
    await _find_configured_agent_tool(agent_id, tool_id, user, db)
    file_obj = await AgentToolConfigFileRepository(db).find_by_ids(agent_id, tool_id, file_id)
    if not file_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileMetadataWithContent.from_file(file_obj)


@router.put(AGENT_TOOL_FILE_PATH, status_code=status.HTTP_202_ACCEPTED)
async def update_agent_tool_file(agent_id: int, tool_id: str, file_id: int, file: UploadFile,
        user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)],
        background_tasks: BackgroundTasks) -> FileMetadata:
    tool = await _find_editable_configured_agent_tool(agent_id, tool_id, user, db)
    f = await _find_agent_tool_file(agent_id, tool_id, file_id, db)
    file_content = await file.read()
    update = FileUpdate(
        content=file_content if len(file_content) > 0 else None,
        content_type=add_encoding_to_content_type(file.content_type, file_content),
        name=file.filename or _DEFAULT_FILE_NAME,
        user_id=user.id,
        status=FileStatus.PENDING
    )
    f.update_with(update)
    await FileRepository(db).update(f)
    # Pass IDs instead of objects to avoid session conflicts
    # The background task will create its own session and re-fetch the entities
    background_tasks.add_task(_update_tool_file, f.id, tool.id, agent_id, user.id, tool.config)
    return FileMetadata.from_file(f)


async def _find_agent_tool_file(agent_id: int, tool_id: str, file_id: int, db: AsyncSession) -> File:
    ret = await AgentToolConfigFileRepository(db).find_by_ids(agent_id, tool_id, file_id)
    if not ret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return ret


async def _update_tool_file(file_id: int, tool_id: str, agent_id: int, user_id: int, tool_config: dict):
    async with AsyncSession(repos_module.engine, expire_on_commit=False) as db:
        file = cast(File, await FileRepository(db).find_by_id(file_id))
        user = cast(User, await UserRepository(db).find_by_id(user_id))
        agent = cast(Agent, await AgentRepository(db).find_by_id(agent_id))
        tool = cast(AgentTool, ToolRepository().find_by_id(tool_id))
        tool.configure(agent, user_id, tool_config, db)

        try:
            await tool.update_file(file, user)
            file.status = FileStatus.PROCESSED
        except QuotaExceededError:
            file.status = FileStatus.QUOTA_EXCEEDED
            logger.warning(f"Quota exceeded for user {user_id} when updating tool file {file_id} {file.name}")
        except Exception as e:
            file.status = FileStatus.ERROR
            logger.error(f"Error updating tool file {file_id} {file.name} {e}", exc_info=True)
        finally:
            await FileRepository(db).update(file)


@router.delete(AGENT_TOOL_FILE_PATH, status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_tool_file(agent_id: int, tool_id: str, file_id: int,
        user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]):
    tool = await _find_editable_configured_agent_tool(agent_id, tool_id, user, db)
    f = await _find_agent_tool_file(agent_id, tool_id, file_id, db)
    f.user_id = user.id
    await tool.remove_file(f)
    await AgentToolConfigFileRepository(db).delete(agent_id, tool_id, file_id)
    await FileRepository(db).delete(f)


@router.post(f"{AGENT_PATH}/clone", status_code=status.HTTP_201_CREATED)
async def clone_agent(agent_id: int, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> PublicAgent:
    agent = await find_agent_by_id(agent_id, user, db)
    _require_editor_for_protected(agent, user, "clone this protected agent")
    cloned_agent = await AgentRepository(db).add(agent.clone(user_id=user.id))

    await _clone_agent_prompts(agent_id, cloned_agent.id, user.id, db)
    await _clone_agent_tools(agent_id, cloned_agent.id, user.id, db)
    await _clone_agent_test_cases(agent_id, cloned_agent.id, user.id, db)
    await _clone_agent_evaluator(agent, cloned_agent, db)

    return PublicAgent.from_agent(cloned_agent, True)


async def _clone_agent_prompts(agent_id: int, cloned_agent_id: int, user_id: int, db: AsyncSession) -> None:
    prompt_repo = AgentPromptRepository(db)
    prompts = await prompt_repo.find_user_agent_prompts(user_id, agent_id)

    if prompts:
        cloned_prompts = [prompt.clone(agent_id=cloned_agent_id, user_id=user_id) for prompt in prompts]
        await prompt_repo.add_many(cloned_prompts)


async def _clone_agent_tools(agent_id: int, cloned_agent_id: int, user_id: int, db: AsyncSession) -> None:
    tool_repo = AgentToolConfigRepository(db)
    tool_configs = await tool_repo.find_by_agent_id(agent_id)
    if not tool_configs:
        return

    cloned_tool_configs = [config.clone(agent_id=cloned_agent_id) for config in tool_configs]
    await tool_repo.add_many(cloned_tool_configs)

    for tool_config in cloned_tool_configs:
        tool = ToolRepository().find_by_id(tool_config.tool_id)
        if tool is None:
            raise ValueError(f"Tool '{tool_config.tool_id}' configured in agent {agent_id} not found")

        tool.configure(tool_config.agent, user_id, tool_config.config, db)
        await tool.clone(agent_id, cloned_agent_id, tool_config.tool_id, user_id, db)


async def _clone_agent_test_cases(agent_id: int, cloned_agent_id: int, user_id: int, db: AsyncSession) -> None:
    test_case_repo = TestCaseRepository(db)
    test_cases = await test_case_repo.find_by_agent(agent_id)

    if not test_cases:
        return

    for test_case in test_cases:
        await clone_test_case(test_case, cloned_agent_id, user_id, db)


async def _clone_agent_evaluator(agent: Agent, cloned_agent: Agent, db: AsyncSession) -> None:
    evaluator_repo = EvaluatorRepository(db)
    evaluator = await evaluator_repo.find_by_id(agent.evaluator_id) if agent.evaluator_id else None
    if not evaluator:
        return

    cloned_evaluator = await evaluator_repo.save(evaluator.clone())
    cloned_agent.evaluator_id = cloned_evaluator.id
    await AgentRepository(db).update(cloned_agent)


@router.get(f"{AGENT_PATH}/dist")
async def download_agent_distribution(agent_id: int, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)]) -> StreamingResponse:
    agent = await find_agent_by_id(agent_id, user, db)
    _require_editor_for_protected(agent, user, "export this protected agent")
    return build_file_download_response(await distribution.generate_agent_zip(agent, user.id,db))


@router.put(f"{AGENT_PATH}/dist")
async def update_agent_from_distribution(agent_id: int, file: UploadFile, user: Annotated[User, Depends(get_current_user)], db: Annotated[AsyncSession, Depends(get_db)],
        background_tasks: BackgroundTasks) -> AgentImportResult:
    agent = await find_editable_agent(agent_id, user, db)
    try:
        return await distribution.update_agent_from_zip(agent, await file.read(), user, db, background_tasks)
    except BadZipFile:
        logger.error(f"Error updating agent {agent_id} from distribution", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalidFileFormat"
        )
    except UnsupportedFileStructureError:
        logger.error(f"Error updating agent {agent_id} from distribution", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupportedFileStructure"
        )
    except MissingRequiredConfigurationError:
        logger.error(f"Error updating agent {agent_id} from distribution", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="missingRequiredConfiguration"
        )
