import base64
import json
from io import BytesIO
import logging
import mimetypes
import re
from typing import Any, Dict, List, Optional, cast
from urllib.parse import quote
from zipfile import ZipFile, ZIP_DEFLATED

import aiofiles
from fastapi.background import BackgroundTasks
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from slugify import slugify
from sqlmodel.ext.asyncio.session import AsyncSession

from ..ai_models.domain import LlmModelType, LlmModel
from ..ai_models.repos import AiModelRepository
from ..core.assets import solve_asset_path
from ..core.env import env
from ..core.domain import CamelCaseModel
from ..files.domain import File, FileStatus
from ..threads.domain import Thread, ThreadMessage, ThreadMessageOrigin
from ..threads.repos import ThreadRepository, ThreadMessageRepository
from ..tools.core import AgentTool
from ..tools.auth import ToolAuthRequestException
from ..tools.repos import ToolRepository
from ..users.domain import User
from .domain import Agent, AgentUpdate, AgentToolConfig, AgentType, LlmTemperature, ReasoningEffort
from .evaluators.domain import Evaluator
from .evaluators.repos import EvaluatorRepository
from .prompts.domain import AgentPrompt
from .prompts.repos import AgentPromptRepository
from .repos import AgentRepository, AgentToolConfigRepository, AgentToolConfigFileRepository
from .template_parser import JinjaTemplateParser
from .test_cases.domain import TestCase
from .test_cases.repos import TestCaseRepository
from .test_cases.runner import EVALUATOR_DEFAULT_TEMPERATURE, EVALUATOR_DEFAULT_REASONING_EFFORT
from .tool_file import upload_tool_file


logger = logging.getLogger(__name__)

class AgentImportResult(CamelCaseModel):
    unavailable_tools: List[str] = []
    tools_requiring_authentication: List[str] = []
    unavailable_model: Optional[str] = None
    default_model: Optional[str] = None


class UnsupportedFileStructureError(Exception):
    pass


class MissingRequiredConfigurationError(Exception):
    pass


class ToolInfo(BaseModel):
    id: str
    config: dict
    files: List[File]


async def generate_agent_zip(agent: Agent, user_id: int, db: AsyncSession) -> File:
    agent_name = slugify(cast(str, agent.name))
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
        tools = await _find_agent_tools(agent, db)
        zip_file.writestr(f"{agent_name}/agent.md", await _generate_agent_markdown(agent, tools, user_id, db))
        if agent.icon:
            zip_file.writestr(f"{agent_name}/icon.png", agent.icon)
        tool_file_repo = AgentToolConfigFileRepository(db)
        for tool in tools:
            for file in tool.files:
                await tool_file_repo.find_with_content_by_ids(agent.id, tool.id, file.id)
                zip_file.writestr(f"{agent_name}/{tool.id}/{file.name}", file.content)
    return File(id=0, name=f"{agent_name}.zip", content=zip_buffer.getvalue(), content_type="application/zip", user_id=user_id, status=FileStatus.PENDING)


async def _find_agent_tools(agent: Agent, db: AsyncSession) -> List[ToolInfo]:
    tool_configs = await AgentToolConfigRepository(db).find_by_agent_id(agent.id)
    tool_file_repo = AgentToolConfigFileRepository(db)
    ret: List[ToolInfo] = []
    for tool_config in tool_configs:
        tool_files = await tool_file_repo.find_by_agent_id_and_tool_id(agent.id, tool_config.tool_id)
        ret.append(ToolInfo(id=tool_config.tool_id, config=tool_config.config, files=tool_files))
    return ret


async def _generate_agent_markdown(agent: Agent, tools: List[ToolInfo], user_id: int, db: AsyncSession) -> str:
    prompts = await AgentPromptRepository(db).find_user_agent_prompts(cast(int, user_id), agent.id)
    template = _build_jinja_env().get_template("agent-template.md")
    return template.render(
        name=agent.name,
        author=agent.user.name if agent.user else "",
        description=agent.description or "",
        system_prompt=agent.system_prompt,
        icon=agent.icon,
        model_name=agent.model.name,
        model_config=_format_model_config(agent.temperature, agent.reasoning_effort, agent.model.model_type),
        advanced_config=_format_advanced_config(agent),
        conversation_starters=[_format_prompt(p) for p in prompts if p.starter],
        user_prompts=[_format_prompt(p) for p in prompts if not p.starter],
        tools=[_format_tool(tool) for tool in tools],
        tests=[await _format_test(test, db) for test in await TestCaseRepository(db).find_by_agent(agent.id)],
        evaluator=await _format_agent_evaluator(agent, db)
    )


def _build_jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(solve_asset_path('.', __file__)), trim_blocks=True, lstrip_blocks=True)


def _format_model_config(temperature: LlmTemperature, reasoning_effort: ReasoningEffort, model_type: LlmModelType) -> dict:
    return {"Temperature": temperature.value.capitalize()} if model_type == LlmModelType.CHAT \
        else {"Reasoning": reasoning_effort.value.capitalize()}


def _format_advanced_config(agent: Agent) -> dict:
    config = {}
    if agent.agent_type == AgentType.DEEP_AGENT:
        config["Agent mode"] = "Advanced"
    if agent.is_protected:
        config["Protected configuration"] = "Yes"
    if agent.recursion_limit != 20:
        config["Thought process steps limit"] = str(agent.recursion_limit)
    return config


def _format_prompt(prompt: AgentPrompt) -> dict:
    return {
        "name": prompt.name,
        "content": prompt.content,
        "visibility": "Public" if prompt.shared else "Private"
    }


def _format_tool(tool: ToolInfo) -> dict:
    return {
        "name": tool.id[0].upper() + tool.id[1:].replace('-', ' ', 1).lower(),
        "files": { file.name:  f"{tool.id}/{quote(file.name, safe='')}" for file in tool.files},
        "config": { _format_tool_config_key(key): _format_tool_config_value(value) for key, value in tool.config.items() if value is not None }
    }


def _format_tool_config_key(key: str) -> str:
    return key[0].upper() + re.sub(r'([A-Z])', r' \1', key[1:]).strip().lower()


def _format_tool_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return ",".join(value) if all(isinstance(item, str) for item in value) else json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value)
    return value


async def _format_test(test_case: TestCase, db: AsyncSession) -> dict:
    messages = await ThreadMessageRepository(db).find_by_thread_id(test_case.thread_id)
    return {
        "name": test_case.thread.name,
        "messages": messages,
        "evaluator": await _format_test_evaluator(test_case, db)
    }


async def _format_agent_evaluator(agent: Agent, db: AsyncSession) -> dict:
    evaluator = await EvaluatorRepository(db).find_by_id(agent.evaluator_id) if agent.evaluator_id else None
    return await _format_evaluator(evaluator, db)


async def _format_test_evaluator(test_case: TestCase, db: AsyncSession) -> dict:
    evaluator = await EvaluatorRepository(db).find_by_id(test_case.evaluator_id) if test_case.evaluator_id else None
    return await _format_evaluator(evaluator, db)


async def _format_evaluator(evaluator: Optional[Evaluator], db: AsyncSession) -> dict:
    if not evaluator:
        return {}
    evaluator_model = cast(LlmModel, await AiModelRepository(db).find_by_id(evaluator.model_id))
    return {
        "model_name": evaluator_model.name,
        "model_config": _format_model_config(evaluator.temperature, evaluator.reasoning_effort, evaluator_model.model_type),
        "prompt": evaluator.prompt
    }


async def update_agent_from_zip(agent: Agent, zip_content: bytes, user: User, db: AsyncSession, background_tasks: BackgroundTasks) -> AgentImportResult:
    with _open_zip_file(zip_content) as zip_file:
        try:
            found_root_folder = [ name.rsplit('/', 1)[0] for name in zip_file.namelist() if name.endswith('/agent.md') ]
            # supporting zip without root folder in case users zip the folder contents and not the folder itself
            root_folder = f"{found_root_folder[0]}/" if found_root_folder else ""

            if not f"{root_folder}agent.md" in zip_file.namelist():
                raise UnsupportedFileStructureError()
            agent_md = zip_file.read(f"{root_folder}agent.md").decode('utf-8')
            async with aiofiles.open(solve_asset_path("agent-template.md", __file__)) as template_file:
                parsed = JinjaTemplateParser(_build_jinja_env()).parse(agent_md, await template_file.read())

            parsed_tools = parsed.get('tools', [])
            tools, unavailable_tools = await _find_tools(parsed_tools)
            unavailable_model, default_model = await _update_agent(agent, parsed, zip_file, root_folder, user, db)
            await _update_prompts(agent.id, parsed.get('conversation_starters', []), parsed.get('user_prompts', []), user.id, db)
            auth_required_tools = await _update_tools(agent, parsed_tools, tools, zip_file, root_folder, user, db, background_tasks)
            await _update_tests(agent.id, parsed.get('tests', []), user.id, db)
            return AgentImportResult(
                unavailable_tools=unavailable_tools,
                tools_requiring_authentication=auth_required_tools,
                unavailable_model=unavailable_model,
                default_model=default_model if unavailable_model else None
            )
        except ValueError as e:
            raise MissingRequiredConfigurationError() from e


def _open_zip_file(zip_content: bytes) -> ZipFile:
    zip_bytes = BytesIO(zip_content)
    try:
        # we test with utf-8 encoding in case the file was zipped in mac since python zip encoding auto detection does not
        # work when zip contains files with special characters (like ñ) on their names
        ret = ZipFile(zip_bytes, metadata_encoding='utf-8')
        ret.namelist()  # Test if metadata can be decoded
        return ret
    except (UnicodeDecodeError, Exception):
        zip_bytes.seek(0)
        # since some zip files might not use utf-8 encoding, we fallback to python zip encoding auto detection when utf-8 decoding fails
        return ZipFile(zip_bytes)


async def _find_tools(parsed_tools: List[Dict[str, Any]]) -> tuple[Dict[str, AgentTool], List[str]]:
    found = {}
    unavailable = []
    for parsed in parsed_tools:
        tool_id = _parse_tool_id(parsed)
        tool = ToolRepository().find_by_id(tool_id)
        if tool:
            found[tool_id] = tool
        else:
            unavailable.append(tool_id)
    return found, unavailable


def _parse_tool_id(tool: Dict[str, Any]) -> str:
    return tool['name'].lower().replace(' ', '-')


async def _update_agent(agent: Agent, parsed: Dict[str, Any], zip_file: ZipFile, root_folder: str, user: User, db: AsyncSession) -> tuple[Optional[str], Optional[str]]:
    update = AgentUpdate()
    update.name = parsed['name']
    update.description = parsed['description']
    update.system_prompt = parsed['system_prompt']

    unavailable_model: Optional[str] = None
    default_model: Optional[str] = None
    try:
        model = await _find_model_by_name(parsed['model_name'], db)
    except ValueError:
        model = cast(LlmModel, await AiModelRepository(db).find_by_id(cast(str, env.agent_default_model)))
        unavailable_model = parsed['model_name']
        default_model = model.name

    update.temperature = LlmTemperature[parsed['model_config']['Temperature'].upper()] if model.model_type == LlmModelType.CHAT else None
    update.reasoning_effort = ReasoningEffort[parsed['model_config']['Reasoning'].upper()] if model.model_type == LlmModelType.REASONING else None
    update.model_id = model.id

    icon_path = f"{root_folder}icon.png"
    if icon_path in zip_file.namelist():
        update.icon = base64.b64encode(zip_file.read(icon_path)).decode('utf-8')

    if parsed.get('evaluator'):
        agent.evaluator_id = await _create_new_evaluator(parsed['evaluator'], db)

    _apply_imported_advanced_config(update, parsed.get("advanced_config", {}), user)

    agent.update_with(update)
    agent = await AgentRepository(db).update(agent)
    return unavailable_model, default_model


def _apply_imported_advanced_config(update: AgentUpdate, advanced_config: Dict[str, Any], user: User) -> None:
    update.recursion_limit = 20
    if limit_str := advanced_config.get("Thought process steps limit"):
        limit = int(str(limit_str).strip())
        if 20 <= limit <= 100:
            update.recursion_limit = limit

    update.agent_type = AgentType.REACT_AGENT
    if advanced_config.get("Agent mode", "").strip().lower() == "advanced":
        update.agent_type = AgentType.DEEP_AGENT

    update.is_protected = False
    if user.is_global_owner() and advanced_config.get("Protected configuration", "").strip().lower() == "yes":
        update.is_protected = True


async def _create_new_evaluator(evaluator_dict: Dict[str, Any], db: AsyncSession) -> int:
    model = await _find_model_by_name(evaluator_dict['model_name'], db)
    evaluator = await EvaluatorRepository(db).save(Evaluator(
        model_id=model.id,
        temperature=LlmTemperature[evaluator_dict['model_config']['Temperature'].upper()] if model.model_type == LlmModelType.CHAT else EVALUATOR_DEFAULT_TEMPERATURE,
        reasoning_effort=ReasoningEffort[evaluator_dict['model_config']['Reasoning'].upper()] if model.model_type == LlmModelType.REASONING else EVALUATOR_DEFAULT_REASONING_EFFORT,
        prompt=evaluator_dict['prompt']
    ))
    return evaluator.id


async def _find_model_by_name(model_name: str, db: AsyncSession) -> LlmModel:
    ret = [model for model in await AiModelRepository(db).find_all() if model.name == model_name]
    if not ret:
        raise ValueError(f"Model {model_name} not found")
    return ret[0]


async def _update_prompts(agent_id: int, conversation_starters: List[Dict[str, Any]], user_prompts: List[Dict[str, Any]], user_id: int, db: AsyncSession):
    repo = AgentPromptRepository(db)
    await repo.delete_user_agent_prompts(user_id, agent_id)
    for p in conversation_starters:
        await _add_prompt(agent_id, user_id, p, db, starter=True)
    for p in user_prompts:
        await _add_prompt(agent_id, user_id, p, db, shared=p["visibility"] == "Public")


async def _add_prompt(agent_id: int, user_id: int, p: Dict[str, Any], db: AsyncSession, shared: bool=False, starter: bool=False):
        await AgentPromptRepository(db).add(AgentPrompt(
            agent_id=agent_id,
            user_id=user_id,
            name=p["name"],
            content=p["content"],
            shared=shared or starter,
            starter=starter
        ))


async def _update_tools(agent: Agent, parsed_tools: List[Dict[str, Any]], tools: Dict[str, AgentTool], zip_file: ZipFile, root_folder: str, user: User, db: AsyncSession, background_tasks: BackgroundTasks) -> List[str]:
    tools_dict = {_parse_tool_id(t): t for t in parsed_tools if _parse_tool_id(t) in tools}
    tool_config_repo = AgentToolConfigRepository(db)
    await tool_config_repo.delete_drafts(agent.id)
    existing_tools = {tc.tool_id: tc for tc in await tool_config_repo.find_by_agent_id(agent.id)}
    auth_required_tools = []

    for tc in existing_tools.values():
        if not tc.tool_id in tools_dict:
            await _remove_tool(tc, user.id, db)
        else:
            if await _update_tool(tc, tools_dict[tc.tool_id], tools[tc.tool_id], zip_file, root_folder, user, db, background_tasks):
                auth_required_tools.append(tc.tool_id)

    for tool_id, config in tools_dict.items():
        if not tool_id in existing_tools:
            if await _configure_new_tool(tool_id, config, agent, tools[tool_id], zip_file, root_folder, user, db, background_tasks):
                auth_required_tools.append(tool_id)

    return auth_required_tools


async def _remove_tool(tc: AgentToolConfig, user_id: int, db: AsyncSession):
    tool = cast(AgentTool, ToolRepository().find_by_id(tc.tool_id))
    tool.configure(tc.agent, user_id, tc.config, db)
    await tool.teardown()
    await AgentToolConfigFileRepository(db).delete_by_agent_id_and_tool_id(tc.agent_id, tc.tool_id)
    await AgentToolConfigRepository(db).delete(tc.agent_id, tc.tool_id)


async def _update_tool(tc: AgentToolConfig, new_config: Dict[str, Any], tool: AgentTool, zip_file: ZipFile, root_folder: str, user: User, db: AsyncSession, background_tasks: BackgroundTasks) -> bool:
    auth_required = await _configure_parsed_tool(tc.tool_id, new_config, tc.agent, tc, tool, user, db)
    existing_files = {f.name: f for f in await AgentToolConfigFileRepository(db).find_by_agent_id_and_tool_id(tc.agent_id, tc.tool_id)}
    new_files = _parse_new_files(tc.tool_id, zip_file, root_folder, user)

    for file_name, file in existing_files.items():
        if not file_name in new_files:
            await _remove_tool_file(file, tc, db)
        else:
            await _update_tool_file(file, new_files[file_name], tc, tool, user, db, background_tasks)

    for file_name, new_file in new_files.items():
        if not file_name in existing_files:
            await upload_tool_file(new_file, tool, tc.agent_id, user, db, background_tasks)
    return auth_required


async def _configure_parsed_tool(tool_id: str, new_config: Dict[str, Any], agent: Agent, tc: Optional[AgentToolConfig], tool: AgentTool, user: User, db: AsyncSession) -> bool:
    config = _parse_tool_config(new_config.get('config', {}), tool, tool_id)
    tool.configure(agent, user.id, config, db)
    try:
        await tool.setup(prev_config=tc)
    except ToolAuthRequestException:
        logger.error(f"Tool OAuth required by tool {tool_id} imported by user {user.id} but still not supported in imports", exc_info=True)
        await AgentToolConfigRepository(db).add(AgentToolConfig(agent_id=agent.id, tool_id=tool_id, config=config))
        return True
    except Exception:
        logger.warning(f"Tool {tool_id} setup failed during import by user {user.id}", exc_info=True)
        raise ValueError(f"Tool '{tool_id}' setup failed while importing config")
    await AgentToolConfigRepository(db).add(AgentToolConfig(agent_id=agent.id, tool_id=tool_id, config=config))
    return _has_secret_config(tool, config)


def _has_secret_config(tool: AgentTool, config: Dict[str, Any]) -> bool:
    props = tool.config_schema.get("properties", {})
    has_secret_config = any(key in props for key in ("apiKey", "bearerToken", "token", "clientSecret"))
    is_oauth_only = "authType" in props and (config.get("authType") or "oauth") == "oauth"
    return has_secret_config and not is_oauth_only


def _parse_tool_config(config: Dict[str, Any], tool: AgentTool, tool_id: str) -> Any:
    tool_schema = tool.get_schema_without_files(tool.config_schema)
    ret = {}
    for key, value in config.items():
        parsed_key = _parse_config_key(key)
        if parsed_key in tool_schema["properties"]:
            ret[parsed_key] = _parse_config_value(value, tool_schema["properties"][parsed_key], parsed_key, tool_id)
        else:
            logger.warning(f"Tool '{tool_id}' config key '{parsed_key}' not found in tool schema, ignoring it.")
    return ret


def _parse_config_key(key: str) -> str:
    ret = ''.join(word.capitalize() for word in key.split(' '))
    return ret[0].lower() + ret[1:]


def _parse_config_value(value: Any, schema: dict, key: str, tool_id: str) -> Any:
    schema_type = schema["type"]
    if schema_type == "string":
        return value
    elif schema_type == "boolean":
        return value.lower() == "true"
    elif schema_type == "array":
        item_type = schema["items"]["type"]
        if item_type == "string":
            return value.split(",")
        if item_type == "object":
            return json.loads(value) if isinstance(value, str) else value
        raise ValueError(f"Invalid array item type '{item_type}' while parsing tool '{tool_id}' config '{key}'")
    else:
        raise ValueError(f"Invalid type '{schema_type}' while parsing tool '{tool_id}' config '{key}'")


def _parse_new_files(tool_id: str, zip_file: ZipFile, root_folder: str, user: User) -> Dict[str, File]:
    return {path.rsplit("/", 1)[1]: _parse_new_file(path, zip_file, user) for path in zip_file.namelist() if path.startswith(f"{root_folder}{tool_id}/") and path != f"{root_folder}{tool_id}/"}


def _parse_new_file(file_path: str, zip_file: ZipFile, user: User) -> File:
    file_name = file_path.rsplit("/", 1)[1]
    return File(
        name=file_name,
        content_type=mimetypes.guess_type(file_name)[0] or "",
        user_id=user.id,
        content=zip_file.read(file_path),
        status=FileStatus.PENDING
    )


async def _remove_tool_file(file: File, tc: AgentToolConfig, db: AsyncSession):
    await AgentToolConfigFileRepository(db).delete(tc.agent_id, tc.tool_id, file.id)


async def _update_tool_file(file: File, new_file: File, tc: AgentToolConfig, tool: AgentTool, user: User, db: AsyncSession, background_tasks: BackgroundTasks):
    existing_file = cast(File, await AgentToolConfigFileRepository(db).find_with_content_by_ids(tc.agent_id, tc.tool_id, file.id))
    if existing_file.content != new_file.content:
        await _remove_tool_file(existing_file, tc, db)
        await upload_tool_file(new_file, tool, tc.agent_id, user, db, background_tasks)


async def _configure_new_tool(tool_id: str, new_config: Dict[str, Any], agent: Agent,tool: AgentTool, zip_file: ZipFile, root_folder: str, user: User, db: AsyncSession, background_tasks: BackgroundTasks) -> bool:
    auth_required = await _configure_parsed_tool(tool_id, new_config, agent, None, tool, user, db)
    files = _parse_new_files(tool_id, zip_file, root_folder, user)
    for file in files.values():
        await upload_tool_file(file, tool, agent.id, user, db, background_tasks)
    return auth_required


async def _update_tests(agent_id: int, tests: List[Dict[str, Any]], user_id: int, db: AsyncSession):
    await _delete_existing_tests(agent_id, db)
    for parsed_test in tests:
        await _add_new_test(agent_id, parsed_test, user_id, db)


async def _delete_existing_tests(agent_id: int, db: AsyncSession):
    test_repo = TestCaseRepository(db)
    existing_tests = await test_repo.find_by_agent(agent_id)
    for test in existing_tests:
        await TestCaseRepository(db).delete(test)


async def _add_new_test(agent_id: int, test: Dict[str, Any], user_id: int, db: AsyncSession):
    thread = await ThreadRepository(db).add(Thread(
        agent_id=agent_id,
        user_id=user_id,
        is_test_case=True,
        name=test['name']
    ))
    evaluator_id = await _create_new_evaluator(test['evaluator'], db) if test.get('evaluator') else None
    await TestCaseRepository(db).save(TestCase(
        thread_id=thread.id,
        agent_id=agent_id,
        evaluator_id=evaluator_id
    ))
    for i, msg in enumerate(test['messages']):
        origin = ThreadMessageOrigin.USER if i % 2 == 0 else ThreadMessageOrigin.AGENT
        await ThreadMessageRepository(db).add(ThreadMessage(
            thread_id=thread.id,
            origin=origin,
            text=msg['text']
        ))
