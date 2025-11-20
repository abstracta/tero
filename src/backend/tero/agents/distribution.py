import base64
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
from jinja2.nodes import Name
from PIL import Image
from pydantic import BaseModel
from slugify import slugify
from sqlmodel.ext.asyncio.session import AsyncSession

from ..ai_models.domain import LlmModelType, LlmModel
from ..ai_models.repos import AiModelRepository
from ..core.assets import solve_asset_path
from ..files.domain import File, FileStatus
from ..threads.domain import Thread, ThreadMessage, ThreadMessageOrigin
from ..threads.repos import ThreadRepository, ThreadMessageRepository
from ..tools.core import AgentTool
from ..tools.oauth import ToolOAuthRequest
from ..tools.repos import ToolRepository
from ..users.domain import User
from .domain import Agent, AgentUpdate, AgentToolConfig, LlmTemperature, ReasoningEffort
from .prompts.domain import AgentPrompt
from .prompts.repos import AgentPromptRepository
from .repos import AgentRepository, AgentToolConfigRepository, AgentToolConfigFileRepository
from .template_parser import JinjaTemplateParser
from .test_cases.domain import TestCase
from .test_cases.repos import TestCaseRepository
from .tool_file import upload_tool_file


logger = logging.getLogger(__name__)


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
            icon_data = _create_icon_with_background(agent.icon, agent.icon_bg_color) if agent.icon_bg_color else agent.icon
            zip_file.writestr(f"{agent_name}/icon.png", icon_data)
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
        model_config=_format_model_config(agent),
        conversation_starters=[_format_prompt(p) for p in prompts if p.starter],
        user_prompts=[_format_prompt(p) for p in prompts if not p.starter],
        tools=[_format_tool(tool) for tool in tools],
        tests=[await _format_test(test, db) for test in await TestCaseRepository(db).find_by_agent(agent.id)]
    )


def _build_jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(solve_asset_path('.', __file__)), trim_blocks=True, lstrip_blocks=True)


def _format_model_config(agent: Agent) -> dict:
    return {"Temperature": agent.temperature.value.capitalize()} if agent.model.model_type == LlmModelType.CHAT \
        else {"Reasoning": agent.reasoning_effort.value.capitalize()}


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
    return ",".join(value) if isinstance(value, list) else value


async def _format_test(test_case: TestCase, db: AsyncSession) -> dict:
    messages = await ThreadMessageRepository(db).find_by_thread_id(test_case.thread_id)
    return {
        "name": test_case.thread.name,
        "messages": messages
    }


def _create_icon_with_background(icon_bytes: bytes, bg_color: str) -> bytes:
    try:
        icon_img = Image.open(BytesIO(icon_bytes))
        if icon_img.mode != 'RGBA':
            icon_img = icon_img.convert('RGBA')
        try:
            bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
            bg_image = Image.new('RGBA', icon_img.size, bg_rgb + (255,))
            result_img = Image.alpha_composite(bg_image, icon_img)
        except (ValueError, TypeError):
            result_img = icon_img
        output = BytesIO()
        result_img.save(output, format='PNG')
        return output.getvalue()
    except Exception as e:
        logger.warning(f"Error creating icon with background, using icon as is.", exc_info=True)
        return icon_bytes


async def update_agent_from_zip(agent: Agent, zip_content: bytes, user: User, db: AsyncSession, background_tasks: BackgroundTasks) -> Agent:
    with _open_zip_file(zip_content) as zip_file:
        found_root_folder = [ name.rsplit('/', 1)[0] for name in zip_file.namelist() if name.endswith('/agent.md') ]
        # supporting zip without root folder in case users zip the folder contents and not the folder itself
        root_folder = f"{found_root_folder[0]}/" if found_root_folder else ""
        
        if not f"{root_folder}agent.md" in zip_file.namelist():
            raise ValueError("Agent markdown file not found")
        agent_md = zip_file.read(f"{root_folder}agent.md").decode('utf-8')
        async with aiofiles.open(solve_asset_path("agent-template.md", __file__)) as template_file:
            parsed = JinjaTemplateParser(_build_jinja_env()).parse(agent_md, await template_file.read())

        # we find tools before updating the agent to avoid leaving some things changed and others not when tools are invalid
        parsed_tools = parsed.get('tools', [])
        tools = await _find_tools(parsed_tools)
        await _update_agent(agent, parsed, zip_file, root_folder, db)
        await _update_prompts(agent.id, parsed.get('conversation_starters', []), parsed.get('user_prompts', []), user.id, db)
        await _update_tools(agent, parsed_tools, tools, zip_file, root_folder, user, db, background_tasks)
        await _update_tests(agent.id, parsed.get('tests', []), user.id, db)
        return agent

    
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


async def _find_tools(parsed_tools: List[Dict[str, Any]]) -> Dict[str, AgentTool]:
    ret = {}
    for parsed in parsed_tools:
        tool_id = _parse_tool_id(parsed)
        tool = ToolRepository().find_by_id(tool_id)
        if not tool:
            raise ValueError(f"Tool {tool_id} not found")
        ret[tool_id] = tool
    return ret


def _parse_tool_id(tool: Dict[str, Any]) -> str:
    return tool['name'].lower().replace(' ', '-')


async def _update_agent(agent: Agent, parsed: Dict[str, Any], zip_file: ZipFile, root_folder: str, db: AsyncSession):
    update = AgentUpdate()
    update.name = parsed['name']
    update.description = parsed['description']
    update.system_prompt = parsed['system_prompt']

    model = await _find_model_by_name(parsed['model_name'], db)
    update.model_id = model.id
    update.temperature = LlmTemperature[parsed['model_config']['Temperature'].upper()] if model.model_type == LlmModelType.CHAT else None
    update.reasoning_effort = ReasoningEffort[parsed['model_config']['Reasoning'].upper()] if model.model_type == LlmModelType.REASONING else None
    
    icon_path = f"{root_folder}icon.png"
    if icon_path in zip_file.namelist():
        update.icon = base64.b64encode(zip_file.read(icon_path)).decode('utf-8')

    agent.update_with(update)
    agent = await AgentRepository(db).update(agent)


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


async def _update_tools(agent: Agent, parsed_tools: List[Dict[str, Any]], tools: Dict[str, AgentTool], zip_file: ZipFile, root_folder: str, user: User, db: AsyncSession, background_tasks: BackgroundTasks):
    tools_dict = {_parse_tool_id(tool): tool for tool in parsed_tools}
    tool_config_repo = AgentToolConfigRepository(db)
    await tool_config_repo.delete_drafts(agent.id)
    existing_tools = {tc.tool_id: tc for tc in await tool_config_repo.find_by_agent_id(agent.id)}

    for tc in existing_tools.values():
        if not tc.tool_id in tools_dict:
            await _remove_tool(tc, user.id, db)
        else:
            await _update_tool(tc, tools_dict[tc.tool_id], tools[tc.tool_id], zip_file, root_folder, user, db, background_tasks)
    
    for tool_id, config in tools_dict.items():
        if not tool_id in existing_tools:
            await _configure_new_tool(tool_id, config, agent, tools[tool_id], zip_file, root_folder, user, db, background_tasks)


async def _remove_tool(tc: AgentToolConfig, user_id: int, db: AsyncSession):
    tool = cast(AgentTool, ToolRepository().find_by_id(tc.tool_id))
    tool.configure(tc.agent, user_id, tc.config, db)
    await tool.teardown()
    await AgentToolConfigFileRepository(db).delete_by_agent_id_and_tool_id(tc.agent_id, tc.tool_id)
    await AgentToolConfigRepository(db).delete(tc.agent_id, tc.tool_id)


async def _update_tool(tc: AgentToolConfig, new_config: Dict[str, Any], tool: AgentTool, zip_file: ZipFile, root_folder: str, user: User, db: AsyncSession, background_tasks: BackgroundTasks):
    await _configure_parsed_tool(tc.tool_id, new_config, tc.agent, tc, tool, user, db)
    existing_files = {f.name: f for f in await AgentToolConfigFileRepository(db).find_by_agent_id_and_tool_id(tc.agent_id, tc.tool_id)}
    new_files = await _parse_new_files(tc.tool_id, zip_file, root_folder, user)

    for file_name, file in existing_files.items():
        if not file_name in new_files:
            await _remove_tool_file(file, tc, db)
        else:
            await _update_tool_file(file, new_files[file_name], tc, tool, user, db, background_tasks)
    
    for file_name, new_file in new_files.items():
        if not file_name in existing_files:
            await upload_tool_file(new_file, tool, tc.agent_id, user, db, background_tasks)


async def _configure_parsed_tool(tool_id: str, new_config: Dict[str, Any], agent: Agent, tc: Optional[AgentToolConfig], tool: AgentTool, user: User, db: AsyncSession):
    config = _parse_tool_config(new_config.get('config', {}), tool, tool_id)
    tool.configure(agent, user.id, config, db)
    try:
        await tool.setup(prev_config=tc)
    except ToolOAuthRequest as e:
        logger.error(f"Tool OAuth required by tool {tool_id} imported by user {user.id} but still not supported in imports", exc_info=True)
    await AgentToolConfigRepository(db).add(AgentToolConfig(
            agent_id=agent.id,
            tool_id=tool_id,
            config=config
        ))


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
        if not item_type == "string":
            raise ValueError(f"Invalid array item type '{item_type}' while parsing tool '{tool_id}' config '{key}'")
        return value.split(",")
    else:
        raise ValueError(f"Invalid type '{schema_type}' while parsing tool '{tool_id}' config '{key}'")


async def _parse_new_files(tool_id: str, zip_file: ZipFile, root_folder: str, user: User) -> Dict[str, File]:
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


async def _configure_new_tool(tool_id: str, new_config: Dict[str, Any], agent: Agent,tool: AgentTool, zip_file: ZipFile, root_folder: str, user: User, db: AsyncSession, background_tasks: BackgroundTasks):
    await _configure_parsed_tool(tool_id, new_config, agent, None, tool, user, db)
    files = await _parse_new_files(tool_id, zip_file, root_folder, user)
    for file in files.values():
        await upload_tool_file(file, tool, agent.id, user, db, background_tasks)


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
    await TestCaseRepository(db).save(TestCase(
        thread_id=thread.id,
        agent_id=agent_id
    ))
    for i, msg in enumerate(test['messages']):
        origin = ThreadMessageOrigin.USER if i % 2 == 0 else ThreadMessageOrigin.AGENT
        await ThreadMessageRepository(db).add(ThreadMessage(
            thread_id=thread.id,
            origin=origin,
            text=msg['text']
        ))
