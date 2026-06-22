from typing import Optional, cast
from langchain_core.messages import HumanMessage, AIMessage
from sqlmodel.ext.asyncio.session import AsyncSession

from ..ai_models import ai_factory
from ..ai_models.repos import AiModelRepository
from ..core.env import env
from ..usage.domain import MessageUsage
from ..usage.repos import UsageRepository
from .domain import Agent, AutomaticAgentField


async def generate_agent_field(agent: Agent, field: AutomaticAgentField, user_id: int, db: AsyncSession) -> str:
    model = await AiModelRepository(db).find_by_id(env.internal_generator_model)
    if not model:
        raise ValueError("Internal generator model not found")
    message_usage = MessageUsage(user_id=user_id, agent_id=agent.id, model_id=model.id)
    try:
        prompt = _build_prompt(agent, field)
        llm = ai_factory.build_internal_generator_chat_model(model)
        response = await llm.ainvoke([HumanMessage(prompt)])
        response = cast(AIMessage, response)
        message_usage.increment_with_metadata(response.usage_metadata, model)
        max_length = _get_field_max_length(field)
        if max_length:
            response.content = response.content[:max_length]
        return cast(str, response.content)
    finally:
        await UsageRepository(db).add(message_usage)


def _get_field_max_length(field: AutomaticAgentField) -> Optional[int]:
    max_length = [m for m in Agent.model_fields[field.value.lower()].metadata if hasattr(m, 'max_length')]
    return max_length[0].max_length if max_length else None

    
def _build_prompt(agent: Agent, field: AutomaticAgentField) -> str:
    ret = ""
    if field == AutomaticAgentField.NAME:
        ret = f"""Generate a short name (no more than 30 characters) that allows users to easily identify the agent. 
Use properly formatted names that are friendly to the user (like Test Generator). Do not use camel case.
Do not include any formatting like bold, italic, underline, etc. Just the generated name.
When possible generate names that make the agent unique and identifiable. Avoid generic names like "Assistant" or "Helper"."""
    elif field == AutomaticAgentField.DESCRIPTION:
        ret = f"""Generate a short description (no more than 100 characters) that allows users to easily understand the agent's purpose.
Do not include the name of the agent in the description.
Describe the agent purpose, like "Allows to search the web for information". Do not directly include instructions or words like "you" or "your".
Give more relevance to the agent name, when present, than to the system prompt.
Avoid general descriptions that do not provide any information about the agent (like: A helpful assistant)."""
    elif field == AutomaticAgentField.SYSTEM_PROMPT:
        ret = f"""Generate a system prompt that allows the agent to perform it's intended purpose.
The system prompt should encourage the agent to use the provided tools and information provided in context to answer user questions.
It should also avoid generating responses that are not based on tools or previous context.
Avoid any fancy formatting like bold, italic, underline, blockquotes and code blocks. Use basic markdown formatting like lists, paragraphs, etc.
The agent should answer in the same language as the user.
The prompt should encourage the use of markdown to format responsess, potentially including code blocks, tables, plantuml diagrams code blocks, echarts configuration code blocks or any standard markdown format"""
    if agent.name and field != AutomaticAgentField.NAME:
        ret += f"\n\n##Agent name\n\n{agent.name}"
    if agent.description and field != AutomaticAgentField.DESCRIPTION:
        ret += f"\n\n##Agent description\n\n{agent.description}"
    if agent.system_prompt and field != AutomaticAgentField.SYSTEM_PROMPT:
        ret += f"\n\n##Agent system prompt\n\n{agent.system_prompt}"
    return ret
