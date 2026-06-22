from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from sqlmodel.ext.asyncio.session import AsyncSession

from ...ai_models import ai_factory
from ...ai_models.domain import LlmModel
from ...ai_models.repos import AiModelRepository
from ...core.env import env
from ...threads.core import trim_messages_to_fit_model
from ...threads.domain import MAX_THREAD_NAME_LENGTH
from ...usage.domain import MessageUsage
from ...usage.repos import UsageRepository
from ..domain import Agent


_PROMPT_TEMPLATE = (
    "You generate descriptive test case titles.\n"
    "Craft a concise, human-friendly title (40 characters or fewer when possible) "
    "that highlights the capability or behavior the user message is asking for. "
    "Focus on the topic being exercised rather than referencing validations or agent responses.\n"
    "Only respond with the title, without quotes or additional text, and do not include the agent name or description.\n\n"
    "Agent name: {agent_name}\n"
    "Agent description: {agent_description}\n"
    "User message: {user_message}\n"
    "Expected agent response: {agent_message}"
)


async def generate_test_case_name(
    agent: Agent,
    user_message: str,
    agent_message: str,
    user_id: int,
    db: AsyncSession,
) -> str:
    model = cast(LlmModel, await AiModelRepository(db).find_by_id(env.internal_generator_model))
    if not model:
        raise RuntimeError("Internal generator model not found")

    message_usage = MessageUsage(user_id=user_id, agent_id=agent.id, model_id=model.id)
    try:
        llm = ai_factory.build_internal_generator_chat_model(model)
        trimmed_user_message, trimmed_agent_message = _trim_messages_to_fit_model(
            user_message, agent_message, model, llm, agent
        )
        prompt = _build_prompt(agent, trimmed_user_message, trimmed_agent_message)
        response = await llm.ainvoke([HumanMessage(prompt)])
        response = cast(AIMessage, response)
        message_usage.increment_with_metadata(response.usage_metadata, model)
        content = cast(str, response.content).strip()
        if len(content) > MAX_THREAD_NAME_LENGTH:
            content = content[:MAX_THREAD_NAME_LENGTH]
        return content
    finally:
        await UsageRepository(db).add(message_usage)


def _trim_messages_to_fit_model(
    user_message: str, agent_message: str, model: LlmModel, llm: BaseChatModel, agent: Agent
) -> tuple[str, str]:
    messages = [
        HumanMessage(content=user_message),
        AIMessage(content=agent_message),
    ]
    prompt_skeleton = _PROMPT_TEMPLATE.format(
        agent_name=agent.name or "",
        agent_description=agent.description or "",
        user_message="",
        agent_message="",
    )
    token_counter = llm.get_num_tokens_from_messages
    reserved_tokens = token_counter([HumanMessage(prompt_skeleton)])
    trimmed_messages = trim_messages_to_fit_model(
        messages,
        token_counter=token_counter,
        model=model,
        reserved_tokens=reserved_tokens,
    )
    trimmed_user = trimmed_messages[0].content if trimmed_messages else ""
    trimmed_agent = trimmed_messages[1].content if len(trimmed_messages) > 1 else ""
    return cast(str, trimmed_user), cast(str, trimmed_agent)


def _build_prompt(agent: Agent, user_message: str, agent_message: str) -> str:
    return _PROMPT_TEMPLATE.format(
        agent_name=agent.name or "",
        agent_description=agent.description or "",
        user_message=user_message,
        agent_message=agent_message,
    )
