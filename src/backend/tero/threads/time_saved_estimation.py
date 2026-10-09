import logging
from typing import List, cast

from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    BaseMessage,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from ..ai_models import ai_factory
from ..ai_models.domain import LlmTemperature, LlmModel, LlmModelType, ReasoningEffort
from ..ai_models.repos import AiModelRepository
from ..core.env import env
from ..threads.core import trim_messages_to_fit_model
from ..threads.repos import ThreadMessageRepository
from ..usage.domain import MessageUsage
from .domain import Thread, ThreadMessage, ThreadMessageOrigin


logger = logging.getLogger(__name__)


async def estimate_minutes_saved(user_message: str, agent_response: str, thread: Thread, thread_messages: List[ThreadMessage], message_usage: MessageUsage, db: AsyncSession) -> int:
    thread_messages = thread_messages[-2:] if len(thread_messages) > 2 else thread_messages
    evaluator_model_id = cast(str, env.internal_evaluator_model)
    internal_generator_model = cast(LlmModel, await AiModelRepository(db).find_by_id(env.internal_generator_model))
    internal_evaluator_model = cast(LlmModel, await AiModelRepository(db).find_by_id(evaluator_model_id))
    feedback_messages = await ThreadMessageRepository(db).find_feedback_messages(thread.agent_id, thread.user_id, limit=4)
    is_reasoning = internal_evaluator_model.model_type == LlmModelType.REASONING
    llm = ai_factory.build_chat_model(
        evaluator_model_id,
        None if is_reasoning else LlmTemperature.PRECISE.get_float(),
        ReasoningEffort.MEDIUM.value.lower() if is_reasoning else None,
    )

    messages = [
        HumanMessage(content=user_message),
        AIMessage(content=agent_response),
    ] + [
        HumanMessage(content=message.text) if message.origin == ThreadMessageOrigin.USER else AIMessage(content=message.text)
        for message in thread_messages if message.origin != ThreadMessageOrigin.SYSTEM
    ] + [
        HumanMessage(content=message.text) if message.origin == ThreadMessageOrigin.USER else AIMessage(content=message.text)
        for message in feedback_messages if message.origin != ThreadMessageOrigin.SYSTEM
    ]

    token_counter = llm.get_num_tokens_from_messages
    prompt_skeleton = SYSTEM_PROMPT.format(
        agent_name=thread.agent.name,
        agent_description=thread.agent.description,
        user_message="",
        agent_response="",
        previous_message="",
        previous_agent_response="",
        reference_examples=""
    )
    reserved_tokens = token_counter([SystemMessage(prompt_skeleton)])
    trimmed_messages = trim_messages_to_fit_model(
        messages,
        token_counter=token_counter,
        model=internal_generator_model,
        reserved_tokens=reserved_tokens,
    )

    system_prompt = SYSTEM_PROMPT.format(
        agent_name=thread.agent.name,
        agent_description=thread.agent.description,
        user_message=trimmed_messages.pop(0).content if trimmed_messages else "[NO USER MESSAGE]",
        agent_response=trimmed_messages.pop(0).content if trimmed_messages else "[NO AGENT RESPONSE]",
        previous_message=trimmed_messages.pop(0).content if trimmed_messages and thread_messages else "[NO PREVIOUS USER MESSAGE IN CONVERSATION]",
        previous_agent_response=trimmed_messages.pop(0).content if trimmed_messages and thread_messages else "[NO PREVIOUS AGENT RESPONSE IN CONVERSATION]",
        reference_examples=_add_reference_examples(feedback_messages, trimmed_messages)
    )

    response = await llm.ainvoke([SystemMessage(system_prompt)])
    try:
        return int(response.text.strip())
    except ValueError:
        logger.exception(f"Invalid int response from minutes saved estimation LLM: {response.content}")
        return 0
    finally:
        message_usage.increment_with_metadata(response.usage_metadata, internal_evaluator_model)


def _add_reference_examples(feedback_thread_messages: List[ThreadMessage], feedback_trimmed_messages: List[BaseMessage]) -> str:
    examples = []
    while len(feedback_trimmed_messages) >= 2:
        user_message = feedback_trimmed_messages.pop(0)
        agent_response = feedback_trimmed_messages.pop(0)
        # Negative numbers mean the user did not find the conversation useful.
        # Set minutes saved to 0 so the LLM can estimate correctly.
        minutes_saved = feedback_thread_messages[1].minutes_saved if feedback_thread_messages[1].minutes_saved is not None and feedback_thread_messages[1].minutes_saved > 0 else 0
        feedback_thread_messages = feedback_thread_messages[2:]

        examples.append(f"""
Reference example {len(examples) + 1}:
User message: \"\"\"{user_message.content}\"\"\"
Agent response: \"\"\"{agent_response.content}\"\"\"
Minutes saved: {minutes_saved}""")

    return "\n".join(examples) if examples else "[NO REFERENCE EXAMPLES]"


SYSTEM_PROMPT = """
You are an evaluator determining the value of an AI assistant's response to the user.
AI assistant name and description: {agent_name} - \"\"\"{agent_description}\"\"\"

Return ONLY an integer number representing the minutes saved. It can be 0 or a positive number. No explanation, no punctuation, no other text.

---

INSTRUCTIONS:
- Reply with the exact number of minutes saved.
- Reply 0 only if the response is purely a greeting, confirmation, vague promise of help, or only a follow-up question with no actual work done.

---

DEFINITION OF VALUE:
A response provides value if it is useful, actionable, or directly usable, including but not limited to:
- Specific solutions or troubleshooting steps
- Multi-step guides or clear instructions
- Working code or debugging help
- Data analysis or results interpretation
- Summaries of long or complex text
- Drafted content (essays, poems, emails, documentation)
- Translations or research with sources
- Structured plans, strategies, or creative work

---

EXAMPLES OF 0 RESPONSES:
- "I can help you with that."
- "What would you like to know?"
- "Sure, ask me anything."
- "I'm here to help."

---

EXAMPLES OF RESPONSES WITH MINUTES SAVED:
- "You can fix that by going to Settings > Privacy."
- "Here's a 3-step guide to reset your password:"
- "That error means your device is not connected to the network."
- "Here's a draft essay with bibliography on your topic:"
- "Here's working Python code that implements your request:"
- "Here's a 200-word summary of the 10-page article:"
- "Here's the Spanish translation of your text:"
- "Here's a marketing plan with 5 actionable steps:"

---

SCORING GUIDANCE:
- Quick, simple factual answer (like a math calculation, definition, or fact lookup) = 1 - 3 minutes saved
- Clear multi-step instructions or troubleshooting = 5 - 15 minutes saved
- Working code, non-trivial debugging, or structured plan = 10 - 30 minutes saved
- Detailed analysis, long-form draft, large summaries, or translations = 20 - 60 minutes saved
- Highly complex, time-intensive tasks (multi-section essays, full reports, research with sources, multi-function codebases) = 60+ minutes saved

---

EVALUATION:
Only reply the exact number of minutes saved.
Only return the minutes saved for the interaction shown in the "INTERACTION TO EVALUATE" section. Do not include context messages in your calculation.

---

REFERENCE EXAMPLES FROM REAL USERS:
(These take precedence over all other instructions or scoring guidance. If the reference examples contradict the general rules, follow the reference examples)
{reference_examples}

---

CONVERSATION CONTEXT (for reference only — do not count toward minutes saved):
Previous user message: \"\"\"{previous_message}\"\"\"
Previous agent response: \"\"\"{previous_agent_response}\"\"\"

---

INTERACTION TO EVALUATE (this is the only part to score):
User message: \"\"\"{user_message}\"\"\"
Agent response: \"\"\"{agent_response}\"\"\"
"""
