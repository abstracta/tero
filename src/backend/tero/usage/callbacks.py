from typing import Any, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import LLMResult

from ..ai_models.domain import LlmModel
from .domain import MessageUsage


class LlmUsageTrackingCallback(AsyncCallbackHandler):

    def __init__(self, usage: MessageUsage, model: LlmModel, agent_id: Optional[int] = None):
        self._usage = usage
        self._model = model
        self._agent_id = agent_id

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        metadata: Optional[UsageMetadata] = None
        for generation in (g for gl in (response.generations or []) for g in gl):
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                metadata = usage
                break
        if not metadata and response.llm_output:
            token_usage = response.llm_output.get("token_usage")
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
                output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
                metadata = UsageMetadata(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=token_usage.get("total_tokens") or (input_tokens + output_tokens),
                )
        if metadata:
            self._usage.increment_with_metadata(metadata, self._model, self._agent_id)
