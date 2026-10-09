import io
from collections.abc import Sequence
from typing import Callable, Iterable, Optional, Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AsyncAzureOpenAI, RateLimitError
from pydantic import SecretStr

from ..core.env import env
from .domain import AiModelProvider
from .openai_provider import (
    _StrictToolsMixin,
    count_tokens,
    get_num_tokens_from_messages_sanitizing_unsupported_blocks,
    openai_resilience_kwargs,
    use_responses_api_for_model,
)


class AzureProvider(AiModelProvider):

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool, request_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> BaseChatModel:
        deployment = env.azure_model_deployments[model]
        return ReasoningTokenCountingAzureChatOpenAI(
            azure_endpoint=env.azure_endpoints[deployment.endpoint_index],
            azure_deployment=deployment.deployment_name,
            api_version=env.azure_api_version,
            api_key=env.azure_api_keys[deployment.endpoint_index],
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            streaming=streaming,
            stream_usage=True,
            use_responses_api=use_responses_api_for_model(model),
            **openai_resilience_kwargs(request_timeout, max_retries))

    def supports_model(self, model: str) -> bool:
        return model in env.azure_model_deployments

    def is_rate_limit_error(self, exc: Exception) -> bool:
        return isinstance(exc, RateLimitError)

    async def transcribe_audio(self, file: io.BytesIO, model: str) -> str:
        deployment = env.azure_model_deployments[model]
        client = AsyncAzureOpenAI(
            api_key=cast(SecretStr, env.azure_api_keys[deployment.endpoint_index]).get_secret_value(),
            api_version=env.azure_api_version,
            azure_endpoint=env.azure_endpoints[deployment.endpoint_index]
        )
        response = await client.audio.transcriptions.create(
            file=file,
            model=deployment.deployment_name
        )
        return response.text

    def build_embedding(self, model: str, request_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> AzureOpenAIEmbeddings:
        deployment = env.azure_model_deployments[model]
        return AzureOpenAIEmbeddings(
            azure_endpoint=env.azure_endpoints[deployment.endpoint_index],
            azure_deployment=deployment.deployment_name,
            api_version=env.azure_api_version,
            api_key=env.azure_api_keys[deployment.endpoint_index],
            **openai_resilience_kwargs(request_timeout, max_retries))

    def count_tokens(self, txt: str, model: str) -> int:
        return count_tokens(txt, model)


class ReasoningTokenCountingAzureChatOpenAI(_StrictToolsMixin, AzureChatOpenAI):

    def get_num_tokens_from_messages(
        self,
        messages: Sequence[BaseMessage],
        tools: Optional[Sequence[Any]] = None,
        *,
        allow_fetching_images: bool = True,
    ) -> int:
        return get_num_tokens_from_messages_sanitizing_unsupported_blocks(
            token_counter=super().get_num_tokens_from_messages,
            messages=messages,
            tools=tools,
            allow_fetching_images=allow_fetching_images,
        )


class UsageTrackingAzureOpenAIEmbeddings(AzureOpenAIEmbeddings):
    usage_tracker: Callable[[int], None]

    def _tokenize(self, texts: list[str], chunk_size: int) -> tuple[Iterable[int], list[list[int] | str], list[int], list[int]]:
        ret = super()._tokenize(texts, chunk_size)
        self.usage_tracker(sum(ret[3]))
        return ret
