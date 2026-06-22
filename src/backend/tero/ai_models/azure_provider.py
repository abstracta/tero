import io
from collections.abc import Sequence
from typing import Callable, Iterable, Optional, Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AsyncAzureOpenAI, RateLimitError
from pydantic import SecretStr
import tiktoken

from ..core.env import env
from .domain import AiModelProvider
from .openai_provider import get_encoding_model, count_tokens, get_num_tokens_from_messages_sanitizing_unsupported_blocks


class AzureProvider(AiModelProvider):

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool) -> BaseChatModel:
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
            # use responses api for codex models because they are not supported by completion endpoint
            use_responses_api="-codex" in model)

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

    def build_embedding(self, model: str) -> AzureOpenAIEmbeddings:
        deployment = env.azure_model_deployments[model]
        return AzureOpenAIEmbeddings(
            azure_endpoint=env.azure_endpoints[deployment.endpoint_index],
            azure_deployment=deployment.deployment_name,
            api_version=env.azure_api_version,
            api_key=env.azure_api_keys[deployment.endpoint_index])

    def count_tokens(self, txt: str, model: str) -> int:
        return count_tokens(txt, model)


class ReasoningTokenCountingAzureChatOpenAI(AzureChatOpenAI):

    # we override this method which is the one used by get_num_tokens_from_messages to count the tokens
    def _get_encoding_model(self) -> tuple[str, tiktoken.Encoding]:
        return get_encoding_model(self.model_name, lambda: AzureChatOpenAI._get_encoding_model(self))

    def get_num_tokens_from_messages(self, messages: Sequence[BaseMessage], tools: Optional[Sequence[Any]] = None) -> int:
        return get_num_tokens_from_messages_sanitizing_unsupported_blocks(
            token_counter=super().get_num_tokens_from_messages,
            messages=messages,
            tools=tools,
        )


class UsageTrackingAzureOpenAIEmbeddings(AzureOpenAIEmbeddings):
    usage_tracker: Callable[[int], None]

    def _tokenize(self, texts: list[str], chunk_size: int) -> tuple[Iterable[int], list[list[int] | str], list[int], list[int]]:
        ret = super()._tokenize(texts, chunk_size)
        self.usage_tracker(sum(ret[3]))
        return ret
