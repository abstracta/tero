import io
from collections.abc import Sequence
from typing import Callable, Iterable, Optional, Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai import AsyncOpenAI, RateLimitError
from pydantic import SecretStr
import tiktoken

from ..core.env import env
from .domain import AiModelProvider


def openai_resilience_kwargs(request_timeout: Optional[float], max_retries: Optional[int]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return kwargs


def use_responses_api_for_model(model: str) -> bool:
    # Legacy models that still use Chat Completions. Newer models use the Responses API.
    return model not in {
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    }


class OpenAIProvider(AiModelProvider):

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool, request_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> BaseChatModel:
        openai_model_id = env.openai_model_id_mapping[model]
        return ReasoningTokenCountingChatOpenAI(
            api_key=env.openai_api_key,
            model=openai_model_id,
            temperature=temperature,
            streaming=streaming,
            use_responses_api=use_responses_api_for_model(model),
            **openai_resilience_kwargs(request_timeout, max_retries))

    def supports_model(self, model: str) -> bool:
        return model in env.openai_model_id_mapping

    def is_rate_limit_error(self, exc: Exception) -> bool:
        return isinstance(exc, RateLimitError)

    async def transcribe_audio(self, file: io.BytesIO, model: str) -> str:
        client = AsyncOpenAI(api_key=cast(SecretStr, env.openai_api_key).get_secret_value())
        response = await client.audio.transcriptions.create(
            file=file,
            model=env.openai_model_id_mapping[model]
        )
        return response.text

    def build_embedding(self, model: str, request_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> Embeddings:
        return OpenAIEmbeddings(
            api_key=env.openai_api_key,
            embedding_ctx_length=env.embedding_context_limit,
            model=env.openai_model_id_mapping[model],
            **openai_resilience_kwargs(request_timeout, max_retries))

    def count_tokens(self, txt: str, model: str) -> int:
        return count_tokens(txt, model)


def count_tokens(txt: str, model: str) -> int:
    return len(tiktoken.encoding_for_model(model).encode(txt))


class _StrictToolsMixin(BaseChatOpenAI):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        # Set strict=False to avoid failing tool invocation in some models (e.g. gpt-5.6 luna) when schema is not 100% accurate
        # or when schema has optional fields (since luna is in any case sending empty strings for optional fields).
        kwargs.setdefault("strict", env.openai_tool_strict)
        return super().bind_tools(tools, **kwargs)


class ReasoningTokenCountingChatOpenAI(_StrictToolsMixin, ChatOpenAI):

    # Sanitize unsupported reasoning and function_call blocks before token counting.
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


_UNSUPPORTED_TOKEN_COUNT_BLOCK_TYPES = {"reasoning", "function_call"}


def get_num_tokens_from_messages_sanitizing_unsupported_blocks(
    token_counter: Callable[..., int],
    messages: Sequence[BaseMessage],
    tools: Optional[Sequence[Any]],
    allow_fetching_images: bool,
) -> int:
    if _has_unsupported_blocks(messages):
        messages = _sanitize_messages_for_token_count(messages)
    return token_counter(messages, tools, allow_fetching_images=allow_fetching_images)


def _has_unsupported_blocks(messages: Sequence[BaseMessage]) -> bool:
    return any(
        isinstance(message.content, list)
        and any(
            isinstance(item, dict) and item.get("type") in _UNSUPPORTED_TOKEN_COUNT_BLOCK_TYPES
            for item in message.content
        )
        for message in messages
    )


def _sanitize_messages_for_token_count(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    ret = []
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            ret.append(message)
            continue

        cleaned_content = [
            item
            for item in content
            if not (isinstance(item, dict) and item.get("type") in _UNSUPPORTED_TOKEN_COUNT_BLOCK_TYPES)
        ]

        ret.append(
            message.model_copy(update={"content": cleaned_content})
        )
    return ret


class UsageTrackingOpenAIEmbeddings(OpenAIEmbeddings):
    usage_tracker: Callable[[int], None]

    def _tokenize(self, texts: list[str], chunk_size: int) -> tuple[Iterable[int], list[list[int] | str], list[int], list[int]]:
        ret = super()._tokenize(texts, chunk_size)
        self.usage_tracker(sum(ret[3]))
        return ret
