import io
from collections.abc import Sequence
from typing import Callable, Iterable, Optional, Any, cast

from langchain_core.messages import BaseMessage
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AsyncOpenAI, RateLimitError
from pydantic import SecretStr
import tiktoken

from ..core.env import env
from .domain import AiModelProvider


class OpenAIProvider(AiModelProvider):

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool) -> BaseChatModel:
        openai_model_id = env.openai_model_id_mapping[model]
        return ReasoningTokenCountingChatOpenAI(
            api_key=env.openai_api_key,
            model=openai_model_id,
            temperature=temperature,
            streaming=streaming,
            # use responses api for codex models because they are not supported by completion endpoint
            use_responses_api="-codex" in model)

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

    def build_embedding(self, model: str) -> Embeddings:
        return OpenAIEmbeddings(
            api_key=env.openai_api_key,
            embedding_ctx_length=env.embedding_context_limit,
            model=env.openai_model_id_mapping[model])

    def count_tokens(self, txt: str, model: str) -> int:
        return count_tokens(txt, model)


def count_tokens(txt: str, model: str) -> int:
    return len(tiktoken.encoding_for_model(model).encode(txt))


class ReasoningTokenCountingChatOpenAI(ChatOpenAI):

    # we override this method which is the one used by get_num_tokens_from_messages to count the tokens
    def _get_encoding_model(self) -> tuple[str, tiktoken.Encoding]:
        return get_encoding_model(self.model_name, lambda: ChatOpenAI._get_encoding_model(self))

    # override this method to avoid failing token count when using gpt 5 codex which returns reasoning elements in message
    def get_num_tokens_from_messages(self, messages: Sequence[BaseMessage], tools: Optional[Sequence[Any]] = None) -> int:
        return get_num_tokens_from_messages_sanitizing_unsupported_blocks(
            token_counter=super().get_num_tokens_from_messages,
            messages=messages,
            tools=tools,
        )


def get_encoding_model(model_name: Optional[str], default: Callable[[], tuple[str, tiktoken.Encoding]]) -> tuple[str, tiktoken.Encoding]:
        if model_name and model_name.startswith("o"):
            # we return gpt-4o for o- series since it is supported by existing implementation of get_num_tokens_from_messages
            return "gpt-4o", tiktoken.get_encoding("o200k_base")
        return default()


_UNSUPPORTED_TOKEN_COUNT_BLOCK_TYPES = {"reasoning", "function_call"}


def get_num_tokens_from_messages_sanitizing_unsupported_blocks(
    token_counter: Callable[[Sequence[BaseMessage], Optional[Sequence[Any]]], int],
    messages: Sequence[BaseMessage],
    tools: Optional[Sequence[Any]],
) -> int:
    if _has_unsupported_blocks(messages):
        messages = _sanitize_messages_for_token_count(messages)
    return token_counter(messages, tools)


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
