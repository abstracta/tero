from functools import cache
import logging
import json
from typing import Any, Callable, Optional, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import RateLimitError
from tokenizers import Tokenizer

from ..core.env import env
from .domain import AiModelProvider


logger = logging.getLogger(__name__)


class VllmAiProvider(AiModelProvider):

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool) -> BaseChatModel:
        index, model_id = self._find_vllm_model(model)
        return VLLMChatModel(
            base_url=env.vllm_urls[index],
            api_key=env.vllm_api_keys[index],
            model=model_id,
            temperature=temperature,
            streaming=streaming)

    def _find_vllm_model(self, model: str) -> tuple[int, str]:
        return next((index, item[1]) for index, item in enumerate(env.vllm_model_id_mapping.items()) if item[0] == model)

    def supports_model(self, model: str) -> bool:
        return model in env.vllm_model_id_mapping

    def is_rate_limit_error(self, exc: Exception) -> bool:
        return isinstance(exc, RateLimitError)

    def build_embedding(self, model: str) -> Embeddings:
        index, model_id = self._find_vllm_model(model)
        return OpenAIEmbeddings(
            base_url=env.vllm_urls[index],
            api_key=env.vllm_api_keys[index],
            model=model_id,
            embedding_ctx_length=env.embedding_context_limit,
            tiktoken_enabled=False)

    def count_tokens(self, txt: str, model: str) -> int:
        _, model_id = self._find_vllm_model(model)
        return len(get_tokenizer(model_id).encode(txt).ids)


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(model_name)


class VLLMChatModel(ChatOpenAI):

    def get_token_ids(self, text: str) -> list[int]:
        tokenizer = get_tokenizer(self.model_name)
        return tokenizer.encode(text).ids

    def get_num_tokens_from_messages(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool] | None = None,
    ) -> int:
        total = 0
        for msg in messages:
            # Message overhead (role, separators, etc.)
            total += 4

            content = msg.content
            if isinstance(content, str):
                total += len(self.get_token_ids(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        total += len(self.get_token_ids(item['text']))
                    elif isinstance(item, str):
                        total += len(self.get_token_ids(item))

        if tools:
            openai_tools = [convert_to_openai_tool(tool) for tool in tools]
            tools_json = json.dumps(openai_tools)
            total += len(self.get_token_ids(tools_json))

        total += 2
        return total
