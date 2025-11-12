import io
from typing import Callable, Optional, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AsyncOpenAI
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
            streaming=streaming)

    def supports_model(self, model: str) -> bool:
        return model in env.openai_model_id_mapping

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
            model=env.openai_model_id_mapping[model])

class ReasoningTokenCountingChatOpenAI(ChatOpenAI):

    # we override this method which is the one used by get_num_tokens_from_messages to count the tokens
    def _get_encoding_model(self) -> tuple[str, tiktoken.Encoding]:
        return get_encoding_model(self.model_name, lambda: ChatOpenAI._get_encoding_model(self))


def get_encoding_model(model_name: str, default: Callable[[], tuple[str, tiktoken.Encoding]]) -> tuple[str, tiktoken.Encoding]:
        if model_name and model_name.startswith("o"):
            # we return gpt-4o for o- series since it is supported by existing implementation of get_num_tokens_from_messages
            return "gpt-4o", tiktoken.get_encoding("o200k_base")
        return default()
