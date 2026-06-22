from abc import ABC, abstractmethod
from enum import Enum
import io
from typing import Any, Callable, Optional

from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tracers import ConsoleCallbackHandler
from pydantic import computed_field
from sqlmodel import Field
from tokenizers import Tokenizer

from ..core.env import env
from ..core.domain import CamelCaseModel


class LlmModelType(Enum):
    REASONING = 'REASONING'
    CHAT = 'CHAT'


class LlmModelVendor(Enum):
    AMAZON = 'AMAZON'
    ANTHROPIC = 'ANTHROPIC'
    GOOGLE = 'GOOGLE'
    OPENAI = 'OPENAI'
    QWEN = 'QWEN'



class LlmModelBase(CamelCaseModel):
    id: str = Field(primary_key=True, max_length=20)
    name: str = Field(max_length=30)
    description: str = Field(max_length=200)
    model_type: LlmModelType
    model_vendor: LlmModelVendor
    token_limit: int
    output_token_limit: int
    prompt_1k_token_usd: float
    completion_1k_token_usd: float


class LlmModel(LlmModelBase, table=True):
    __tablename__: Any = "llm_model"

    @computed_field
    @property
    def is_basic(self) -> bool:
        return self.id in env.agent_basic_models


class LlmModelResponse(LlmModelBase):
    is_basic: bool
    cost_multiplier: Optional[float] = None
    is_base_cost_model: bool = False

    @staticmethod
    def from_model(model: LlmModel, base_completion_cost: Optional[float], base_model_id: Optional[str] = None) -> 'LlmModelResponse':
        multiplier = round(model.completion_1k_token_usd / base_completion_cost, 2) if base_completion_cost else None
        return LlmModelResponse.model_validate({**model.model_dump(), 'cost_multiplier': multiplier, 'is_base_cost_model': model.id == base_model_id})


class LlmTemperature(Enum):
    CREATIVE = 'CREATIVE'
    NEUTRAL = 'NEUTRAL'
    PRECISE = 'PRECISE'

    def get_float(self):
        return env.temperatures[self.value]


class ReasoningEffort(Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'


class AiModelProvider(ABC):

    def build_chat_model(self, model: str, temperature: Optional[float]=None, reasoning_effort: Optional[str] = None) -> BaseChatModel:
        ret = self._build_chat_model(model, temperature, reasoning_effort, False)
        return self._prepare_chat_model(ret)

    def _prepare_chat_model(self, model: Any) -> BaseChatModel:
        model.verbose = True
        model.callbacks=[StdOutCallbackHandler(), ConsoleCallbackHandler()]
        return model

    def build_streaming_chat_model(self, model: str, temperature: Optional[float]=None, reasoning_effort: Optional[str]=None) -> BaseChatModel:
        ret = self._build_chat_model(model, temperature, reasoning_effort, True)
        return self._prepare_chat_model(ret)

    @abstractmethod
    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool) -> BaseChatModel:
        pass

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        pass

    async def transcribe_audio(self, file: io.BytesIO, model: str) -> str:
        raise NotImplementedError("Transcription is not yet supported by this provider")

    def build_embedding(self, model: str) -> Embeddings:
        raise NotImplementedError("Embedding is not yet supported by this provider")

    def count_tokens(self, txt: str, model: str) -> int:
        raise NotImplementedError("Counting tokens is not yet supported by this provider")

    def is_rate_limit_error(self, exc: Exception) -> bool:
        return False
