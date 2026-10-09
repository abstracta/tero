from typing import Optional

from google.api_core.exceptions import ResourceExhausted
from google.genai.errors import ClientError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from ..core.env import env
from .domain import AiModelProvider


class GoogleProvider(AiModelProvider):

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool, request_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> BaseChatModel:
        google_model_id = env.google_model_id_mapping.get(model)
        kwargs: dict = {}
        if request_timeout is not None:
            kwargs["timeout"] = request_timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return ChatGoogleGenerativeAI(
            google_api_key=env.google_api_key,
            model=google_model_id,
            temperature=temperature,
            **kwargs)

    def supports_model(self, model: str) -> bool:
        return model in env.google_model_id_mapping

    def is_rate_limit_error(self, exc: Exception) -> bool:
        if isinstance(exc, ResourceExhausted):
            return True
        if isinstance(exc, ClientError) and exc.code == 429:
            return True
        if isinstance(exc, ChatGoogleGenerativeAIError):
            cause = exc.__cause__
            return isinstance(cause, ClientError) and cause.code == 429
        return False
