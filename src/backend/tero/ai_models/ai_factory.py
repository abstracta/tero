from typing import Any, Optional

from ..core.env import env
from .aws_provider import AWSProvider
from .azure_provider import AzureProvider
from .domain import AiModelProvider, LlmModel, LlmModelType
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider
from .vllm_provider import VllmAiProvider


def _build_providers() -> list[AiModelProvider]:
    ret = []
    if env.openai_api_key:
        ret.append(OpenAIProvider())
    if env.azure_endpoints and env.azure_api_keys:
        ret.append(AzureProvider())
    if env.aws_access_key_id and env.aws_secret_access_key or env.aws_s3_bucket:
        ret.append(AWSProvider())
    if env.google_api_key:
        ret.append(GoogleProvider())
    if env.vllm_urls and env.vllm_api_keys:
        ret.append(VllmAiProvider())
    return ret


def get_provider(model: str) -> AiModelProvider:
    for provider in _build_providers():
        if provider.supports_model(model):
            return provider
    raise ValueError(f"No provider found for model: {model}")


def has_valid_provider(model: str) -> bool:
    return any(provider.supports_model(model) for provider in _build_providers())


def build_chat_model(model: str, temperature: Optional[float]=None, reasoning_effort: Optional[str] = None, request_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> Any:
    return get_provider(model).build_chat_model(model, temperature, reasoning_effort, request_timeout, max_retries)


def build_streaming_chat_model(model: str, temperature: Optional[float]=None, reasoning_effort: Optional[str]=None) -> Any:
    return get_provider(model).build_streaming_chat_model(model, temperature, reasoning_effort)


def build_internal_generator_chat_model(model: LlmModel, request_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> Any:
    temperature = env.internal_generator_temperature if model.model_type == LlmModelType.CHAT else None
    reasoning_effort = (
        env.internal_generator_reasoning_effort.lower()
        if model.model_type == LlmModelType.REASONING
        else None
    )
    return build_chat_model(model.id, temperature, reasoning_effort, request_timeout, max_retries)
