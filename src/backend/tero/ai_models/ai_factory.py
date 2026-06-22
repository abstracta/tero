from typing import Any, Optional

from ..core.env import env
from .aws_provider import AWSProvider
from .azure_provider import AzureProvider
from .domain import AiModelProvider, LlmModel, LlmModelType
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider
from .vllm_provider import VllmAiProvider


providers: list[AiModelProvider] = []
if env.openai_api_key:
    providers.append(OpenAIProvider())
if env.azure_endpoints and env.azure_api_keys:
    providers.append(AzureProvider())
if env.aws_access_key_id and env.aws_secret_access_key or env.aws_s3_bucket:
    providers.append(AWSProvider())
if env.google_api_key:
    providers.append(GoogleProvider())
if env.vllm_urls and env.vllm_api_keys:
    providers.append(VllmAiProvider())


def get_provider(model: str) -> AiModelProvider:
    for provider in providers:
        if provider.supports_model(model):
            return provider
    raise ValueError(f"No provider found for model: {model}")


def has_valid_provider(model: str) -> bool:
    return any(provider.supports_model(model) for provider in providers)


def build_chat_model(model: str, temperature: Optional[float]=None, reasoning_effort: Optional[str] = None) -> Any:
    return get_provider(model).build_chat_model(model, temperature, reasoning_effort)


def build_streaming_chat_model(model: str, temperature: Optional[float]=None, reasoning_effort: Optional[str]=None) -> Any:
    return get_provider(model).build_streaming_chat_model(model, temperature, reasoning_effort)


def build_internal_generator_chat_model(model: LlmModel) -> Any:
    temperature = env.internal_generator_temperature if model.model_type == LlmModelType.CHAT else None
    reasoning_effort = (
        env.internal_generator_reasoning_effort.lower()
        if model.model_type == LlmModelType.REASONING
        else None
    )
    return build_chat_model(model.id, temperature, reasoning_effort)
