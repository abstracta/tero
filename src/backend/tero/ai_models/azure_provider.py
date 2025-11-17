import io
from typing import Optional, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AsyncAzureOpenAI
from pydantic import SecretStr
import tiktoken

from ..core.env import env
from .domain import AiModelProvider
from .openai_provider import get_encoding_model


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
            stream_usage=True)

    def supports_model(self, model: str) -> bool:
        return model in env.azure_model_deployments

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


class ReasoningTokenCountingAzureChatOpenAI(AzureChatOpenAI):

    # we override this method which is the one used by get_num_tokens_from_messages to count the tokens
    def _get_encoding_model(self) -> tuple[str, tiktoken.Encoding]:
        return get_encoding_model(self.model_name, lambda: AzureChatOpenAI._get_encoding_model(self))
