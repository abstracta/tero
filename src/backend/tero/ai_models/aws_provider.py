import io
from typing import Optional


import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from ..core.env import env
from .domain import AiModelProvider


class AWSProvider(AiModelProvider):

    def __init__(self):
        super().__init__()
        self.model_arn_map = {}

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool) -> BaseChatModel:
        aws_model_id = env.aws_model_id_mapping.get(model)
        if not aws_model_id:
            raise ValueError(f"Model {model} not supported by AWS")
        return ChatBedrockConverse(
            aws_access_key_id=env.aws_access_key_id,
            aws_secret_access_key=env.aws_secret_access_key,
            region_name=env.aws_region,
            model=self._get_model_arn(aws_model_id),
            provider=self._get_model_provider(aws_model_id),
            temperature=temperature)
    
    def supports_model(self, model: str) -> bool:
        return model in env.aws_model_id_mapping
    
    def _get_model_arn(self, model: str) -> str:
        if model not in self.model_arn_map:
            if not env.aws_access_key_id or not env.aws_secret_access_key or not env.aws_region:
                raise ValueError("AWS credentials are not set")
            bedrock_client = boto3.client(
                aws_access_key_id=env.aws_access_key_id.get_secret_value(), 
                aws_secret_access_key=env.aws_secret_access_key.get_secret_value(), 
                region_name=env.aws_region, 
                service_name='bedrock')
            inference_profiles = bedrock_client.list_inference_profiles()
            inference_profile_summaries = inference_profiles['inferenceProfileSummaries']
            for inference_profile in inference_profile_summaries:
                profile_id = inference_profile["inferenceProfileId"]
                model_id = profile_id.replace("us.", "").replace("eu.", "")
                if model_id == model:
                    arn = inference_profile["inferenceProfileArn"]
                    self.model_arn_map[model] = arn
                    return arn
            raise ValueError(f"Model {model} not supported")
        return self.model_arn_map[model]
    
    def _get_model_provider(self, model: str) -> str:
        return model.split(".")[0]
