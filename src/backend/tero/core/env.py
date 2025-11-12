import os
from typing import List, Optional

import re

from pydantic import SecretStr, field_validator, BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureModelDeployment(BaseModel):
    deployment_name: str
    endpoint_index: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(enable_decoding=False)
    
    db_url : str
    secret_encryption_key : SecretStr
    frontend_url : str
    frontend_path : Optional[str] = None
    openid_url : str
    frontend_openid_url : Optional[str] = None
    openid_client_id : str
    openid_scope : str
    allowed_users : list[str] = []
    contact_email : str
    azure_app_insights_connection : Optional[str] = None
    azure_endpoints : list[str]
    azure_api_keys : list[SecretStr]
    azure_api_version : str
    azure_model_deployments : dict[str, AzureModelDeployment]
    azure_doc_intelligence_endpoint : Optional[str] = None
    azure_doc_intelligence_key : Optional[SecretStr] = None
    azure_doc_intelligence_cost_per_1k_pages_usd : float
    temperatures: dict[str, float]
    monthly_usd_limit_default : int
    internal_generator_model : str
    internal_generator_temperature : float
    internal_evaluator_model : Optional[str] = None
    internal_evaluator_temperature : Optional[float] = None
    agent_default_model : Optional[str] = None
    agent_basic_models : List[str]
    default_agent_name : str
    embedding_model : str
    embedding_cost_per_1k_tokens : float
    transcription_model : str
    aws_access_key_id : Optional[SecretStr] = None
    aws_secret_access_key : Optional[SecretStr] = None
    aws_region : str
    aws_model_id_mapping : dict[str, str]
    google_api_key : Optional[SecretStr] = None
    google_model_id_mapping : dict[str, str]
    openai_api_key : Optional[SecretStr] = None
    openai_model_id_mapping : dict[str, str]
    docs_tool_chunk_size : int
    docs_tool_chunk_overlap : int
    docs_tool_retrieve_top : int
    docs_tool_description_chunk_size : int
    docs_tool_description_chunk_overlap : int
    tool_oauth_token_ttl_minutes : int
    tool_oauth_state_ttl_minutes : int
    mcp_tool_oauth_client_registration_ttl_minutes : int
    web_tool_tavily_api_key : Optional[SecretStr] = None
    web_tool_google_custom_search_engine_id : Optional[str] = None
    web_tool_google_api_key : Optional[SecretStr] = None
    web_tool_tavily_cost_per_1k_credits_usd : float
    web_tool_google_cost_per_1k_searches_usd : float
    browser_tool_playwright_mcp_url : str
    browser_tool_playwright_output_dir : str
    
    def is_local_env(self) -> bool:
        found = re.search('@([^/]+)(?:\\d+)?/', self.db_url)
        if not found:
            return False
        db_host = found.group(1)
        return db_host == 'localhost' or db_host == 'postgres'
    
    @field_validator('azure_model_deployments', mode='before')
    @classmethod
    def decode_model_deployments(cls, v: str) -> dict[str, AzureModelDeployment]:
        ret = {}
        for pair in v.split(','):
            model_id, deployment = pair.split(':', 1)
            deployment_parts = deployment.split('@', 1)
            ret[model_id] = AzureModelDeployment(deployment_name=deployment_parts[0], endpoint_index=int(deployment_parts[1]) if len(deployment_parts) > 1 else 0)
        return ret
    
    @field_validator('aws_model_id_mapping', 'google_model_id_mapping', 'openai_model_id_mapping', mode='before')
    @classmethod
    def decode_model_id_mapping(cls, v: str) -> dict[str, str]:
        return {k: v for k, v in (pair.split(':', 1) for pair in v.split(','))} if v else {}
    
    @field_validator('temperatures', mode='before')
    @classmethod
    def decode_temperatures(cls, v: str) -> dict[str, float]:
        return {k: float(v) for k, v in (pair.split(':', 1) for pair in v.split(','))} if v else {}
    
    @field_validator('allowed_users', 'azure_endpoints', 'azure_api_keys', 'agent_basic_models', mode='before')
    @classmethod
    def decode_list(cls, v: str) -> list[str]:
        return v.split(',') if v else []

    @model_validator(mode="after")
    def set_defaults(self):
        self.agent_default_model = self.agent_default_model or self.internal_generator_model
        self.internal_evaluator_model = self.internal_evaluator_model or self.internal_generator_model
        self.internal_evaluator_temperature = self.internal_evaluator_temperature or self.internal_generator_temperature
        return self


def _find_env_file() -> str:
    for path in ['.env', '../.env', '../../.env']:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('No .env file found')

    
env = Settings(_env_file=_find_env_file(), _env_file_encoding='utf-8') # type: ignore
