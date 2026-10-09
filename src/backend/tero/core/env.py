import contextvars
import json
import logging
import os
import re
from typing import Any, Callable, List, Optional

from pydantic import SecretStr, field_validator, BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class AzureModelDeployment(BaseModel):
    deployment_name: str
    endpoint_index: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(enable_decoding=False, extra="allow")

    db_url : str
    secret_encryption_key : SecretStr
    frontend_url : str
    frontend_path : Optional[str] = None
    openid_url : str
    frontend_openid_url : Optional[str] = None
    openid_client_id : str
    openid_scope : str
    allowed_users : list[str] = []
    disable_publish_global : Optional[bool] = False
    max_file_size_mb : int = 50
    contact_email : str
    azure_endpoints : list[str] = []
    azure_api_keys : list[SecretStr] = []
    azure_api_version : Optional[str] = None
    azure_model_deployments : dict[str, AzureModelDeployment] = {}
    azure_doc_intelligence_endpoint : Optional[str] = None
    azure_doc_intelligence_key : Optional[SecretStr] = None
    azure_doc_intelligence_cost_per_1k_pages_usd : Optional[float] = None
    temperatures: dict[str, float]
    monthly_usd_limit_default : int
    internal_generator_model : str
    internal_generator_temperature : float
    internal_generator_reasoning_effort : str
    internal_evaluator_model : Optional[str] = None
    agent_default_model : Optional[str] = None
    agent_basic_models : List[str]
    default_agent_name : str
    embedding_model : str
    embedding_context_limit : int = 8191
    embedding_cost_per_1k_tokens : float
    transcription_model : str
    aws_access_key_id : Optional[SecretStr] = None
    aws_secret_access_key : Optional[SecretStr] = None
    aws_region : str
    aws_model_id_mapping : dict[str, str]
    aws_textract_cost_per_1k_pages_usd : Optional[float] = None
    aws_s3_bucket : Optional[str] = None
    google_api_key : Optional[SecretStr] = None
    google_model_id_mapping : dict[str, str]
    openai_api_key : Optional[SecretStr] = None
    openai_model_id_mapping : dict[str, str]
    openai_tool_strict : bool = False
    vllm_urls : List[str] = []
    vllm_api_keys : List[SecretStr] = []
    vllm_model_id_mapping : dict[str, str] = {}
    enterprise_edition : Optional[bool] = True
    docs_tool_chunk_size : int
    docs_tool_chunk_overlap : int
    docs_tool_retrieve_top : int
    docs_tool_description_chunk_size : int
    docs_tool_description_chunk_overlap : int
    docs_tool_hybrid_search_enabled : bool = True
    docs_tool_rrf_k : int = 60
    tool_oauth_token_ttl_minutes : int
    tool_oauth_state_ttl_minutes : int
    mcp_tool_oauth_client_registration_ttl_minutes : int
    web_tool_tavily_api_key : Optional[SecretStr] = None
    web_tool_google_custom_search_engine_id : Optional[str] = None
    web_tool_google_api_key : Optional[SecretStr] = None
    azure_devops_tool_tenant_id : Optional[str] = None
    azure_devops_tool_client_id : Optional[str] = None
    azure_devops_tool_client_secret : Optional[SecretStr] = None
    jira_tool_client_id : Optional[str] = None
    jira_tool_client_secret : Optional[SecretStr] = None
    sql_tool_azure_tenant_id : Optional[str] = None
    sql_tool_azure_client_id : Optional[str] = None
    sql_tool_azure_client_secret : Optional[SecretStr] = None
    web_tool_tavily_cost_per_1k_credits_usd : float
    web_tool_google_cost_per_1k_searches_usd : float
    browser_tool_playwright_mcp_url : str
    browser_tool_playwright_output_dir : str = "var/playwright-output"
    codebase_tool_repos_path : str = "var/codebase-repos"
    odoo_host : Optional[str] = None
    odoo_db : Optional[str] = None
    odoo_user : Optional[str] = None
    odoo_password : Optional[SecretStr] = None
    odoo_employee_ignored_ids : Optional[List[int]] = None
    odoo_employee_email_domains : Optional[List[str]] = None
    talentlms_domain : Optional[str] = None
    talentlms_api_key : Optional[SecretStr] = None
    keycloak_admin_password : Optional[SecretStr] = None
    smtp_hostname : Optional[str] = None
    smtp_port : Optional[int] = None
    smtp_username : Optional[str] = None
    smtp_password : Optional[SecretStr] = None
    tenants : Optional[List[str]] = None
    tenant_overrides: Optional[dict[str, dict[str, Any]]] = None
    tenant_secret_overrides: Optional[dict[str, dict[str, Any]]] = None

    def is_local_env(self) -> bool:
        found = re.search('@([^/]+)(?:\\d+)?/', self.db_url)
        if not found:
            return False
        db_host = found.group(1)
        return db_host == 'localhost' or db_host == 'postgres'

    @field_validator('azure_model_deployments', mode='before')
    @classmethod
    def decode_model_deployments(cls, v: str) -> dict[str, AzureModelDeployment]:
        if not v:
            return {}
        ret = {}
        for pair in v.split(','):
            model_id, deployment = pair.split(':', 1)
            deployment_parts = deployment.split('@', 1)
            ret[model_id] = AzureModelDeployment(deployment_name=deployment_parts[0], endpoint_index=int(deployment_parts[1]) if len(deployment_parts) > 1 else 0)
        return ret

    @field_validator('aws_model_id_mapping', 'google_model_id_mapping', 'openai_model_id_mapping', 'vllm_model_id_mapping', mode='before')
    @classmethod
    def decode_model_id_mapping(cls, v: str) -> dict[str, str]:
        return {k: v for k, v in (pair.split(':', 1) for pair in v.split(','))} if v else {}

    @field_validator('temperatures', mode='before')
    @classmethod
    def decode_temperatures(cls, v: str) -> dict[str, float]:
        return {k: float(v) for k, v in (pair.split(':', 1) for pair in v.split(','))} if v else {}

    @field_validator('allowed_users', 'azure_endpoints', 'azure_api_keys', 'odoo_employee_email_domains', 'agent_basic_models', 'vllm_urls', 'vllm_api_keys', 'tenants', mode='before')
    @classmethod
    def decode_list(cls, v: str) -> list[str]:
        return v.split(',') if v else []

    @field_validator('tenant_overrides', 'tenant_secret_overrides', mode='before')
    @classmethod
    def decode_tenant_overrides(cls, v: str) -> dict[str, dict[str, Any]]:
        if not v or not v.strip():
            return {}
        return  {
            tenant_id: {
                attr: (fn(val) if (fn := _TENANT_OVERRIDABLE.get(attr)) is not None else val)
                for attr, val in attrs.items()
            }
            for tenant_id, attrs in json.loads(v.strip()).items()
        }

    @field_validator('odoo_employee_ignored_ids', mode='before')
    @classmethod
    def decode_int_list(cls, v: str) -> list[int]:
        return [int(id) for id in v.split(',')] if v else []

    @model_validator(mode="after")
    def set_defaults(self):
        self.agent_default_model = self.agent_default_model or self.internal_generator_model
        self.internal_evaluator_model = self.internal_evaluator_model or self.internal_generator_model
        return self

    @model_validator(mode="after")
    def warn_extra_fields(self):
        if self.__pydantic_extra__:
            logger.warning(f"Ignoring unexpected variables found in .env file: {', '.join(self.__pydantic_extra__.keys())}. Please review if you mistyped the variable name or if the variable name has changed in a recent version of the application.")
        return self


def _find_env_file() -> str:
    for path in ['.env', '../.env', '../../.env']:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('No .env file found')


def _decode_tenant_secret(v: Any) -> SecretStr | None:
    return v if v is None or isinstance(v, SecretStr) else SecretStr(str(v))


def _decode_tenant_secret_list(v: Any) -> list[SecretStr]:
    items = Settings.decode_list(v) if isinstance(v, str) else (v if isinstance(v, list) else [])
    return [x if isinstance(x, SecretStr) else SecretStr(str(x)) for x in items if x is not None]


_TENANT_OVERRIDABLE: dict[str, Callable[[Any], Any] | None] = {
    "agent_default_model": None,
    "allowed_users": Settings.decode_list,
    "aws_access_key_id": _decode_tenant_secret,
    "aws_model_id_mapping": Settings.decode_model_id_mapping,
    "aws_region": None,
    "aws_secret_access_key": _decode_tenant_secret,
    "azure_api_keys": _decode_tenant_secret_list,
    "azure_doc_intelligence_endpoint": None,
    "azure_doc_intelligence_key": _decode_tenant_secret,
    "azure_endpoints": Settings.decode_list,
    "azure_model_deployments": Settings.decode_model_deployments,
    "contact_email": None,
    "disable_publish_global": None,
    "enterprise_edition": None,
    "frontend_url": None,
    "google_api_key": _decode_tenant_secret,
    "google_model_id_mapping": Settings.decode_model_id_mapping,
    "monthly_usd_limit_default": None,
    "openai_api_key": _decode_tenant_secret,
    "openai_model_id_mapping": Settings.decode_model_id_mapping,
    "secret_encryption_key": _decode_tenant_secret,
    "vllm_api_keys": _decode_tenant_secret_list,
    "vllm_model_id_mapping": Settings.decode_model_id_mapping,
    "vllm_urls": Settings.decode_list,
    "web_tool_google_custom_search_engine_id": None,
    "web_tool_google_api_key": _decode_tenant_secret,
    "web_tool_tavily_api_key": _decode_tenant_secret,
    "talentlms_domain": None,
}


_base_env = Settings(_env_file=_find_env_file(), _env_file_encoding='utf-8') # type: ignore


_current_tenant_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tenant_name", default=""
)


def set_current_tenant_name(tenant: str) -> None:
    _current_tenant_name.set(tenant)


def get_current_tenant_name() -> str:
    return _current_tenant_name.get()


def _get_current_tenant_override():
    tenant_name = _current_tenant_name.get()
    if not tenant_name:
        return None
    o = getattr(_base_env, "tenant_overrides", None) or {}
    s = getattr(_base_env, "tenant_secret_overrides", None) or {}
    merged = {**o.get(tenant_name, {}), **s.get(tenant_name, {})}
    return merged or None


class _TenantEnv:
    """Proxy that returns tenant-specific config when in tenant context."""
    __slots__ = ("_base",)

    def __init__(self):
        object.__setattr__(self, "_base", _base_env)

    def __getattribute__(self, name: str):
        base = object.__getattribute__(self, "_base")
        if name in _TENANT_OVERRIDABLE:
            overrides = _get_current_tenant_override()
            if overrides and name in overrides:
                return overrides[name]
        return getattr(base, name)

    def __setattr__(self, name: str, value):
        base = object.__getattribute__(self, "_base")
        setattr(base, name, value)

env = _TenantEnv()
