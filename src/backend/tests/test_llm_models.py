import pytest
from unittest.mock import patch

from .common import *

from tero.ai_models.ai_factory import providers
from tero.ai_models.aws_provider import AWSProvider
from tero.ai_models.domain import LlmModel, LlmModelResponse, LlmModelType, LlmModelVendor
from tero.ai_models.google_provider import GoogleProvider


def _base_models() -> List[LlmModel]:
    return [
        LlmModel(id="gpt-5-nano", model_type=LlmModelType.REASONING, name="GPT-5 Nano", description="This is a new reasoning model for simpler tasks. Lower cost than GPT-5 Mini.",
                 token_limit=4000000, output_token_limit=128000, prompt_1k_token_usd=0.00005, completion_1k_token_usd=0.0004, model_vendor=LlmModelVendor.OPENAI),
        LlmModel(id="gpt-5-mini", model_type=LlmModelType.REASONING, name="GPT-5 Mini", description="This is a new reasoning model with a good balance between cost and intelligence. Suited for most agent tasks.",
                 token_limit=400000, output_token_limit=128000, prompt_1k_token_usd=0.00025, completion_1k_token_usd=0.002, model_vendor=LlmModelVendor.OPENAI),
        LlmModel(id="gemini-2.5-flash", model_type=LlmModelType.CHAT, name="Gemini 2.5 Flash", description="This is a fast and efficient model, comparable to GPT-5 Nano, optimized for speed while maintaining high quality responses.",
                 token_limit=1048576, output_token_limit=65536, prompt_1k_token_usd=0.0003, completion_1k_token_usd=0.0025, model_vendor=LlmModelVendor.GOOGLE),
        LlmModel(id="gpt-5.4-mini", model_type=LlmModelType.REASONING, name="GPT-5.4 Mini", description="This is a fast reasoning model with strong performance. Suited for most agent tasks like research, drafting, and data analysis.",
                 token_limit=400000, output_token_limit=128000, prompt_1k_token_usd=0.00075, completion_1k_token_usd=0.0045, model_vendor=LlmModelVendor.OPENAI),
        LlmModel(id="claude-haiku-4-5", model_type=LlmModelType.CHAT, name="Claude Haiku 4.5", description="This is a fast model suited for straightforward tasks. Good for summarization, extraction, and other tasks that don't require complex reasoning.",
                 token_limit=200000, output_token_limit=64000, prompt_1k_token_usd=0.001, completion_1k_token_usd=0.005, model_vendor=LlmModelVendor.ANTHROPIC),
        LlmModel(id="gpt-5.1-codex-max", model_type=LlmModelType.REASONING, name="GPT-5.1 Codex Max", description="This is the most capable Codex model for long-running agentic coding. Best for complex codebases and multi-step tasks. Similar pricing to GPT-5.1 Codex.",
                 token_limit=400000, output_token_limit=128000, prompt_1k_token_usd=0.00125, completion_1k_token_usd=0.01, model_vendor=LlmModelVendor.OPENAI),
        LlmModel(id="gemini-2.5-pro", model_type=LlmModelType.CHAT, name="Gemini 2.5 Pro", description="This is an advanced reasoning model, outperforming GPT-5 Mini with a larger context while being more affordable.",
                 token_limit=1048576, output_token_limit=65536, prompt_1k_token_usd=0.00125, completion_1k_token_usd=0.01, model_vendor=LlmModelVendor.GOOGLE),
        LlmModel(id="gpt-5", model_type=LlmModelType.REASONING, name="GPT-5", description="This is the best reasoning model for coding and complex agentic tasks from OpenAI. Aligned with GPT-5 Mini for most use cases.",
                 token_limit=400000, output_token_limit=128000, prompt_1k_token_usd=0.00125, completion_1k_token_usd=0.01, model_vendor=LlmModelVendor.OPENAI),
        LlmModel(id="gpt-5.4", model_type=LlmModelType.REASONING, name="GPT-5.4", description="This is the most capable reasoning model from OpenAI for professional work. Best for demanding analysis and complex agentic tasks. Higher cost than GPT-5.",
                 token_limit=1050000, output_token_limit=128000, prompt_1k_token_usd=0.0025, completion_1k_token_usd=0.015, model_vendor=LlmModelVendor.OPENAI),
        LlmModel(id="claude-sonnet-4", model_type=LlmModelType.CHAT, name="Claude Sonnet 4", description="This is a similar model to GPT-5 but with different strengths.",
                 token_limit=200000, output_token_limit=64000, prompt_1k_token_usd=0.003, completion_1k_token_usd=0.015, model_vendor=LlmModelVendor.ANTHROPIC),
        LlmModel(id="claude-sonnet-4-6", model_type=LlmModelType.CHAT, name="Claude Sonnet 4.6", description="This is an improved version of Claude Sonnet 4 with better reasoning. Good for complex analysis, coding, and creative writing. Similar pricing to Claude Sonnet 4.",
                 token_limit=200000, output_token_limit=64000, prompt_1k_token_usd=0.003, completion_1k_token_usd=0.015, model_vendor=LlmModelVendor.ANTHROPIC),
        LlmModel(id="claude-opus-4-6", model_type=LlmModelType.CHAT, name="Claude Opus 4.6", description="This is a more capable model than Claude Sonnet 4 for highly complex tasks. Best for demanding reasoning and analysis. Higher cost than Claude Sonnet models.",
                 token_limit=200000, output_token_limit=64000, prompt_1k_token_usd=0.005, completion_1k_token_usd=0.025, model_vendor=LlmModelVendor.ANTHROPIC),
    ]


def _to_responses(models: List[LlmModel]) -> List[LlmModelResponse]:
    base_model_id = env.agent_default_model
    base_cost = next((m.completion_1k_token_usd for m in models if m.id == base_model_id), None)
    return [LlmModelResponse.from_model(m, base_cost, base_model_id) for m in models]


@pytest.fixture(name="ai_models")
def ai_models_fixture() -> List[LlmModelResponse]:
    return _to_responses(_base_models())

async def test_find_models(client: AsyncClient, ai_models: List[LlmModel]):
    resp = await client.get(f"{BASE_PATH}/models")
    assert_response(resp, ai_models)


@patch("tero.ai_models.ai_factory.providers", [p for p in providers if not isinstance(p, AWSProvider)])
async def test_claude_sonnet_4_not_available(client: AsyncClient):
    resp = await client.get(f"{BASE_PATH}/models")
    filtered = [m for m in _base_models() if m.id not in ("claude-sonnet-4", "claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5")]
    assert_response(resp, _to_responses(filtered))


@patch("tero.ai_models.ai_factory.providers", [p for p in providers if not isinstance(p, GoogleProvider)])
async def test_gemini_2_5_pro_not_available(client: AsyncClient):
    resp = await client.get(f"{BASE_PATH}/models")
    filtered = [m for m in _base_models() if m.id not in ("gemini-2.5-flash", "gemini-2.5-pro")]
    assert_response(resp, _to_responses(filtered))
