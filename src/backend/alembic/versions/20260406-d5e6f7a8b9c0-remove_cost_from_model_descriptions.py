"""remove-cost-from-model-descriptions

Revision ID: d5e6f7a8b9c0
Revises: 5f4c2a1b9d8e
Create Date: 2026-04-06

"""

from typing import Sequence, Union
from alembic import op


revision: str = 'd5e6f7a8b9c0'
down_revision: Union[str, None] = '5f4c2a1b9d8e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE llm_model
        SET description = CASE id
            WHEN 'gemini-2.5-pro' THEN 'This is an advanced reasoning model, outperforming GPT-5 Mini with a larger context.'
            WHEN 'gpt-5-mini' THEN 'This is a fast reasoning model with good performance, suited for most agent tasks like research, drafting, and data analysis.'
            WHEN 'gpt-5-nano' THEN 'This is a very fast reasoning model for simple tasks, such as summarization, and extraction.'
            WHEN 'claude-sonnet-4-6' THEN 'This is an improved version of Claude Sonnet 4 with better reasoning. Good for complex analysis, coding, and creative writing.'
            WHEN 'claude-opus-4-6' THEN 'This is a more capable model than Claude Sonnet 4 for highly complex tasks. Best for demanding reasoning and analysis.'
            WHEN 'gpt-5.4' THEN 'This is the most capable reasoning model from OpenAI. Best for demanding analysis and complex agentic tasks.'
            WHEN 'gpt-5.1-codex-max' THEN 'This is the most capable Codex model for coding. Best for complex codebases and multi-step tasks.'
        END
        WHERE id IN ('gemini-2.5-pro', 'gpt-5-mini', 'gpt-5-nano', 'claude-sonnet-4-6', 'claude-opus-4-6', 'gpt-5.4', 'gpt-5.1-codex-max')
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE llm_model
        SET description = CASE id
            WHEN 'gemini-2.5-pro' THEN 'This is an advanced reasoning model, outperforming GPT-5 Mini with a larger context while being more affordable.'
            WHEN 'gpt-5-mini' THEN 'This is a new reasoning model with a good balance between cost and intelligence. Suited for most agent tasks.'
            WHEN 'gpt-5-nano' THEN 'This is a new reasoning model for simpler tasks. Lower cost than GPT-5 Mini.'
            WHEN 'claude-sonnet-4-6' THEN 'This is an improved version of Claude Sonnet 4 with better reasoning. Good for complex analysis, coding, and creative writing. Similar pricing to Claude Sonnet 4.'
            WHEN 'claude-opus-4-6' THEN 'This is a more capable model than Claude Sonnet 4 for highly complex tasks. Best for demanding reasoning and analysis. Higher cost than Claude Sonnet models.'
            WHEN 'gpt-5.4' THEN 'This is the most capable reasoning model from OpenAI for professional work. Best for demanding analysis and complex agentic tasks. Higher cost than GPT-5.'
            WHEN 'gpt-5.1-codex-max' THEN 'This is the most capable Codex model for long-running agentic coding. Best for complex codebases and multi-step tasks. Similar pricing to GPT-5.1 Codex.'
        END
        WHERE id IN ('gemini-2.5-pro', 'gpt-5-mini', 'gpt-5-nano', 'claude-sonnet-4-6', 'claude-opus-4-6', 'gpt-5.4', 'gpt-5.1-codex-max')
    """)
