"""add-claude-5-and-gpt-5-6-models

Revision ID: c1d2e3f4a5b6
Revises: b7c4e9a12f30
Create Date: 2026-07-28

"""

from typing import Sequence, Union
from alembic import op


revision: str = 'c1d2e3f4a5b6'
down_revision: Union[str, None] = 'b7c4e9a12f30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd)
        VALUES
        ('claude-sonnet-5', 'Claude Sonnet 5', 'REASONING', 'This is an improved version of Claude Sonnet 4.6 with stronger reasoning. Good for complex analysis, coding, and creative writing.', 'ANTHROPIC', 1000000, 128000, 0.003, 0.015),
        ('claude-opus-5', 'Claude Opus 5', 'REASONING', 'This is a more capable model than Claude Sonnet 5 for highly complex tasks. Best for demanding reasoning and analysis. Higher cost than Claude Sonnet models.', 'ANTHROPIC', 1000000, 128000, 0.005, 0.025),
        ('gpt-5.4-nano', 'GPT-5.4 Nano', 'REASONING', 'This is the most affordable GPT-5.4-class model for simple high-volume tasks. Good for classification, extraction, and lightweight agent workflows.', 'OPENAI', 400000, 128000, 0.0002, 0.00125),
        ('gpt-5.6-sol', 'GPT-5.6 Sol', 'REASONING', 'This is the flagship GPT-5.6 model for the most demanding reasoning and agentic workloads. Best for complex analysis, software engineering, and multi-step knowledge work.', 'OPENAI', 1000000, 128000, 0.005, 0.03),
        ('gpt-5.6-terra', 'GPT-5.6 Terra', 'REASONING', 'This is a balanced GPT-5.6 model for everyday enterprise work. Good for scaling intelligent applications with strong reasoning at lower cost than GPT-5.6 Sol.', 'OPENAI', 1000000, 128000, 0.0025, 0.015),
        ('gpt-5.6-luna', 'GPT-5.6 Luna', 'REASONING', 'This is the fastest and most affordable GPT-5.6 model. Suited for high-volume, latency-sensitive workloads like assistants, search, and summarization.', 'OPENAI', 1000000, 128000, 0.001, 0.006)
    """)
    op.execute("""
        UPDATE agent
        SET model_id = 'claude-sonnet-5',
            name = REPLACE(name, 'Claude Sonnet 4', 'Claude Sonnet 5')
        WHERE model_id = 'claude-sonnet-4'
    """)
    op.execute("""
        UPDATE evaluator
        SET model_id = 'claude-sonnet-5'
        WHERE model_id = 'claude-sonnet-4'
    """)
    op.execute("DELETE FROM llm_model WHERE id = 'claude-sonnet-4'")


def downgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd)
        VALUES
        ('claude-sonnet-4', 'Claude Sonnet 4', 'CHAT', 'This is a similar model to GPT-5 but with different strengths.', 'ANTHROPIC', 200000, 64000, 0.003, 0.015)
    """)
    op.execute("""
        UPDATE agent
        SET model_id = 'claude-sonnet-4',
            name = REPLACE(name, 'Claude Sonnet 5', 'Claude Sonnet 4')
        WHERE model_id = 'claude-sonnet-5'
    """)
    op.execute("""
        UPDATE evaluator
        SET model_id = 'claude-sonnet-4'
        WHERE model_id = 'claude-sonnet-5'
    """)
    op.execute("DELETE FROM llm_model WHERE id IN ('claude-sonnet-5', 'claude-opus-5', 'gpt-5.4-nano', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna')")
