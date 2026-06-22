"""add-gpt-5-4-mini-and-claude-haiku-4-5

Revision ID: e6f7a8b9c0d1
Revises: 342c58cae8b5
Create Date: 2026-05-15

"""

from typing import Sequence, Union
from alembic import op


revision: str = 'e6f7a8b9c0d1'
down_revision: Union[str, None] = '342c58cae8b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd)
        VALUES
        ('gpt-5.4-mini', 'GPT-5.4 Mini', 'REASONING', 'This is a fast reasoning model with strong performance. Suited for most agent tasks like research, drafting, and data analysis.', 'OPENAI', 400000, 128000, 0.00075, 0.0045),
        ('claude-haiku-4-5', 'Claude Haiku 4.5', 'CHAT', 'This is a fast model suited for straightforward tasks. Good for summarization, extraction, and other tasks that don''t require complex reasoning.', 'ANTHROPIC', 200000, 64000, 0.001, 0.005)
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id IN ('gpt-5.4-mini', 'claude-haiku-4-5')")
