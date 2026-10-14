"""gpt-long-context-pricing

Revision ID: a8b9c0d1e2f3
Revises: c1d2e3f4a5b6
Create Date: 2026-08-12

Update GPT-5.4 and GPT-5.6 Sol/Terra/Luna to Global long-context pricing
"""

from typing import Sequence, Union
from alembic import op


revision: str = 'a8b9c0d1e2f3'
down_revision: Union[str, None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.005, completion_1k_token_usd = 0.0225
        WHERE id = 'gpt-5.4'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.01, completion_1k_token_usd = 0.045
        WHERE id = 'gpt-5.6-sol'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.005, completion_1k_token_usd = 0.0225
        WHERE id = 'gpt-5.6-terra'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.002, completion_1k_token_usd = 0.009
        WHERE id = 'gpt-5.6-luna'
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.0025, completion_1k_token_usd = 0.015
        WHERE id = 'gpt-5.4'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.005, completion_1k_token_usd = 0.03
        WHERE id = 'gpt-5.6-sol'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.0025, completion_1k_token_usd = 0.015
        WHERE id = 'gpt-5.6-terra'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.001, completion_1k_token_usd = 0.006
        WHERE id = 'gpt-5.6-luna'
    """)
