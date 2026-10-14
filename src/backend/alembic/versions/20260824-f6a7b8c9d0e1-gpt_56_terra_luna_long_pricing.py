"""gpt-5.6-terra-luna-long-pricing

Revision ID: f6a7b8c9d0e1
Revises: c3d4e5f6a7b8
Create Date: 2026-08-24

Update GPT-5.6 Terra and Luna to current Azure Global long-context pricing
"""

from typing import Sequence, Union
from alembic import op


revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.004, completion_1k_token_usd = 0.018
        WHERE id = 'gpt-5.6-terra'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.0004, completion_1k_token_usd = 0.0018
        WHERE id = 'gpt-5.6-luna'
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.005, completion_1k_token_usd = 0.0225
        WHERE id = 'gpt-5.6-terra'
    """)
    op.execute("""
        UPDATE llm_model SET prompt_1k_token_usd = 0.002, completion_1k_token_usd = 0.009
        WHERE id = 'gpt-5.6-luna'
    """)
