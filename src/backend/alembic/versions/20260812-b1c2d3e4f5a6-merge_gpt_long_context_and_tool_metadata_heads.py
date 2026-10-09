"""merge_gpt_long_context_and_tool_metadata_heads

Revision ID: b1c2d3e4f5a6
Revises: a8b9c0d1e2f3, cf5150e1c47a
Create Date: 2026-08-12

"""
from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = ('a8b9c0d1e2f3', 'cf5150e1c47a')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
