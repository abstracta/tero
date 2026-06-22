"""merge_claude_and_gpt_5_4_heads

Revision ID: 5f4c2a1b9d8e
Revises: 8cf95a46d129, c4d5e6f7a8b9
Create Date: 2026-04-13 13:15:00.000000

"""
from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = '5f4c2a1b9d8e'
down_revision: Union[str, Sequence[str], None] = ('8cf95a46d129', 'c4d5e6f7a8b9')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
