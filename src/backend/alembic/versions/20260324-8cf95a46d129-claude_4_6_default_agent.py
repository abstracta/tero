"""claude_4.6_default_agent

Revision ID: 8cf95a46d129
Revises: a1b2c3d4e5f7
Create Date: 2026-03-24 20:04:24.227740

"""
import sqlalchemy as sa
import sqlmodel
from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = '8cf95a46d129'
down_revision: Union[str, None] = 'a1b2c3d4e5f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE agent SET model_id = 'claude-sonnet-4-6', name = 'Claude Sonnet 4.6' WHERE model_id = 'claude-sonnet-4' AND user_id is NULL
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE agent SET model_id = 'claude-sonnet-4', name = 'Claude Sonnet 4' WHERE model_id = 'claude-sonnet-4-6' AND user_id is NULL
    """)
