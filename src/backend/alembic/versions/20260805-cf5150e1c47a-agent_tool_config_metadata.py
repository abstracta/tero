"""agent_tool_metadata

Revision ID: cf5150e1c47a
Revises: c1d2e3f4a5b6
Create Date: 2026-08-05
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'cf5150e1c47a'
down_revision: Union[str, None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_tool_config', sa.Column('tool_metadata', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('agent_tool_config', 'tool_metadata')
