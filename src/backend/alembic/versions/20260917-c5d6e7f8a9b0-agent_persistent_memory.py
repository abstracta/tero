"""agent persistent memory and deep-agent defaults

Revision ID: c5d6e7f8a9b0
Revises: c3d9e5a71b48
Create Date: 2026-09-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c5d6e7f8a9b0"
down_revision: Union[str, None] = "c3d9e5a71b48"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent",
        sa.Column("memory_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.execute(
        """
        UPDATE agent
        SET agent_type = 'DEEP_AGENT', memory_enabled = true
        WHERE name IN ('GPT-5 Nano', 'GPT-5')
          AND user_id IS NULL
          AND team_id = 1
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE agent
        SET agent_type = 'REACT_AGENT'
        WHERE name IN ('GPT-5 Nano', 'GPT-5')
          AND user_id IS NULL
          AND team_id = 1
        """
    )
    op.drop_column("agent", "memory_enabled")
