"""agent snapshots

Revision ID: 9d5bf2a1c780
Revises: 11ed51c75684
Create Date: 2026-09-22

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "9d5bf2a1c780"
down_revision: Union[str, None] = "11ed51c75684"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_snapshot",
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=30), nullable=True),
        sa.Column("icon", sa.LargeBinary(), nullable=True),
        sa.Column("last_update", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agent.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["team_id"], ["team.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("agent_id", "team_id"),
    )


def downgrade() -> None:
    op.drop_table("agent_snapshot")
