"""recursion-limit

Revision ID: fd24763a078c
Revises: 8cf370c923b4
Create Date: 2026-02-10 12:30:05.846201

"""

import sqlalchemy as sa
import sqlmodel
from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "fd24763a078c"
down_revision: Union[str, None] = "8cf370c923b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent",
        sa.Column(
            "recursion_limit",
            sa.Integer(),
            nullable=False,
            server_default=sa.literal_column("40"),
        ),
    )
    op.create_check_constraint(
        "agent_recursion_limit_range",
        "agent",
        "recursion_limit >= 20 AND recursion_limit <= 100",
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE agent DROP CONSTRAINT IF EXISTS agent_recursion_limit_range"
    )
    op.drop_column("agent", "recursion_limit")
