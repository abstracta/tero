"""add exclude_from_metrics to user

Revision ID: 8a1f2c3d4e5f
Revises: b1c2d3e4f5a6
Create Date: 2026-08-13

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8a1f2c3d4e5f"
down_revision: Union[str, None] = "b1c2d3e4f5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('user', sa.Column('exclude_from_metrics', sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    op.drop_column('user', 'exclude_from_metrics')
