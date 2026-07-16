"""api_keys

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-05-29

"""

from typing import Sequence, Union

import sqlmodel
from alembic import op
import sqlalchemy as sa

revision: str = 'f7a8b9c0d1e2'
down_revision: Union[str, None] = 'e6f7a8b9c0d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('api_key',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sqlmodel.AutoString(length=100), nullable=False),
        sa.Column('key_id', sqlmodel.AutoString(length=32), nullable=False),
        sa.Column('hashed_secret', sqlmodel.AutoString(length=256), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_api_key_key_id'), 'api_key', ['key_id'], unique=True)
    op.create_index(op.f('ix_api_key_user_id'), 'api_key', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_api_key_key_id'), table_name='api_key')
    op.drop_index(op.f('ix_api_key_user_id'), table_name='api_key')
    op.drop_table('api_key')
