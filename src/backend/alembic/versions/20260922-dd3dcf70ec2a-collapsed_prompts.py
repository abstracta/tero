"""collapsed prompts

Revision ID: dd3dcf70ec2a
Revises: c5d6e7f8a9b0
Create Date: 2026-09-22 18:02:21.456669

"""

import sqlalchemy as sa
import sqlmodel
from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'dd3dcf70ec2a'
down_revision: Union[str, None] = 'c5d6e7f8a9b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_prompt', sa.Column('collapsed', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('thread_message', sa.Column('collapsed_name', sqlmodel.AutoString(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column('thread_message', 'collapsed_name')
    op.drop_column('agent_prompt', 'collapsed')
