"""deepagents

Revision ID: a76abde4b0bb
Revises: e1f2a3b4c5d6
Create Date: 2026-04-29 10:59:44.035216

"""

import sqlalchemy as sa
from typing import Sequence, Union
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a76abde4b0bb'
down_revision: Union[str, None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    sa.Enum('DEEP_AGENT', 'REACT_AGENT', name='agenttype').create(op.get_bind())
    op.add_column('agent', sa.Column('agent_type', postgresql.ENUM('DEEP_AGENT', 'REACT_AGENT', name='agenttype', create_type=False), nullable=False, server_default='REACT_AGENT'))


def downgrade() -> None:
    op.drop_column('agent', 'agent_type')
    sa.Enum('DEEP_AGENT', 'REACT_AGENT', name='agenttype').drop(op.get_bind())
