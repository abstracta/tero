"""set-default-agents-reasoning-to-low

Revision ID: 342c58cae8b5
Revises: a76abde4b0bb
Create Date: 2026-05-20 21:40:14.262265

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '342c58cae8b5'
down_revision: Union[str, None] = 'a76abde4b0bb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE agent
        SET reasoning_effort = 'LOW'
        WHERE name IN ('GPT-5 Nano', 'GPT-5')
          AND user_id IS NULL
          AND team_id = 1
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE agent
        SET reasoning_effort = 'MEDIUM'
        WHERE name IN ('GPT-5 Nano', 'GPT-5')
          AND user_id IS NULL
          AND team_id = 1
    """)
