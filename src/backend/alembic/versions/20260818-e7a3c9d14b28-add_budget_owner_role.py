"""add_budget_owner_role

Revision ID: e7a3c9d14b28
Revises: 8a1f2c3d4e5f
Create Date: 2026-08-18

"""
from typing import Sequence, Union
from alembic import op
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = 'e7a3c9d14b28'
down_revision: Union[str, None] = '8a1f2c3d4e5f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.sync_enum_values(  # type: ignore
        enum_schema='public',
        enum_name='role',
        new_values=['TEAM_OWNER', 'TEAM_MEMBER', 'TEAM_EDITOR', 'BUDGET_OWNER'],
        affected_columns=[TableReference(table_schema='public', table_name='team_role', column_name='role')],
        enum_values_to_rename=[],
    )
    op.execute("UPDATE team_role SET role = 'BUDGET_OWNER' WHERE team_id = 1 AND user_id = 1 AND role = 'TEAM_OWNER'")


def downgrade() -> None:
    op.execute("UPDATE team_role SET role = 'TEAM_OWNER' WHERE role = 'BUDGET_OWNER'")
    op.sync_enum_values(  # type: ignore
        enum_schema='public',
        enum_name='role',
        new_values=['TEAM_OWNER', 'TEAM_MEMBER', 'TEAM_EDITOR'],
        affected_columns=[TableReference(table_schema='public', table_name='team_role', column_name='role')],
        enum_values_to_rename=[],
    )
