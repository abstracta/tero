"""docs_file_rate_limit_error

Revision ID: 5a3f198afd24
Revises: f7a8b9c0d1e2
Create Date: 2026-07-01 10:46:25.434879

"""

import sqlalchemy as sa
import sqlmodel
from typing import Sequence, Union
from alembic import op
from alembic_postgresql_enum import TableReference

revision: str = '5a3f198afd24'
down_revision: Union[str, None] = 'f7a8b9c0d1e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.sync_enum_values(  # type: ignore
        enum_schema='public',
        enum_name='filestatus',
        new_values=['PENDING', 'PROCESSED', 'ERROR', 'QUOTA_EXCEEDED', 'RATE_LIMITED'],
        affected_columns=[TableReference(table_schema='public', table_name='file', column_name='status')],
        enum_values_to_rename=[],
    )


def downgrade() -> None:
    op.sync_enum_values(  # type: ignore
        enum_schema='public',
        enum_name='filestatus',
        new_values=['PENDING', 'PROCESSED', 'ERROR', 'QUOTA_EXCEEDED'],
        affected_columns=[TableReference(table_schema='public', table_name='file', column_name='status')],
        enum_values_to_rename=[],
    )
