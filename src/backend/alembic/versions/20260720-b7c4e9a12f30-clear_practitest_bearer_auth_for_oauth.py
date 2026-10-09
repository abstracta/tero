"""clear practitest bearer auth for oauth migration

Revision ID: b7c4e9a12f30
Revises: 5a3f198afd24
Create Date: 2026-07-20 12:45:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "b7c4e9a12f30"
down_revision: Union[str, None] = "5a3f198afd24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DELETE FROM tool_auth_token WHERE tool_id = 'practitest'")
    op.execute("DELETE FROM tool_oauth_client_info WHERE tool_id = 'practitest'")
    op.execute("DELETE FROM tool_oauth_state WHERE tool_id = 'practitest'")
    op.execute("""
        UPDATE agent_tool_config
        SET config = (
            CASE
                WHEN config->>'serverUrl' LIKE 'https://api.practitest.com/mcp/v1/server%'
                    THEN jsonb_set(config::jsonb, '{serverUrl}', '"https://prod.practitest.com/mcp/v1/server"'::jsonb)
                WHEN config->>'serverUrl' LIKE 'https://eu1-prod-api.practitest.app/mcp/v1/server%'
                    THEN jsonb_set(config::jsonb, '{serverUrl}', '"https://eu1-prod.practitest.app/mcp/v1/server"'::jsonb)
                ELSE config::jsonb
            END - 'token'
        )::json
        WHERE tool_id = 'practitest'
    """)


def downgrade() -> None:
    pass
