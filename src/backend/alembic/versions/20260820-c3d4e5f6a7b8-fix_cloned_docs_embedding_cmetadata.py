"""fix cloned docs embeddings pointing at the source agent's files

Revision ID: c3d4e5f6a7b8
Revises: e7a3c9d14b28
Create Date: 2026-08-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "e7a3c9d14b28"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    # PGVector and SQLRecordManager create these at tenant setup, not via Alembic.
    # Fresh DBs (or tenants that never used docs) have nothing to repair.
    if not connection.execute(sa.text("""
        SELECT to_regclass('langchain_pg_embedding') IS NOT NULL
           AND to_regclass('langchain_pg_collection') IS NOT NULL
           AND to_regclass('upsertion_record') IS NOT NULL
    """)).scalar():
        return

    # Old clone copied embeddings without updating cmetadata.id, so docs search
    # on the clone still referenced the source agent's file. Remap to this
    # agent's copy when name+content match.
    connection.execute(sa.text("""
        WITH mapped AS (
            SELECT DISTINCT ON (e.id)
                e.id AS embedding_id,
                own_file.id::text AS new_file_id
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            JOIN file AS referenced_file ON referenced_file.id::text = e.cmetadata->>'id'
            JOIN agent_tool_config_file AS own
              ON own.tool_id = 'docs'
             AND c.name = 'docs_' || own.agent_id
            JOIN file AS own_file
              ON own_file.id = own.file_id
             AND own_file.name = referenced_file.name
             AND own_file.content = referenced_file.content
            WHERE c.name LIKE 'docs_%'
              AND NOT EXISTS (
                  SELECT 1
                  FROM agent_tool_config_file atcf
                  WHERE atcf.tool_id = 'docs'
                    AND c.name = 'docs_' || atcf.agent_id
                    AND atcf.file_id::text = e.cmetadata->>'id'
              )
            ORDER BY e.id, own_file.id
        )
        UPDATE langchain_pg_embedding e
        SET cmetadata = jsonb_set(e.cmetadata, '{id}', to_jsonb(m.new_file_id))
        FROM mapped m
        WHERE e.id = m.embedding_id
    """))

    # Chunks that still don't belong to this agent (file removed, or no matching
    # clone copy) would keep surfacing in search. Drop them rather than leave orphans.
    connection.execute(sa.text("""
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE e.collection_id = c.uuid
          AND c.name LIKE 'docs_%'
          AND NOT EXISTS (
              SELECT 1
              FROM agent_tool_config_file atcf
              WHERE atcf.tool_id = 'docs'
                AND c.name = 'docs_' || atcf.agent_id
                AND atcf.file_id::text = e.cmetadata->>'id'
          )
    """))

    # Old clone copied upsertion_record with the source embedding ids, so they no
    # longer match the clone's embeddings. Drop those stale rows.
    connection.execute(sa.text("""
        DELETE FROM upsertion_record ur
        WHERE ur.namespace LIKE 'postgres/docs_%'
          AND NOT EXISTS (
              SELECT 1
              FROM langchain_pg_embedding e
              JOIN langchain_pg_collection c ON c.uuid = e.collection_id
              WHERE ur.namespace = 'postgres/' || c.name
                AND ur.key = e.id
          )
    """))

    # Rebuild upsertion_record from the embeddings we kept so it points at the
    # clone's ids. Agents that were never cloned already match.
    connection.execute(sa.text("""
        INSERT INTO upsertion_record (uuid, key, namespace, group_id, updated_at)
        SELECT gen_random_uuid()::varchar, e.id, 'postgres/' || c.name, e.cmetadata->>'id',
               EXTRACT(EPOCH FROM NOW())
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name LIKE 'docs_%'
          AND NOT EXISTS (
              SELECT 1 FROM upsertion_record ur
              WHERE ur.namespace = 'postgres/' || c.name AND ur.key = e.id
          )
    """))


def downgrade() -> None:
    pass
