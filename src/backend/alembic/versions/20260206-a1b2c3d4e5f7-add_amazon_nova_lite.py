"""add-amazon-nova-lite

Revision ID: a1b2c3d4e5f7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-06

Adds Amazon Nova Lite 2 to llm_model. Configure AWS_MODEL_ID_MAPPING with:
  amazon-nova-lite:amazon.nova-2-lite-v1:0
"""
from typing import Sequence, Union

from alembic import op
from alembic_postgresql_enum import TableReference

from tero.core.env import env


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.sync_enum_values(  # type: ignore
        enum_schema="public",
        enum_name='llmmodelvendor',
        new_values=['AMAZON', 'ANTHROPIC', 'GOOGLE', 'OPENAI', 'QWEN'],
        affected_columns=[TableReference(table_schema="public", table_name='llm_model', column_name='model_vendor')],
        enum_values_to_rename=[],
    )
    op.execute("""
        INSERT INTO llm_model (id, name, model_type, model_vendor, description, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd) VALUES
        ('amazon-nova-lite', 'Amazon Nova Lite 2', 'REASONING', 'AMAZON', 'This is Amazon''s efficient reasoning model for everyday use. Good for summaries, simple questions, and general assistance; comparable to GPT-4o Mini with competitive pricing.', 1000000, 65536, 0.0003, 0.0025)
    """)
    if (not env.azure_endpoints or not env.azure_api_keys) and not env.openai_api_key:
        system_prompt = """You are a helpful AI assistant.
Use provided tools and information provided in context to answer user questions.
Avoid generating responses that are not based on tools or previous context.
Provide short, concise and correct answers.
Answer in the same language as the user.
Use markdown to format your responses. You can include code blocks, tables, plantuml diagrams code blocks, echarts configuration code blocks and any standard markdown format"""
        op.execute(f"""
INSERT INTO agent (name, description, last_update, model_id, system_prompt, temperature, reasoning_effort, icon, team_id)
VALUES (
'Nova Lite 2',
'I can help you with general questions and tasks by using Nova Lite 2.',
NOW(),
'amazon-nova-lite',
'{system_prompt}',
'NEUTRAL',
'MEDIUM',
NULL,
1)""")


def downgrade() -> None:
    op.execute("DELETE FROM agent WHERE model_id = 'amazon-nova-lite' AND user_id is NULL")
    op.execute("DELETE FROM llm_model WHERE id = 'amazon-nova-lite'")
    op.sync_enum_values(  # type: ignore
        enum_schema="public",
        enum_name='llmmodelvendor',
        new_values=['ANTHROPIC', 'GOOGLE', 'OPENAI', 'QWEN'],
        affected_columns=[TableReference(table_schema="public", table_name='llm_model', column_name='model_vendor')],
        enum_values_to_rename=[],
    )
