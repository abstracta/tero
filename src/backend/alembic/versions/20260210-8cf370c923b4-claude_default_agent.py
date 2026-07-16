"""claude-default-agent

Revision ID: 8cf370c923b4
Revises: 613cc99427e2
Create Date: 2026-02-10 14:46:50.083309

"""
from typing import Sequence, Union
from alembic import op

from tero.core.env import env

# revision identifiers, used by Alembic.
revision: str = '8cf370c923b4'
down_revision: Union[str, None] = '613cc99427e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    if (not env.azure_endpoints or not env.azure_api_keys) and not env.openai_api_key:
        op.execute("UPDATE agent SET team_id = NULL WHERE user_id is NULL")
        system_prompt = """You are a helpful AI assistant.
Use provided tools and information provided in context to answer user questions.
Avoid generating responses that are not based on tools or previous context.
Provide short, concise and correct answers.
Answer in the same language as the user.
Use markdown to format your responses. You can include code blocks, tables, plantuml diagrams code blocks, echarts configuration code blocks and any standard markdown format"""
        op.execute(f"""
INSERT INTO agent (name, description, last_update, model_id, system_prompt, temperature, reasoning_effort, icon, team_id)
VALUES (
'Claude Sonnet 4',
'I can help you with general questions and tasks by using Claude Sonnet 4.',
NOW(),
'claude-sonnet-4',
'{system_prompt}',
'NEUTRAL',
'MEDIUM',
NULL,
1)""")


def downgrade() -> None:
    op.execute("DELETE FROM agent WHERE model_id = 'claude-sonnet-4' AND user_id is NULL")
    op.execute("UPDATE agent SET team_id = 1 WHERE user_id is NULL")
