"""add gpt 6 luna and sol

Revision ID: 11ed51c75684
Revises: dd3dcf70ec2a
Create Date: 2026-09-28 16:15:52.103921

"""

from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = '11ed51c75684'
down_revision: Union[str, None] = 'dd3dcf70ec2a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd)
        VALUES
        ('gpt-6-luna', 'GPT-6 Luna', 'REASONING', 'This is the fastest and most affordable GPT-6 model. Suited for everyday assistants, search, summarization, and other high-volume work.', 'OPENAI', 1050000, 128000, 0.0002, 0.00075),
        ('gpt-6-sol', 'GPT-6 Sol', 'REASONING', 'This is the flagship GPT-6 model for the most demanding reasoning and agentic workloads. Best for complex analysis, software engineering, and multi-step knowledge work.', 'OPENAI', 1050000, 128000, 0.004, 0.015)
    """)
    op.execute("""
        INSERT INTO agent (name, description, last_update, model_id, system_prompt, temperature, reasoning_effort, recursion_limit, team_id, icon, is_protected, agent_type, memory_enabled, evaluator_id)
        SELECT 'GPT-6 Luna', 'I can help you with simple questions and tasks by using GPT-6 Luna.', NOW(), 'gpt-6-luna', system_prompt, temperature, reasoning_effort, recursion_limit, team_id, icon, is_protected, agent_type, memory_enabled, evaluator_id
        FROM agent
        WHERE name = 'GPT-5 Nano' AND user_id IS NULL AND team_id = 1
    """)
    op.execute("""
        INSERT INTO agent (name, description, last_update, model_id, system_prompt, temperature, reasoning_effort, recursion_limit, team_id, icon, is_protected, agent_type, memory_enabled, evaluator_id)
        SELECT 'GPT-6 Sol', 'I can help you with general questions and tasks by using GPT-6 Sol.', NOW(), 'gpt-6-sol', system_prompt, temperature, reasoning_effort, recursion_limit, team_id, icon, is_protected, agent_type, memory_enabled, evaluator_id
        FROM agent
        WHERE name = 'GPT-5' AND user_id IS NULL AND team_id = 1
    """)
    op.execute("""
        UPDATE agent
        SET team_id = NULL
        WHERE name IN ('GPT-5 Nano', 'GPT-5')
          AND user_id IS NULL
          AND team_id = 1
    """)


def downgrade() -> None:
    op.execute("""
        DELETE FROM agent
        WHERE name IN ('GPT-6 Luna', 'GPT-6 Sol')
          AND user_id IS NULL
          AND team_id = 1
    """)
    op.execute("""
        UPDATE agent
        SET team_id = 1
        WHERE name IN ('GPT-5 Nano', 'GPT-5')
          AND user_id IS NULL
          AND team_id IS NULL
    """)
    op.execute("DELETE FROM llm_model WHERE id IN ('gpt-6-luna', 'gpt-6-sol')")
