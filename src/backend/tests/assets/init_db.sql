insert into "user" (username, name, monthly_usd_limit, monthly_hours, created_at, deleted_at) values
('test', 'John Doe', 10, 160, '2025-01-20 12:00', Null),
('test2', 'Jane Doe', 20, 160, '2025-01-20 12:00', Null),
('test3', 'John Doe 3', 10, 160, '2025-01-20 12:00', '2025-02-21 12:00'),
('test4', 'John Doe 4', 10, 160, '2025-01-10 12:00', '2025-01-20 12:00'),
('test5', 'John Doe 5', 10, 160, '2025-01-10 12:00', Null);

insert into llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd) values
('claude-sonnet-4', 'Claude Sonnet 4', 'CHAT', 'This is a similar model to GPT-5 but with different strengths.', 'ANTHROPIC', 200000, 64000, 0.003, 0.015),
('claude-sonnet-4-6', 'Claude Sonnet 4.6', 'CHAT', 'This is an improved version of Claude Sonnet 4 with better reasoning. Good for complex analysis, coding, and creative writing. Similar pricing to Claude Sonnet 4.', 'ANTHROPIC', 200000, 64000, 0.003, 0.015),
('claude-opus-4-6', 'Claude Opus 4.6', 'CHAT', 'This is a more capable model than Claude Sonnet 4 for highly complex tasks. Best for demanding reasoning and analysis. Higher cost than Claude Sonnet models.', 'ANTHROPIC', 200000, 64000, 0.005, 0.025),
('gemini-2.5-pro', 'Gemini 2.5 Pro', 'CHAT', 'This is an advanced reasoning model, outperforming GPT-5 Mini with a larger context while being more affordable.', 'GOOGLE', 1048576, 65536, 0.00125, 0.01),
('gemini-2.5-flash', 'Gemini 2.5 Flash', 'CHAT', 'This is a fast and efficient model, comparable to GPT-5 Nano, optimized for speed while maintaining high quality responses.', 'GOOGLE', 1048576, 65536, 0.0003, 0.0025),
('gpt-5', 'GPT-5', 'REASONING', 'This is the best reasoning model for coding and complex agentic tasks from OpenAI. Aligned with GPT-5 Mini for most use cases.', 'OPENAI', 400000, 128000, 0.00125, 0.01),
('gpt-5-mini', 'GPT-5 Mini', 'REASONING', 'This is a new reasoning model with a good balance between cost and intelligence. Suited for most agent tasks.', 'OPENAI', 400000, 128000, 0.00025, 0.002),
('gpt-5-nano', 'GPT-5 Nano', 'REASONING', 'This is a new reasoning model for simpler tasks. Lower cost than GPT-5 Mini.', 'OPENAI', 4000000, 128000, 0.00005, 0.0004),
('gpt-5.1-codex-max', 'GPT-5.1 Codex Max', 'REASONING', 'This is the most capable Codex model for long-running agentic coding. Best for complex codebases and multi-step tasks. Similar pricing to GPT-5.1 Codex.', 'OPENAI', 400000, 128000, 0.00125, 0.01),
('gpt-5.4', 'GPT-5.4', 'REASONING', 'This is the most capable reasoning model from OpenAI for professional work. Best for demanding analysis and complex agentic tasks. Higher cost than GPT-5.', 'OPENAI', 1050000, 128000, 0.0025, 0.015),
('gpt-5.4-mini', 'GPT-5.4 Mini', 'REASONING', 'This is a fast reasoning model with strong performance. Suited for most agent tasks like research, drafting, and data analysis.', 'OPENAI', 400000, 128000, 0.00075, 0.0045),
('claude-haiku-4-5', 'Claude Haiku 4.5', 'CHAT', 'This is a fast model suited for straightforward tasks. Good for summarization, extraction, and other tasks that don''t require complex reasoning.', 'ANTHROPIC', 200000, 64000, 0.001, 0.005);


insert into team (name) values
('Test Team'),
('Another Team'),
('Third Team'),
('Fourth Team');

insert into team_role (team_id, user_id, role, status) values
(1, 1, 'TEAM_OWNER', 'ACCEPTED'),
(2, 1, 'TEAM_OWNER',  'ACCEPTED'),
(1, 2, 'TEAM_MEMBER', 'ACCEPTED'),
(2, 2, 'TEAM_OWNER', 'ACCEPTED'),
(4, 1, 'TEAM_MEMBER', 'ACCEPTED'),
(4, 2, 'TEAM_MEMBER', 'PENDING'),
(4, 5, 'TEAM_MEMBER', 'REJECTED');

insert into agent (name, description, user_id, last_update, team_id, model_id, system_prompt, temperature, reasoning_effort, recursion_limit, is_protected, agent_type) values
('Agent 1', 'This is the first agent', 1, '2025-02-21 12:00', Null, 'gpt-5-mini', 'You are a helpful AI agent.', 'PRECISE', 'LOW', 20, false, 'REACT_AGENT'),
('Agent 2', 'This is the second agent', 1, '2025-02-21 12:01', 1, 'gpt-5-mini', 'You are a helpful AI agent.', 'CREATIVE', 'LOW', 20, false, 'DEEP_AGENT'),
('Agent 3', 'This is the third agent', 2, '2025-02-21 12:02', 4, 'gpt-5', 'You are a helpful AI agent.', 'PRECISE', 'LOW', 20, true, 'REACT_AGENT'),
('Agent 4', 'This is the fourth agent', 2, '2025-02-21 12:03', Null, 'gpt-5', 'You are a helpful AI agent.', 'CREATIVE', 'LOW', 20, false, 'REACT_AGENT'),
('Agent 5', 'This is the fifth agent', 2, '2025-02-21 12:04', 2, 'gpt-5', 'You are a helpful AI agent.', 'PRECISE', 'LOW', 20, false, 'REACT_AGENT'),
('GPT-5 Nano', 'This is the default agent', Null, '2025-02-21 12:00', 1, 'gpt-5-nano', 'You are a helpful AI agent.', 'NEUTRAL', 'LOW', 20, false, 'REACT_AGENT');

insert into user_agent (user_id, agent_id, creation) values
(1, 1, '2025-02-21 12:00'),
(1, 2, '2025-02-21 12:01'),
(2, 2, '2025-02-21 12:02'),
(2, 4, '2025-02-21 12:03'),
(2, 5, '2025-02-21 12:04');

insert into agent_tool_config (agent_id, tool_id, config, draft) values
(4, 'docs', '{}', false);

insert into file (name, status, content_type, user_id, timestamp, content, processed_content, file_processor) values
('test.txt', 'PROCESSED','text/plain', 2, '2025-02-21 12:00', '\x48656c6c6f21', 'Hello!', 'BASIC');

insert into agent_tool_config_file (agent_id, tool_id, file_id) values
(4, 'docs', 1);

insert into thread (name, user_id, agent_id, creation, deleted, is_test_case) values
('Thread 1', 1, 1, '2025-02-21 12:00', False, False),
('Thread 2', 1, 2, '2025-02-21 12:01', False, False),
('Thread 3', 2, 2, '2025-02-21 12:02', False, False),
('Thread 4', 2, 2, '2025-02-21 12:03', False, False),
('Thread 5', 3, 3, '2025-02-21 12:04', False, False),
('Thread 6', 2, 5, '2025-02-21 12:05', False, False),
('Test Case #1', 1, 1, '2025-02-21 12:10', False, True),
('Test Case #2', 1, 1, '2025-02-21 12:11', False, True),
('Test Case #3', 2, 2, '2025-02-21 12:12', False, True),
('Test Case Execution #1', 1, 1, '2025-02-21 12:15', False, True),
('Test Case Execution #2', 1, 1, '2025-02-21 12:16', False, True),
('Test Case Execution #3', 2, 2, '2025-02-21 12:17', False, True),
('Test Case #4', 1, 1, '2025-02-21 12:13', False, True),
('Test Case For Evaluation #5', 1, 1, '2025-02-21 12:14', False, True);

insert into thread_message (thread_id, origin, text, timestamp, minutes_saved, stopped, parent_id) values
(1, 'USER', 'This is a message', '2025-02-21 12:00', 5, False, Null),
(1, 'AGENT', 'This is a response', '2025-02-21 12:01', 30, False, 1),
(2, 'USER', 'This is another message', '2025-02-21 12:02', 30, False, Null),
(2, 'AGENT', 'This is a response', '2025-02-21 12:03', 5, False, 3),
(4, 'USER', 'This is a message', '2025-01-20 12:04', Null, False, Null),
(4, 'AGENT', 'This is a response', '2025-01-20 12:05', 60, False, 5),
(5, 'USER', 'This is a message', '2025-02-21 12:06', Null, False, Null),
(5, 'AGENT', 'This is a response', '2025-02-21 12:07', 50, False, 7),
(6, 'USER', 'This is a message', '2025-02-21 12:08', Null, False, Null),
(6, 'AGENT', 'This is a response', '2025-02-21 12:09', 60, False, 9),
(7, 'USER', 'Which is the first natural number? Only provide the number', '2025-02-21 12:10', Null, False, Null),
(7, 'AGENT', '1', '2025-02-21 12:11', Null, False, Null),
(8, 'USER', 'Which is the capital of Uruguay? Output just the name', '2025-02-21 12:11', Null, False, Null),
(8, 'AGENT', 'Montevideo', '2025-02-21 12:12', Null, False, Null),
(9, 'USER', 'Test case message 3', '2025-02-21 12:12', Null, False, Null),
(9, 'AGENT', 'Test case response 3', '2025-02-21 12:13', Null, False, Null),
(10, 'USER', 'Which is the first natural number? Only provide the number', '2025-02-21 12:15', Null, False, Null),
(10, 'AGENT', '1', '2025-02-21 12:16', Null, False, Null),
(11, 'USER', 'Which is the capital of Uruguay? Output just the name', '2025-02-21 12:16', Null, False, Null),
(11, 'AGENT', 'Montevideo', '2025-02-21 12:17', Null, False, Null),
(12, 'USER', 'Test case message 3', '2025-02-21 12:17', Null, False, Null),
(12, 'AGENT', 'Test case execution response 3', '2025-02-21 12:18', Null, False, Null),
(13, 'USER', 'What is 2 + 2? Only provide the number', '2025-02-21 12:13', Null, False, Null),
(13, 'AGENT', '4', '2025-02-21 12:14', Null, False, Null),
(13, 'USER', 'What is 3 + 3? Only provide the number', '2025-02-21 12:14', Null, False, Null),
(13, 'AGENT', '6', '2025-02-21 12:15', Null, False, Null),
(14, 'USER', 'Which is the first natural number? Only provide the number', '2025-02-21 12:15', Null, False, Null),
(14, 'AGENT', 'one', '2025-02-21 12:16', Null, False, Null);

insert into test_case (thread_id, agent_id, last_update) values
(7, 1, '2025-02-21 12:10'),
(8, 1, '2025-02-21 12:11'),
(9, 2, '2025-02-21 12:12'),
(13, 1, '2025-02-21 12:15'),
(14, 1, '2025-02-21 12:16');

insert into test_suite_run (agent_id, status, executed_at, completed_at, total_tests, passed_tests, failed_tests, error_tests, skipped_tests) values
(1, 'SUCCESS', '2025-02-21 12:15', '2025-02-21 12:17', 2, 2, 0, 0, 0),
(2, 'FAILURE', '2025-02-21 12:17', '2025-02-21 12:18', 1, 0, 0, 1, 0);

insert into test_case_result (thread_id, test_case_id, test_suite_run_id, status, executed_at, test_case_name) values
(10, 7, 1, 'SUCCESS', '2025-02-21 12:15', 'Test Case #1'),
(11, 8, 1, 'SUCCESS', '2025-02-21 12:16', 'Test Case #2'),
(12, 9, 2, 'ERROR', '2025-02-21 12:17', 'Test Case #3');

insert into usage (message_id, user_id, agent_id, model_id, timestamp, quantity, usd_cost, type) values
(2, 1, 1, 'gpt-5-mini', '2025-02-21 12:00', 100, 0.5, 'PROMPT_TOKENS'),
(2, 1, 1, 'gpt-5-mini', '2025-02-21 12:00', 200, 3.0, 'COMPLETION_TOKENS'),
(4, 1, 2, 'gpt-5', '2025-02-21 12:01', 50, 0.25, 'PROMPT_TOKENS'),
(4, 1, 2, 'gpt-5', '2025-02-21 12:01', 100, 1.5, 'COMPLETION_TOKENS'),
(4, 1, 2, 'gpt-5', '2025-02-21 12:01', 50, 0.5, 'PDF_PARSING'),
(6, 1, 2, 'gpt-5', '2025-01-20 12:02', 50, 0.25, 'PROMPT_TOKENS'),
(6, 1, 2, 'gpt-5', '2025-01-20 12:02', 100, 1.5, 'COMPLETION_TOKENS'),
(8, 2, 3, 'gpt-5', '2025-02-21 12:08', 50, 0.25, 'PROMPT_TOKENS'),
(8, 2, 3, 'gpt-5', '2025-02-21 12:08', 100, 1.5, 'COMPLETION_TOKENS'),
(10, 2, 5, 'gpt-5', '2025-02-21 12:08', 55, 0.25, 'PROMPT_TOKENS'),
(10, 2, 5, 'gpt-5', '2025-02-21 12:08', 105, 1.5, 'COMPLETION_TOKENS'),
(14, 1, 1, 'gpt-5-mini', '2025-02-21 12:15', 50, 0.25, 'PROMPT_TOKENS'),
(14, 1, 1, 'gpt-5-mini', '2025-02-21 12:15', 100, 1.5, 'COMPLETION_TOKENS'),
(16, 1, 1, 'gpt-5-mini', '2025-02-21 12:16', 60, 0.3, 'PROMPT_TOKENS'),
(16, 1, 1, 'gpt-5-mini', '2025-02-21 12:16', 120, 1.8, 'COMPLETION_TOKENS'),
(18, 2, 2, 'gpt-5-mini', '2025-02-21 12:17', 40, 0.2, 'PROMPT_TOKENS'),
(18, 2, 2, 'gpt-5-mini', '2025-02-21 12:17', 80, 1.2, 'COMPLETION_TOKENS');

insert into agent_prompt (name, content, last_update, shared, agent_id, user_id, starter) values
('Test prompt private 1', 'Test prompt content', '2025-02-22 12:00', false, 1, 1, false),
('Test prompt shared', 'Test shared prompt content', '2025-02-22 12:00', true, 1, 2, false),
('Test prompt private 2', 'Test prompt content 2', '2025-02-22 12:00', false, 1, 2, false),
('Test prompt shared 2', 'Test shared prompt content 2', '2025-02-22 12:00', true, 3, 2, false);

insert into external_agent (name, icon) values
('ChatGPT', Null),
('Cursor', Null),
('Claude', Null);

insert into external_agent_time_saving (external_agent_id, user_id, date, minutes_saved) values
(1, 1, '2025-02-21 12:00', 60),
(2, 1, '2025-02-21 12:01', 120),
(3, 1, '2025-01-20 12:02', 60);

-- Triggers for test suite run events LISTEN/NOTIFY (from migration 20251124-c759e1cf2817)
CREATE OR REPLACE FUNCTION notify_test_suite_run_event() RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'test_suite_events',
        json_build_object(
            'suite_run_id', NEW.test_suite_run_id,
            'event_id', NEW.id
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_insert_test_suite_run_event
AFTER INSERT ON test_suite_run_event
FOR EACH ROW EXECUTE FUNCTION notify_test_suite_run_event();

CREATE OR REPLACE FUNCTION notify_test_suite_run_status() RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'test_suite_status',
        json_build_object(
            'suite_run_id', NEW.id,
            'status', NEW.status
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_update_test_suite_run_status
AFTER UPDATE OF status ON test_suite_run
FOR EACH ROW
WHEN (OLD.status IS DISTINCT FROM NEW.status)
EXECUTE FUNCTION notify_test_suite_run_status();
