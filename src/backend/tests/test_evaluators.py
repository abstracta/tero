from .common import *

from .test_test_cases import (
    TestCaseExpectation,
    TestCaseStep,
    _assert_test_case_stream,
    _run_test_suite,
    _stream_test_suite_execution,
    TEST_CASE_1_THREAD_ID,
    TEST_CASE_2_THREAD_ID,
    TEST_CASE_4_THREAD_ID,
    TEST_CASE_5_THREAD_ID,
)

from tero.agents.evaluators.api import AGENT_EVALUATOR_PATH, TEST_CASE_EVALUATOR_PATH
from tero.agents.evaluators.domain import PublicEvaluator
from tero.agents.test_cases.api import TEST_CASES_PATH
from tero.agents.test_cases.domain import TestCaseResultStatus, TestSuiteRun, TestSuiteRunStatus
from tero.ai_models.domain import LlmTemperature, ReasoningEffort


EVALUATOR_MODEL_ID = "gpt-5-mini"
EVALUATOR_TEMPERATURE = LlmTemperature.NEUTRAL
EVALUATOR_REASONING_EFFORT = ReasoningEffort.MEDIUM


async def test_test_case_evaluator_inherits_from_agent(client: AsyncClient):
    new_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="Test evaluator prompt"
    )
    resp = await _update_agent_evaluator(client, AGENT_ID, new_evaluator)
    _assert_evaluator_response_matches(resp, new_evaluator)

    test_case_id = await _add_test_case(client, AGENT_ID)

    resp = await _get_test_case_evaluator(client, AGENT_ID, test_case_id)
    _assert_evaluator_response_matches(resp, new_evaluator)


async def test_update_test_case_evaluator(client: AsyncClient):
    agent_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="Agent evaluator prompt"
    )
    resp = await _update_agent_evaluator(client, AGENT_ID, agent_evaluator)
    _assert_evaluator_response_matches(resp, agent_evaluator)

    test_case_id = await _add_test_case(client, AGENT_ID)

    test_case_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="Test case evaluator prompt"
    )
    resp = await _update_test_case_evaluator(client, AGENT_ID, test_case_id, test_case_evaluator)
    _assert_evaluator_response_matches(resp, test_case_evaluator)

    resp = await _get_test_case_evaluator(client, AGENT_ID, test_case_id)
    _assert_evaluator_response_matches(resp, test_case_evaluator)


async def test_test_case_with_own_evaluator_doesnt_inherit_agent_updates(client: AsyncClient):
    initial_agent_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="Initial agent evaluator"
    )
    resp = await _update_agent_evaluator(client, AGENT_ID, initial_agent_evaluator)
    _assert_evaluator_response_matches(resp, initial_agent_evaluator)

    test_case_id = await _add_test_case(client, AGENT_ID)

    test_case_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="Test case evaluator"
    )
    resp = await _update_test_case_evaluator(client, AGENT_ID, test_case_id, test_case_evaluator)
    _assert_evaluator_response_matches(resp, test_case_evaluator)

    updated_agent_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="Updated agent evaluator"
    )
    resp = await _update_agent_evaluator(client, AGENT_ID, updated_agent_evaluator)
    _assert_evaluator_response_matches(resp, updated_agent_evaluator)

    resp = await _get_test_case_evaluator(client, AGENT_ID, test_case_id)
    _assert_evaluator_response_matches(resp, test_case_evaluator)


@freeze_time(CURRENT_TIME)
async def test_run_test_case_with_specific_evaluator(client: AsyncClient, last_message_id: int, last_suite_run_id: int, last_result_id: int):
    expected_input = "Which is the first natural number? Only provide the number. Don't ask follow up questions"
    expected_response_chunks = ["1"]
    expected_suite_run_id = last_suite_run_id + 1
    results_per_run = 4
    expected_result_id = last_result_id + results_per_run
    request_body = {"test_case_ids": [TEST_CASE_5_THREAD_ID]}

    test_case_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="""
            Treat numerically equivalent answers as correct, even when written in different formats.
            Consider numeric words and digit forms as equivalent when they represent the same value.
        """
    )
    resp = await _update_test_case_evaluator(client, AGENT_ID, TEST_CASE_5_THREAD_ID, test_case_evaluator)
    _assert_evaluator_response_matches(resp, test_case_evaluator)

    resp = await _run_test_suite(client, AGENT_ID, request_body)
    assert_response(resp, TestSuiteRun(
        id=expected_suite_run_id,
        agent_id=AGENT_ID,
        status=TestSuiteRunStatus.RUNNING,
        executed_at=CURRENT_TIME,
        completed_at=None,
        total_tests=4,
        passed_tests=0,
        failed_tests=0,
        error_tests=0,
        skipped_tests=0
    ))

    async with _stream_test_suite_execution(client, AGENT_ID, expected_suite_run_id) as resp:
        resp.raise_for_status()
        await _assert_test_case_stream(
            resp,
            expected_suite_run_id,
            [
                TestCaseExpectation(
                    test_case_id=TEST_CASE_5_THREAD_ID,
                    result_id=expected_result_id,
                    status=TestCaseResultStatus.SUCCESS,
                    steps=[
                        TestCaseStep(
                            input=expected_input,
                            response_chunks=expected_response_chunks,
                            user_message_id=last_message_id + 1,
                            agent_message_id=last_message_id + 2
                        )
                    ]
                )
            ],
            skipped_test_case_ids=[TEST_CASE_1_THREAD_ID, TEST_CASE_2_THREAD_ID, TEST_CASE_4_THREAD_ID]
        )

    test_case_evaluator = PublicEvaluator(
        model_id=EVALUATOR_MODEL_ID,
        temperature=EVALUATOR_TEMPERATURE,
        reasoning_effort=EVALUATOR_REASONING_EFFORT,
        prompt="""
            Evaluate using strict exact-match criteria.

            Rules:
            - Compare the reference output and actual output literally.
            - Do not apply semantic equivalence or normalization.
            - Numeric words and digit forms are different values.
            - Return true only when both outputs are exactly the same text.

            Provide a short explanation.
        """
    )
    resp = await _update_test_case_evaluator(client, AGENT_ID, TEST_CASE_5_THREAD_ID, test_case_evaluator)
    _assert_evaluator_response_matches(resp, test_case_evaluator)

    resp = await _run_test_suite(client, AGENT_ID, request_body)
    assert_response(resp, TestSuiteRun(
        id=expected_suite_run_id + 1,
        agent_id=AGENT_ID,
        status=TestSuiteRunStatus.RUNNING,
        executed_at=CURRENT_TIME,
        completed_at=None,
        total_tests=4,
        passed_tests=0,
        failed_tests=0,
        error_tests=0,
        skipped_tests=0
    ))

    async with _stream_test_suite_execution(client, AGENT_ID, expected_suite_run_id + 1) as resp:
        resp.raise_for_status()
        await _assert_test_case_stream(
            resp,
            expected_suite_run_id + 1,
            [
                TestCaseExpectation(
                    test_case_id=TEST_CASE_5_THREAD_ID,
                    result_id=expected_result_id + results_per_run,
                    status=TestCaseResultStatus.FAILURE,
                    steps=[
                        TestCaseStep(
                            input=expected_input,
                            response_chunks=expected_response_chunks,
                            user_message_id=last_message_id + 3,
                            agent_message_id=last_message_id + 4
                        )
                    ]
                )
            ],
            skipped_test_case_ids=[TEST_CASE_1_THREAD_ID, TEST_CASE_2_THREAD_ID, TEST_CASE_4_THREAD_ID]
        )


async def _update_agent_evaluator(client: AsyncClient, agent_id: int, config: PublicEvaluator) -> Response:
    return await client.put(
        AGENT_EVALUATOR_PATH.format(agent_id=agent_id),
        json={
            "modelId": config.model_id,
            "temperature": config.temperature.value,
            "reasoningEffort": config.reasoning_effort.value,
            "prompt": config.prompt
        }
    )


async def _add_test_case(client: AsyncClient, agent_id: int) -> int:
    resp = await client.post(TEST_CASES_PATH.format(agent_id=agent_id))
    resp.raise_for_status()
    return resp.json()["thread"]["id"]


async def _get_test_case_evaluator(client: AsyncClient, agent_id: int, test_case_id: int) -> Response:
    return await client.get(TEST_CASE_EVALUATOR_PATH.format(agent_id=agent_id, test_case_id=test_case_id))


async def _update_test_case_evaluator(client: AsyncClient, agent_id: int, test_case_id: int, config: PublicEvaluator) -> Response:
    return await client.put(
        TEST_CASE_EVALUATOR_PATH.format(agent_id=agent_id, test_case_id=test_case_id),
        json={
            "modelId": config.model_id,
            "temperature": config.temperature.value,
            "reasoningEffort": config.reasoning_effort.value,
            "prompt": config.prompt
        }
    )


def _assert_evaluator_response_matches(resp: Response, expected: PublicEvaluator) -> None:
    assert resp.status_code == 200
    resp_data = resp.json()
    assert resp_data["modelId"] == expected.model_id
    assert resp_data["temperature"] == expected.temperature.value
    assert resp_data["reasoningEffort"] == expected.reasoning_effort.value
    assert resp_data["prompt"] == expected.prompt
