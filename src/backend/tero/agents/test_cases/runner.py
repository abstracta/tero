import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import List, cast, Any, AsyncIterator, Tuple, Optional, Dict, Coroutine

from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.store.base import BaseStore
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from openevals.llm import create_async_llm_as_judge
from openevals.types import EvaluatorResult
from sqlmodel.ext.asyncio.session import AsyncSession

from ...core import repos as repos_module
from ...core.env import env
from ...agents.evaluators.domain import Evaluator
from ...agents.evaluators.repos import EvaluatorRepository
from ...ai_models import ai_factory
from ...ai_models.domain import LlmModel, LlmModelType, LlmTemperature, ReasoningEffort
from ...ai_models.repos import AiModelRepository
from ...threads.domain import ThreadMessage, Thread, ThreadMessageOrigin, AgentMessageEvent, AgentActionEvent
from ...threads.engine import AgentEngine
from ...threads.repos import ThreadMessageRepository, ThreadRepository
from ...usage.domain import MessageUsage
from ...usage.repos import UsageRepository
from ..domain import Agent
from ..repos import AgentRepository
from .domain import TestCase, TestCaseResultStatus, TestCaseEventType, TestCaseResult, TestSuiteRun, TestSuiteEventType, TestSuiteRunStatus, TestSuiteRunEvent
from .repos import TestCaseResultRepository, TestSuiteRunRepository, TestSuiteRunEventRepository, TestCaseRepository


logger = logging.getLogger(__name__)
EVALUATOR_DEFAULT_INSTRUCTIONS = """
Compare the actual output with the reference output based on these criteria:
1. Semantic equivalence - Does the actual output convey the same meaning as the reference output?
2. Completeness - Does the actual output contain all key information from the reference output?
3. Accuracy - Is the actual output factually correct when compared to the reference output?
4. Relevance - Does the actual output appropriately address the input?
5. Conciseness - Does the actual output avoid including extra information not present in the reference output? If the reference output is concise the response should also be concise for example if the reference output is "Agent response" the actual output should also be "Agent response" or similar.

Be lenient with minor differences in wording, formatting, or style. Focus on whether the core meaning and key information match. Be strict about factual errors, missing critical information, or extraneous details that go beyond the expected output.
""".strip()
EVALUATOR_APPENDED_BLOCK = """
Respond with 'Y' if the actual output sufficiently matches the reference output, or 'N' if there are significant discrepancies. Then provide a brief explanation.

Input:
{{inputs}}

Reference Output:
{{reference_outputs}}

Actual Output:
{{outputs}}
""".strip()
EVALUATOR_DEFAULT_TEMPERATURE = LlmTemperature.NEUTRAL
EVALUATOR_DEFAULT_REASONING_EFFORT = ReasoningEffort.MEDIUM


class EvaluatorUsageTrackingCallback(AsyncCallbackHandler):
    def __init__(self, usage: MessageUsage, model: LlmModel):
        self.usage = usage
        self.model = model

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
         # Try to extract usage from llm_output first (most common format)
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage')
            if token_usage:
                self.usage.increment_with_metadata(UsageMetadata(
                    input_tokens=token_usage.get('prompt_tokens', 0),
                    output_tokens=token_usage.get('completion_tokens', 0),
                    total_tokens=token_usage.get('total_tokens', 0)
                ), self.model)
                return

        # Fallback: try to extract from generation_info
        if response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, 'generation_info') and generation.generation_info:
                        token_usage = generation.generation_info.get('token_usage')
                        if token_usage:
                            self.usage.increment_with_metadata(UsageMetadata(
                                input_tokens=token_usage.get('prompt_tokens', 0),
                                output_tokens=token_usage.get('completion_tokens', 0),
                                total_tokens=token_usage.get('total_tokens', 0)
                            ), self.model)
                            return


class BackgroundTestSuiteRunner:

    async def _broadcast_event(self, event_db: AsyncSession, suite_run_id: int, event_type: str, data: Dict[str, Any]) -> None:
        repo = TestSuiteRunEventRepository(event_db)
        await repo.add(TestSuiteRunEvent(
            test_suite_run_id=suite_run_id,
            type=event_type,
            data=json.dumps(data)
        ))

    def _build_test_case_evaluator_prompt(self, evaluator: Optional[Evaluator]) -> ChatPromptTemplate:
        instructions = evaluator.prompt if evaluator else EVALUATOR_DEFAULT_INSTRUCTIONS
        human_message = instructions.rstrip() + "\n\n" + EVALUATOR_APPENDED_BLOCK
        return ChatPromptTemplate(
            [("system", "You are an expert evaluator assessing whether the actual output from an AI agent matches the expected output for a given test case."),
                ("human", human_message)],
            template_format="mustache"
        )

    async def _mark_as_skipped(self, result: TestCaseResult, repo: TestCaseResultRepository) -> Tuple[TestCaseEventType, Dict[str, Any]]:
        result.status = TestCaseResultStatus.SKIPPED
        result.evaluator_analysis = None
        await repo.save(result)
        return (TestCaseEventType.PHASE, {
            "phase": "completed",
            "status": TestCaseResultStatus.SKIPPED.value,
            "evaluation": None
        })

    async def _run_test_case_stream(self, db: AsyncSession, test_case: TestCase, agent: Agent, user_id: int, result: TestCaseResult, stop_event: asyncio.Event, store: BaseStore) -> AsyncIterator[Tuple[TestCaseEventType, Any]]:
        results_repo = TestCaseResultRepository(db)
        try:
            thread_message_repo = ThreadMessageRepository(db)
            messages = await thread_message_repo.find_by_thread_id(test_case.thread_id)

            if not messages:
                yield (TestCaseEventType.METADATA, {
                    "testCaseId": test_case.thread_id,
                    "resultId": result.id,
                })
                yield await self._mark_as_skipped(result, results_repo)
                return

            execution_thread = await ThreadRepository(db).add(Thread(
                agent_id=agent.id,
                user_id=user_id,
                is_test_case=True
            ))

            result.thread_id = execution_thread.id
            result.status = TestCaseResultStatus.RUNNING
            result = await results_repo.save(result)

            yield (TestCaseEventType.METADATA, {
                "testCaseId": test_case.thread_id,
                "resultId": result.id,
            })


            for i in range(0, len(messages), 2):
                user_input = messages[i].text
                expected_output = messages[i + 1].text if len(messages) > i + 1 else ""
                execution_messages = await thread_message_repo.find_by_thread_id(execution_thread.id)

                yield (TestCaseEventType.PHASE, {"phase": "executing"})

                actual_output = ""

                async for event_type, content in self._execute_agent_with_input_stream(
                    db, agent, user_input, user_id, execution_thread.id, execution_messages, stop_event, store
                ):
                    if event_type == TestCaseEventType.AGENT_MESSAGE_COMPLETE:
                        actual_output = content["text"]
                        yield (event_type, content)
                    elif event_type in [TestCaseEventType.USER_MESSAGE, TestCaseEventType.AGENT_MESSAGE_START,
                                        TestCaseEventType.AGENT_MESSAGE_CHUNK, TestCaseEventType.EXECUTION_STATUS]:
                        yield (event_type, content)

                if stop_event.is_set():
                    yield await self._mark_as_skipped(result, results_repo)
                    return

                yield (TestCaseEventType.PHASE, {"phase": "evaluating"})

                evaluation_result = await self._evaluate_test_case_result(
                    db, user_input, expected_output, actual_output, user_id, agent, test_case, stop_event
                )

                if evaluation_result is None:
                    yield await self._mark_as_skipped(result, results_repo)
                    return

                evaluation_result_status = cast(bool, evaluation_result.get("score"))
                evaluation_result_analysis = (evaluation_result.get("comment") or "").replace("Thus, the score should be: SCORE_YOU_ASSIGN.", "")

                if not evaluation_result_status:
                    result.status = TestCaseResultStatus.FAILURE
                    result.evaluator_analysis = evaluation_result_analysis
                    await results_repo.save(result)

                    yield (TestCaseEventType.PHASE, {
                        "phase": "completed",
                        "status": TestCaseResultStatus.FAILURE.value,
                        "evaluation": {
                            "passed": False,
                        }
                    })
                    return

            result.status = TestCaseResultStatus.SUCCESS
            result.evaluator_analysis = None
            await results_repo.save(result)

            yield (TestCaseEventType.PHASE, {
                "phase": "completed",
                "status": TestCaseResultStatus.SUCCESS.value,
                "evaluation": {
                    "passed": True,
                }
            })

        except Exception as exc:
            error_code = None
            if isinstance(exc, GraphRecursionError) or (
                isinstance(exc, BaseExceptionGroup) and exc.subgroup(GraphRecursionError) is not None
            ):
                error_code = "RECURSION_LIMIT_EXCEEDED"
            else:
                logger.exception(f"Unexpected error running test case {test_case.thread_id}")
            try:
                result.status = TestCaseResultStatus.ERROR
                result.error_code = error_code
                await results_repo.save(result)
            except Exception:
                pass
            yield (TestCaseEventType.PHASE, {
                "phase": "completed",
                "status": TestCaseResultStatus.ERROR.value,
                "errorCode": error_code,
                "evaluation": None
            })

    async def _execute_agent_with_input_stream(self, db: AsyncSession, agent: Agent, user_input: str, user_id: int, thread_id: int, previous_messages: List[ThreadMessage], stop_event: asyncio.Event, store: BaseStore) -> AsyncIterator[Tuple[TestCaseEventType, Any]]:
        thread_message_repo = ThreadMessageRepository(db)
        input_message = await thread_message_repo.add(ThreadMessage(
            text=user_input,
            origin=ThreadMessageOrigin.USER,
            timestamp=datetime.now(timezone.utc),
            thread_id=thread_id
        ))

        yield (TestCaseEventType.USER_MESSAGE, {
            "id": input_message.id,
            "text": input_message.text,
        })

        input_message_usage = MessageUsage(user_id=user_id, agent_id=agent.id, model_id=agent.model_id, message_id=input_message.id)
        engine = AgentEngine(agent, user_id, db, store)

        response_message = await thread_message_repo.add(ThreadMessage(
            text="",
            origin=ThreadMessageOrigin.AGENT,
            timestamp=datetime.now(timezone.utc),
            thread_id=thread_id,
            parent_id=input_message.id
        ))

        yield (TestCaseEventType.AGENT_MESSAGE_START, {
            "id": response_message.id
        })

        messages_for_engine = previous_messages + [input_message]
        status_updates: List[AgentActionEvent] = []
        complete_response = ""
        try:
            async for event in engine.answer(messages_for_engine, input_message_usage, stop_event):
                if isinstance(event, AgentActionEvent):
                    status_updates.append(event)
                    yield (TestCaseEventType.EXECUTION_STATUS, event)
                elif isinstance(event, AgentMessageEvent):
                    complete_response += event.content
                    yield (TestCaseEventType.AGENT_MESSAGE_CHUNK, {
                        "id": response_message.id,
                        "chunk": event.content
                    })
        finally:
            response_message.text = complete_response
            response_message.status_updates = [e.model_dump(mode="json", by_alias=True) for e in status_updates] or None
            response_message.timestamp = datetime.now(timezone.utc)
            await thread_message_repo.update(response_message)
            await UsageRepository(db).add(input_message_usage)

        yield (TestCaseEventType.AGENT_MESSAGE_COMPLETE, {
            "id": response_message.id,
            "text": complete_response,
        })

    async def _find_evaluator(self, db: AsyncSession, agent: Agent, test_case: TestCase) -> Optional[Evaluator]:
        evaluator_repo = EvaluatorRepository(db)
        test_case_evaluator = await evaluator_repo.find_by_id(test_case.evaluator_id) if test_case.evaluator_id else None
        return test_case_evaluator if test_case_evaluator else await evaluator_repo.find_by_id(agent.evaluator_id) if agent.evaluator_id else None

    def _get_evaluator_temperature(self, llm_model: LlmModel, evaluator: Optional[Evaluator]) -> Optional[float]:
        temperature = evaluator.temperature.get_float() if evaluator else EVALUATOR_DEFAULT_TEMPERATURE.get_float()
        return temperature if llm_model.model_type == LlmModelType.CHAT else None

    def _get_evaluator_reasoning_effort(self, llm_model: LlmModel, evaluator: Optional[Evaluator]) -> Optional[str]:
        reasoning_effort = evaluator.reasoning_effort.value.lower() if evaluator else EVALUATOR_DEFAULT_REASONING_EFFORT.value.lower()
        return reasoning_effort if llm_model.model_type == LlmModelType.REASONING else None

    async def _evaluate_test_case_result(
        self,
        db: AsyncSession,
        user_input: str,
        expected_output: str,
        actual_output: str,
        user_id: int,
        agent: Agent,
        test_case: TestCase,
        stop_event: asyncio.Event
    ) -> Optional[EvaluatorResult]:
        llm_model_repo = AiModelRepository(db)
        evaluator = await self._find_evaluator(db, agent, test_case)
        evaluator_model = cast(LlmModel, await llm_model_repo.find_by_id(evaluator.model_id if evaluator else cast(str, env.internal_evaluator_model)))
        evaluator_usage = MessageUsage(user_id=user_id, agent_id=agent.id, model_id=evaluator_model.id, message_id=None)
        usage_callback = EvaluatorUsageTrackingCallback(evaluator_usage, evaluator_model)

        judge_llm = ai_factory.build_chat_model(
            evaluator_model.id,
            temperature=self._get_evaluator_temperature(evaluator_model, evaluator),
            reasoning_effort=self._get_evaluator_reasoning_effort(evaluator_model, evaluator)
        )
        judge_llm.callbacks.append(usage_callback)
        judge = create_async_llm_as_judge(
            prompt=self._build_test_case_evaluator_prompt(evaluator),
            judge=judge_llm,
        )

        judge_coroutine = judge(
            inputs=user_input,
            outputs=actual_output,
            reference_outputs=expected_output
        )

        judge_task = asyncio.create_task(cast(Coroutine[Any, Any, EvaluatorResult | list[EvaluatorResult]], judge_coroutine))
        stop_task = asyncio.create_task(stop_event.wait())

        done, _ = await asyncio.wait(
            [judge_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        if stop_task in done:
            judge_task.cancel()
            try:
                await judge_task
            except asyncio.CancelledError:
                pass
            return None

        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass

        evaluation_result = await judge_task

        await UsageRepository(db).add(evaluator_usage)

        return cast(EvaluatorResult, evaluation_result)

    async def run(
        self,
        agent_id: int,
        all_test_case_ids: List[int],
        test_case_ids_to_run: List[int],
        user_id: int,
        suite_run_id: int,
        stop_event: asyncio.Event,
        store: BaseStore,
    ) -> None:
        # Event writes use a dedicated session, separate from the main DB session.
        # Tools like docs add Usage objects to the main session; SQLAlchemy's
        # autoflush may begin flushing them mid-transaction. If _broadcast_event
        # commits on the same session during that flush, it poisons the session
        # and causes cascading failures (e.g. duplicate key on Usage re-insert
        # attempts).
        async with AsyncSession(repos_module.engine, expire_on_commit=False) as event_db, \
                   AsyncSession(repos_module.engine, expire_on_commit=False) as db:
            try:
                agent_repo = AgentRepository(db)
                suite_repo = TestSuiteRunRepository(db)
                test_case_repo = TestCaseRepository(db)
                results_repo = TestCaseResultRepository(db)

                agent = cast(Agent, await agent_repo.find_by_id(agent_id))
                suite_run = cast(TestSuiteRun, await suite_repo.find_by_id_and_agent_id(suite_run_id, agent_id))
                all_test_cases = [cast(TestCase, await test_case_repo.find_by_id(tc_id, agent_id)) for tc_id in all_test_case_ids]
                test_cases_to_run_objs = [tc for tc in all_test_cases if tc.thread_id in test_case_ids_to_run]

                pending_results: dict[int, TestCaseResult] = {}
                for test_case in all_test_cases:
                    test_case_name = test_case.thread.name if test_case.thread else None
                    if test_case.thread_id in test_case_ids_to_run:
                        pending_result = await results_repo.save(TestCaseResult(
                            test_case_id=test_case.thread_id,
                            test_suite_run_id=suite_run.id,
                            status=TestCaseResultStatus.PENDING,
                            test_case_name=test_case_name
                        ))
                        pending_results[test_case.thread_id] = pending_result
                    else:
                        await results_repo.save(TestCaseResult(
                            test_case_id=test_case.thread_id,
                            test_suite_run_id=suite_run.id,
                            status=TestCaseResultStatus.SKIPPED,
                            test_case_name=test_case_name
                        ))

                passed = 0
                failed = 0
                errors = 0
                skipped = len(all_test_cases) - len(test_cases_to_run_objs)

                for index, test_case in enumerate(test_cases_to_run_objs):
                    if stop_event.is_set():
                        async for event_type, event_data, counts_as_skip in self._skip_remaining_tests_background(
                            db,
                            test_cases_to_run_objs[index:],
                            pending_results,
                            results_repo
                        ):
                            if counts_as_skip:
                                skipped += 1
                            await self._broadcast_event(event_db, suite_run.id, event_type, event_data)
                        break

                    result = pending_results[test_case.thread_id]
                    await self._broadcast_event(event_db, suite_run.id, TestSuiteEventType.TEST_START.value, {
                        "testCaseId": test_case.thread_id,
                        "resultId": result.id
                    })

                    async for event_type, content in self._run_test_case_stream(
                        db, test_case, agent, user_id, pending_results[test_case.thread_id], stop_event, store
                    ):
                        if event_type == TestCaseEventType.PHASE and content.get("phase") == "completed":
                            status_value = content.get("status")
                            if status_value == TestCaseResultStatus.SUCCESS.value:
                                passed += 1
                            elif status_value == TestCaseResultStatus.FAILURE.value:
                                failed += 1
                            elif status_value == TestCaseResultStatus.SKIPPED.value:
                                skipped += 1
                            elif status_value == TestCaseResultStatus.ERROR.value:
                                errors += 1
                            else:
                                skipped += 1

                        await self._broadcast_event(event_db, suite_run.id, f"suite.test.{event_type.value}",
                            content.model_dump() if event_type == TestCaseEventType.EXECUTION_STATUS else content
                        )

                    await self._broadcast_event(event_db, suite_run.id, TestSuiteEventType.TEST_COMPLETE.value, {
                        "testCaseId": test_case.thread_id,
                        "resultId": result.id,
                        "status": result.status.value,
                        "errorCode": result.error_code,
                        "evaluation": {
                            "analysis": result.evaluator_analysis
                        }
                    })

                suite_run.completed_at = datetime.now(timezone.utc)
                suite_run.total_tests = len(all_test_cases)
                suite_run.passed_tests = passed
                suite_run.failed_tests = failed
                suite_run.error_tests = errors
                suite_run.skipped_tests = skipped
                suite_run.status = TestSuiteRunStatus.FAILURE if errors > 0 or failed > 0 or stop_event.is_set() else TestSuiteRunStatus.SUCCESS

                suite_run = await suite_repo.save(suite_run)

                await self._broadcast_event(event_db, suite_run.id, TestSuiteEventType.COMPLETE.value, {
                    "suiteRunId": suite_run.id,
                    "status": suite_run.status.value,
                    "totalTests": suite_run.total_tests,
                    "passed": suite_run.passed_tests,
                    "failed": suite_run.failed_tests,
                    "errors": suite_run.error_tests,
                    "skipped": suite_run.skipped_tests
                })

            except Exception:
                logger.exception(f"Error running test suite for agent {agent_id}")
                try:
                    await db.rollback()
                except Exception:
                    pass
                await self._cancel_suite_run(suite_run_id, agent_id, db)
                await self._broadcast_event(event_db, suite_run_id, TestSuiteEventType.ERROR.value, {})
            finally:
                await TestSuiteRunEventRepository(event_db).delete_by_suite_run_id(suite_run_id)

    async def _skip_remaining_tests_background(
        self,
        db: AsyncSession,
        remaining_tests: List[TestCase],
        pending_results: Dict[int, TestCaseResult],
        results_repo: TestCaseResultRepository
    ) -> AsyncIterator[Tuple[str, Dict[str, Any], bool]]:
        for test_case in remaining_tests:
            result = pending_results.get(test_case.thread_id)
            if not result:
                continue
            result.status = TestCaseResultStatus.SKIPPED
            result.evaluator_analysis = None
            await results_repo.save(result)

            yield (f"suite.test.{TestCaseEventType.PHASE.value}", {
                "phase": "completed",
                "status": result.status.value,
                "evaluation": None
            }, True)

            yield (TestSuiteEventType.TEST_COMPLETE.value, {
                "testCaseId": test_case.thread_id,
                "resultId": result.id,
                "status": result.status.value,
                "evaluation": {
                    "analysis": result.evaluator_analysis
                }
            }, False)

    async def _cancel_suite_run(self, suite_run_id: int, agent_id: int, db: AsyncSession) -> None:
        try:
            results_repo = TestCaseResultRepository(db)
            suite_repo = TestSuiteRunRepository(db)

            current_suite = await suite_repo.find_by_id_and_agent_id(suite_run_id, agent_id)
            if not current_suite or current_suite.status != TestSuiteRunStatus.RUNNING:
                return

            all_results = await results_repo.find_by_suite_run_id(suite_run_id)
            for result in all_results:
                if result.status in [TestCaseResultStatus.PENDING, TestCaseResultStatus.RUNNING]:
                    result.status = TestCaseResultStatus.SKIPPED
                    db.add(result)

            current_suite.status = TestSuiteRunStatus.FAILURE
            current_suite.completed_at = datetime.now(timezone.utc)

            current_suite.error_tests = sum(1 for r in all_results if r.status == TestCaseResultStatus.ERROR)
            current_suite.passed_tests = sum(1 for r in all_results if r.status == TestCaseResultStatus.SUCCESS)
            current_suite.failed_tests = sum(1 for r in all_results if r.status == TestCaseResultStatus.FAILURE)
            current_suite.skipped_tests = sum(1 for r in all_results if r.status == TestCaseResultStatus.SKIPPED)

            db.add(current_suite)

            await db.commit()

        except Exception as error:
            logger.exception(f"Error during background cleanup for suite {suite_run_id}: {error}")
            try:
                await db.rollback()
            except:
                pass
