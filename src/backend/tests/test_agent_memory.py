from typing import Any
from unittest.mock import MagicMock

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend
from httpx import AsyncClient, Response
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import PrivateAttr

from tero.agents.domain import Agent, AgentType
from tero.core.api import BASE_PATH
from tero.threads.domain import ThreadMessage, ThreadMessageOrigin
from tero.threads.agents_store import (
    MEMORY_PATH,
    MEMORY_STORE_KEY,
    agent_test_case_memory_namespace_prefix,
    agent_memory_namespace,
    agent_store_namespace,
    clear_store_namespace,
    ensure_memory_file,
    scoped_agent_memory_namespace,
)
from tero.threads.engine import AgentEngine
from tests.common import USER_ID, add_message_to_thread, create_thread


def _deep_agent(memory_enabled: bool, agent_id: int = 7) -> Agent:
    return Agent(
        id=agent_id,
        user_id=1,
        model_id="gpt-5-mini",
        system_prompt="Test prompt",
        agent_type=AgentType.DEEP_AGENT,
        memory_enabled=memory_enabled,
    )


def _message(thread_id: int = 11) -> ThreadMessage:
    return ThreadMessage(thread_id=thread_id, origin=ThreadMessageOrigin.USER, text="Hello")


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFakeModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._seen_messages.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def _agent_response_text(response: Response) -> str:
    response.raise_for_status()
    body = (await response.aread()).decode()
    assert "event: error" not in body

    chunks = []
    for event in body.split("\r\n\r\n"):
        lines = event.split("\r\n")
        if any(line.startswith("event: ") for line in lines):
            continue
        chunks.extend(line[6:] for line in lines if line.startswith("data: "))
    return "".join(chunks)


async def _create_deep_agent(client: AsyncClient, memory_enabled: bool) -> int:
    agent_response = await client.post(f"{BASE_PATH}/agents")
    agent_response.raise_for_status()
    agent_id = agent_response.json()["id"]

    update_response = await client.put(
        f"{BASE_PATH}/agents/{agent_id}",
        json={
            "modelId": "gpt-5-mini",
            "systemPrompt": "You are a helpful AI agent.",
            "temperature": "PRECISE",
            "reasoningEffort": "LOW",
            "agentType": AgentType.DEEP_AGENT.value,
            "memoryEnabled": memory_enabled,
        },
    )
    update_response.raise_for_status()
    return agent_id


async def test_deep_agent_routes_memory_to_user_agent_namespace(monkeypatch):
    captured = {}
    graph = MagicMock()
    monkeypatch.setattr(
        "tero.threads.engine.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or graph,
    )
    store = InMemoryStore()

    runtime, _ = AgentEngine(_deep_agent(True), 3, store)._build_runtime(
        MagicMock(), [], [_message()]
    )

    assert runtime is graph
    assert captured["memory"] == [MEMORY_PATH]
    assert captured["system_prompt"] == "Test prompt"
    backend = captured["backend"]
    assert isinstance(backend, CompositeBackend)

    memory_write = await backend.awrite(MEMORY_PATH, "Remember this")
    thread_write = await backend.awrite("/notes.txt", "Thread only")

    assert memory_write.error is None
    assert thread_write.error is None
    memory_item = await store.aget(agent_memory_namespace(3, 7), MEMORY_STORE_KEY)
    thread_item = await store.aget(agent_store_namespace(3, 11), "/notes.txt")
    assert memory_item is not None
    assert memory_item.value["content"] == "Remember this"
    assert thread_item is not None
    assert thread_item.value["content"] == "Thread only"


async def test_disabling_memory_preserves_it_for_reactivation(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "tero.threads.engine.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or MagicMock(),
    )
    store = InMemoryStore()
    namespace = agent_memory_namespace(3, 7)
    remembered = {"content": "# Agent Memory\n- User prefers concise answers", "encoding": "utf-8"}
    await store.aput(namespace, MEMORY_STORE_KEY, remembered)

    AgentEngine(_deep_agent(False), 3, store)._build_runtime(
        MagicMock(), [], [_message()]
    )

    assert captured["memory"] is None
    assert isinstance(captured["backend"], StoreBackend)
    thread_write = await captured["backend"].awrite("/notes.txt", "Thread only")
    assert thread_write.error is None
    assert await store.aget(agent_store_namespace(3, 11), "/notes.txt") is not None
    stored = await store.aget(namespace, MEMORY_STORE_KEY)
    assert stored is not None
    assert stored.value == remembered

    AgentEngine(_deep_agent(True), 3, store)._build_runtime(
        MagicMock(), [], [_message()]
    )

    assert captured["memory"] == [MEMORY_PATH]
    stored = await store.aget(namespace, MEMORY_STORE_KEY)
    assert stored is not None
    assert stored.value == remembered


async def test_clear_store_namespace_only_removes_target_namespace():
    store = InMemoryStore()
    target = ("test", "target")
    preserved = ("test", "preserved")
    await store.aput(target, "one", {"content": "1"})
    await store.aput(target, "two", {"content": "2"})
    await store.aput(preserved, "one", {"content": "keep"})

    await clear_store_namespace(store, target)

    assert await store.asearch(target) == []
    assert len(await store.asearch(preserved)) == 1


async def test_subagent_memory_isolation_and_cleanup():
    store = InMemoryStore()
    parent = _deep_agent(memory_enabled=True)
    subagent = _deep_agent(memory_enabled=True, agent_id=8)
    test_namespace_prefix = agent_test_case_memory_namespace_prefix(
        user_id=3,
        suite_run_id=17,
        test_case_id=21,
    )

    def test_memory_namespace(agent_id: int) -> tuple[str, ...]:
        return scoped_agent_memory_namespace(test_namespace_prefix, agent_id)

    real_namespace = agent_memory_namespace(3, subagent.id)
    test_namespace = test_memory_namespace(subagent.id)
    remembered = {"content": "Production memory", "encoding": "utf-8"}
    await store.aput(real_namespace, MEMORY_STORE_KEY, remembered)

    engine = AgentEngine(
        parent,
        user_id=3,
        store=store,
        memory_namespace=test_memory_namespace(parent.id),
        memory_namespace_for_agent=test_memory_namespace,
    )
    backend = engine._build_agent_backend(subagent, thread_id=100)
    assert isinstance(backend, CompositeBackend)

    result = await backend.awrite(MEMORY_PATH, "Test-only memory")

    assert result.error is None
    test_memory = await store.aget(test_namespace, MEMORY_STORE_KEY)
    assert test_memory is not None
    assert test_memory.value["content"] == "Test-only memory"
    production_memory = await store.aget(real_namespace, MEMORY_STORE_KEY)
    assert production_memory is not None
    assert production_memory.value == remembered

    await clear_store_namespace(store, test_namespace_prefix)

    assert await store.asearch(test_namespace_prefix) == []
    production_memory = await store.aget(real_namespace, MEMORY_STORE_KEY)
    assert production_memory is not None
    assert production_memory.value == remembered


async def test_ensure_memory_file_initializes_without_overwriting():
    store = InMemoryStore()
    namespace = ("user_1", "agent_7", "memory")

    await ensure_memory_file(store, namespace)
    initialized = await store.aget(namespace, MEMORY_STORE_KEY)
    assert initialized is not None
    assert initialized.value == {"content": "# Agent Memory\n", "encoding": "utf-8"}

    remembered = {"content": "# Agent Memory\n- Keep this", "encoding": "utf-8"}
    await store.aput(namespace, MEMORY_STORE_KEY, remembered)
    await ensure_memory_file(store, namespace)
    preserved = await store.aget(namespace, MEMORY_STORE_KEY)
    assert preserved is not None
    assert preserved.value == remembered


async def test_deep_agent_memory_is_loaded_across_threads():
    codename = "ORBIT-CYAN-7319"
    agent = _deep_agent(memory_enabled=True)
    store = InMemoryStore()
    namespace = agent_memory_namespace(USER_ID, agent.id)
    await ensure_memory_file(store, namespace)

    first_model = _ToolCallingFakeModel(responses=[
        AIMessage(
            content="",
            tool_calls=[{
                "name": "write_file",
                "args": {"file_path": MEMORY_PATH, "content": f"# Agent Memory\n- Codename: {codename}"},
                "id": "write-memory",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="SAVED"),
    ])
    first_graph = create_deep_agent(
        model=first_model,
        backend=AgentEngine(agent, USER_ID, store)._build_agent_backend(agent, thread_id=11),
        memory=[MEMORY_PATH],
    )
    await first_graph.ainvoke({"messages": [HumanMessage(content="Remember my codename.")]})

    stored_memory = await store.aget(namespace, MEMORY_STORE_KEY)
    assert stored_memory is not None
    assert codename in stored_memory.value["content"]

    second_model = _ToolCallingFakeModel(responses=[AIMessage(content="Done")])
    second_graph = create_deep_agent(
        model=second_model,
        backend=AgentEngine(agent, USER_ID, store)._build_agent_backend(agent, thread_id=12),
        memory=[MEMORY_PATH],
    )
    await second_graph.ainvoke({"messages": [HumanMessage(content="What is my codename?")]})

    assert any(
        codename in str(message.content)
        for message in second_model._seen_messages[0]
        if isinstance(message, SystemMessage)
    )


async def test_deep_agent_does_not_remember_user_fact_when_memory_is_disabled(
    client: AsyncClient,
    agent_store: BaseStore,
):
    codename = "EMBER-VIOLET-4826"
    agent_id = await _create_deep_agent(client, memory_enabled=False)

    first_thread_response = await create_thread(agent_id, client)
    first_thread_response.raise_for_status()
    first_thread_id = first_thread_response.json()["id"]
    async with add_message_to_thread(
        client,
        first_thread_id,
        f"Remember that my verification codename is {codename}.",
    ) as response:
        await _agent_response_text(response)

    assert await agent_store.asearch(agent_memory_namespace(USER_ID, agent_id)) == []

    second_thread_response = await create_thread(agent_id, client)
    second_thread_response.raise_for_status()
    second_thread_id = second_thread_response.json()["id"]
    assert second_thread_id != first_thread_id

    async with add_message_to_thread(
        client,
        second_thread_id,
        "What is my verification codename?",
    ) as response:
        assert codename not in await _agent_response_text(response)
