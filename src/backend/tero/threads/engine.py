import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable
from contextlib import AsyncExitStack
import json
import logging
from datetime import datetime, timezone
from typing import List, Any, cast, Optional, Callable

from deepagents import (
    CompiledSubAgent,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from deepagents.backends import CompositeBackend
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.config import get_stream_writer
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.messages.utils import _is_message_type
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.store.base import BaseStore
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.domain import Agent, AgentType
from ..agents.repos import AgentToolConfigRepository
from ..ai_models import ai_factory
from ..ai_models.repos import AiModelRepository
from ..core.env import env
from ..core import repos as repos_module
from ..core.repos import rollback_if_aborted
from ..threads.core import trim_messages_to_fit_model
from ..tools.auth import ToolAuthRequestException
from ..tools.core import AgentTool, AgentToolMetadata, report_tool_error, stored_status_result
from ..tools.repos import ToolRepository
from ..tools.subagent import SubagentTool
from ..usage.callbacks import LlmUsageTrackingCallback
from ..usage.domain import MessageUsage
from .agents_store import (
    MEMORY_PATH,
    MEMORY_ROUTE,
    AgentStoreBackend,
    agent_memory_namespace,
    agent_store_namespace,
    ensure_memory_file,
)
from .domain import ThreadMessage, ThreadMessageOrigin, MAX_THREAD_NAME_LENGTH, AgentEvent, AgentActionEvent, AgentFileEvent, AgentMessageEvent, AgentAction, ModelRateLimitError

logger = logging.getLogger(__name__)


@tool
def clock() -> str:
    """Returns the current time in UTC."""
    return f"{datetime.now(timezone.utc)}."


class AgentEngine:
    _MEMORY_INPUT_KEY = "input"

    def __init__(
        self,
        agent: Agent,
        user_id: int,
        store: BaseStore,
        memory_namespace: Optional[tuple[str, ...]] = None,
        memory_namespace_for_agent: Optional[Callable[[int], tuple[str, ...]]] = None,
    ):
        self._agent = agent
        self._user_id = user_id
        self._tool_db: Optional[AsyncSession] = None
        self._store = store
        self._memory_namespace_for_agent = memory_namespace_for_agent or (
            lambda agent_id: agent_memory_namespace(user_id, agent_id)
        )
        self._memory_namespace = memory_namespace or self._memory_namespace_for_agent(agent.id)

    async def load_tools(
        self,
        stack: AsyncExitStack,
        thread_id: Optional[int] = None,
        agent: Optional[Agent] = None,
    ) -> List[AgentTool]:
        # Tools use a separate session so a failed tool statement cannot abort thread/message persistence.
        if self._tool_db is None:
            self._tool_db = await stack.enter_async_context(AsyncSession(repos_module.engine, expire_on_commit=False))
        owner = agent or self._agent
        tool_configs = await AgentToolConfigRepository(self._tool_db).find_by_agent_id(
            agent_id=owner.id
        )
        ret = []
        for tc in tool_configs:
            agent_tool = ToolRepository().find_by_id(tc.tool_id)
            if not agent_tool:
                raise ValueError(f"Tool {tc.tool_id} not found")
            agent_tool.configure(owner, self._user_id, tc.config, self._tool_db, thread_id=thread_id)
            tool = await stack.enter_async_context(agent_tool.load())
            ret.append(tool)
        return ret

    async def load_tools_including_subagents(self, stack: AsyncExitStack) -> None:
        await self._load_tools_including_subagents(stack, self._agent, set())

    async def _load_tools_including_subagents(
        self,
        stack: AsyncExitStack,
        agent: Agent,
        seen: set[int],
    ) -> None:
        if agent.id in seen:
            return
        seen.add(agent.id)
        tools = await self.load_tools(stack, agent=agent)
        for tool in tools:
            if isinstance(tool, SubagentTool):
                await self._load_tools_including_subagents(stack, tool.subagent, seen)

    @staticmethod
    async def _build_langchain_tools(agent_tools: List[AgentTool]) -> List[BaseTool]:
        # tools are grouped by class so tools supporting several configs per agent (eg: sql) can expose a
        # single set of langchain tools for all of their instances, instead of one set per config
        groups: dict[type, List[AgentTool]] = {}
        for agent_tool in agent_tools:
            groups.setdefault(type(agent_tool), []).append(agent_tool)
        return [langchain_tool for tool_class, instances in groups.items()
                for langchain_tool in await tool_class.build_shared_langchain_tools(instances)]

    async def answer(self, messages: List[ThreadMessage], message_usage: MessageUsage, stop_event: asyncio.Event) -> AsyncIterator[AgentEvent]:
        provider = ai_factory.get_provider(self._agent.model.id)
        llm = provider.build_streaming_chat_model(self._agent.model.id, self._agent.model_temperature, self._agent.model_reasoning_effort, max_tokens=self._agent.model.output_token_limit)
        async with AsyncExitStack() as stack:
            thread_id = messages[0].thread_id
            agent_tools = await self.load_tools(stack, thread_id=thread_id)
            tools = await self._build_langchain_tools(
                [t for t in agent_tools if not isinstance(t, SubagentTool)]
            )
            tools.append(clock)
            if self._agent.agent_type == AgentType.DEEP_AGENT and self._agent.memory_enabled:
                await ensure_memory_file(self._store, self._memory_namespace)
            self._display_name_by_subagent_type: dict[str, str] = {}
            self._display_name_by_ns: dict[tuple[str, ...], str] = {}
            subagents = await self._build_compiled_subagents(
                [t for t in agent_tools if isinstance(t, SubagentTool)],
                stack,
                thread_id,
                message_usage,
                ancestors=frozenset({self._agent.id}),
            )
            agent, input = self._build_runtime(llm, tools, messages, subagents=subagents)
            await self._write_files_to_store(messages[-1])
            generated_content = ""
            stream = agent.astream(
                input,
                {"recursion_limit": self._agent.recursion_limit},
                stream_mode=["updates", "messages", "custom"],
                subgraphs=True,
                version="v2",
            )
            try:
                async for chunk in stream:
                    if stop_event.is_set():
                        break
                    mode = chunk["type"]
                    ns = chunk.get("ns", ())  # LangGraph subgraph path: empty = parent, non-empty = nested subagent
                    content = chunk["data"]
                    if mode == "updates":
                        agent_name = self._display_name_by_ns.get(tuple(ns)) if ns else None
                        async for status_update in self._process_updates(content, agent_name=agent_name):
                            yield status_update
                    elif mode == "custom":
                        event = cast(AgentActionEvent, content)
                        if event.agent_name and ns:
                            self._display_name_by_ns[tuple(ns)] = event.agent_name
                        yield self._tag_subagent_event(event, ns)
                    elif mode == "messages":
                        msg, metadata = content
                        metadata = cast(dict, metadata)
                        # lc_agent_name is LangChain's metadata key for the nested agent
                        lc_agent_name = metadata.get("lc_agent_name")
                        if lc_agent_name and ns:
                            agent_name = self._display_name_by_subagent_type.get(lc_agent_name)
                            if agent_name:
                                self._display_name_by_ns[tuple(ns)] = agent_name
                        # we need to filter AI messages since AI messages from tools are also returned
                        if self._should_stream_message(msg, metadata) and msg.content:
                            content = self._get_content(msg.content)
                            generated_content += content
                            yield AgentMessageEvent(content=content)
                        if isinstance(msg, AIMessage) and not lc_agent_name:
                            message_usage.increment_with_metadata(msg.usage_metadata, self._agent.model)
                        elif isinstance(msg, ToolMessage):
                            agent_tool_metadata = AgentToolMetadata.model_validate(msg.response_metadata)
                            message_usage.increment_tool_usage(agent_tool_metadata.tool_usage)
                            if agent_tool_metadata.file:
                                yield AgentFileEvent(file=agent_tool_metadata.file)
            except* Exception as eg:
                if any(provider.is_rate_limit_error(e) for e in eg.exceptions):
                    raise ModelRateLimitError()
                raise

            # If the response was stopped, approximate the token usage
            if stop_event.is_set():
                approximate_input_tokens = llm.get_num_tokens_from_messages(input["messages"]) + self._count_tools_tokens(tools, llm)
                approximate_output_tokens = llm.get_num_tokens(generated_content) if generated_content else 0
                message_usage.increment_with_metadata(
                    {
                        "input_tokens": approximate_input_tokens,
                        "output_tokens": approximate_output_tokens,
                        "total_tokens": approximate_input_tokens + approximate_output_tokens
                    }, self._agent.model)

    def _get_content(self, msg: str | list[str | dict]) -> str:
        if isinstance(msg, str):
            return msg
        if isinstance(msg, list) and msg:
            texts = []
            for item in msg:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
            return "".join(texts)
        raise ValueError(f"Invalid message type: {type(msg)}")

    def _tag_subagent_event(self, event: AgentActionEvent, ns: tuple[str, ...]) -> AgentActionEvent:
        if event.tool_name == "task" and not ns:
            subagent_type = (event.args or {}).get("subagent_type")
            display_name = (
                self._display_name_by_subagent_type.get(subagent_type)
                if isinstance(subagent_type, str)
                else None
            )
            return event.model_copy(update={"description": display_name}) if display_name else event
        if event.agent_name or not ns:
            return event
        agent_name = self._display_name_by_ns.get(tuple(ns))
        return event.model_copy(update={"agent_name": agent_name}) if agent_name else event

    async def _process_updates(
        self,
        content: Any,
        *,
        agent_name: Optional[str] = None,
    ) -> AsyncIterator[AgentActionEvent]:
        if isinstance(content, dict):
            ((key, value), *_) = content.items()
            if key.endswith(".before_model") or key == "pre_model_hook":
                yield AgentActionEvent(action=AgentAction.PRE_MODEL_HOOK, agent_name=agent_name)
            elif key == "model" or key == "agent":
                async for update in self._process_agent(value, agent_name=agent_name):
                    yield update
        else:
            try:
                json_content = json.dumps(content, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                json_content = str(content)
            yield AgentActionEvent(action=AgentAction.UNDEFINED, result=json_content, agent_name=agent_name)

    async def _process_agent(
        self,
        value: Any,
        *,
        agent_name: Optional[str] = None,
    ) -> AsyncIterator[AgentActionEvent]:
        if not isinstance(value, dict) or not value.get("messages"):
            return
        message = value["messages"][0]
        finish_reason = message.response_metadata.get("finish_reason")

        if finish_reason != "stop":
            if hasattr(message, "tool_calls") and message.tool_calls:
                result = []
                for tool_calls in message.tool_calls:
                    result.append(tool_calls["name"])
                yield AgentActionEvent(action=AgentAction.PLANNING, result=result, agent_name=agent_name)

    def _build_runtime(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        messages: List[ThreadMessage],
        subagents: Optional[List[CompiledSubAgent]] = None,
    ) -> tuple[Any, Any]:
        thread_id = messages[0].thread_id
        agent = self._build_agent_graph(
            self._agent,
            tools,
            thread_id,
            llm,
            subagents=subagents,
        )
        input_data = self._build_input(
            messages,
            include_system_prompt=self._agent.agent_type == AgentType.REACT_AGENT,
        )
        return agent, input_data

    async def _build_compiled_subagents(
        self,
        subagent_tools: List[SubagentTool],
        stack: AsyncExitStack,
        thread_id: int,
        message_usage: MessageUsage,
        ancestors: frozenset[int] = frozenset(),
    ) -> List[CompiledSubAgent]:
        compiled_subagents = []
        for subagent_tool in subagent_tools:
            subagent = subagent_tool.subagent
            # Skip to break A -> B -> A cycles
            if subagent.id in ancestors:
                continue

            if subagent.agent_type == AgentType.DEEP_AGENT and subagent.memory_enabled:
                await ensure_memory_file(
                    self._store,
                    self._memory_namespace_for_agent(subagent.id),
                )

            name = SubagentTool.tool_id_for(subagent.id)
            agent_tools = await self.load_tools(stack, thread_id=thread_id, agent=subagent)
            nested_subagent_tools = [t for t in agent_tools if isinstance(t, SubagentTool)]
            nested_subagents = None
            if subagent.agent_type == AgentType.DEEP_AGENT and nested_subagent_tools:
                nested_subagents = await self._build_compiled_subagents(
                    nested_subagent_tools,
                    stack,
                    thread_id,
                    message_usage,
                    ancestors | {subagent.id},
                )

            provider = ai_factory.get_provider(subagent.model.id)
            llm = provider.build_streaming_chat_model(
                subagent.model.id,
                subagent.model_temperature,
                subagent.model_reasoning_effort,
                max_tokens=subagent.model.output_token_limit,
            )
            llm.callbacks = [
                *(llm.callbacks if isinstance(llm.callbacks, list) else []),
                LlmUsageTrackingCallback(message_usage, subagent.model, subagent.id),
            ]
            tools = await self._build_langchain_tools(
                [t for t in agent_tools if not isinstance(t, SubagentTool)]
            )
            tools.append(clock)
            display_name = cast(str, subagent.name)
            self._display_name_by_subagent_type[name] = display_name
            runnable = self._build_agent_graph(
                subagent,
                tools,
                thread_id,
                llm,
                name=name,
                agent_display_name=display_name,
                subagents=nested_subagents,
            )
            description = (
                f"{display_name}. {subagent.description}"
                if subagent.description
                else display_name
            )
            compiled_subagents.append(
                CompiledSubAgent(
                    name=name,
                    description=description,
                    runnable=runnable,
                )
            )
        return compiled_subagents

    def _build_agent_graph(
        self,
        agent: Agent,
        tools: List[BaseTool],
        thread_id: int,
        llm: BaseChatModel,
        *,
        name: Optional[str] = None,
        agent_display_name: Optional[str] = None,
        subagents: Optional[List[CompiledSubAgent]] = None,
    ) -> Any:
        tool_error_handler = _HandleToolErrorsMiddleware(
            cast(AsyncSession, self._tool_db),
            agent_display_name=agent_display_name,
        )
        if agent.agent_type == AgentType.DEEP_AGENT:
            _ensure_deep_agent_harness(llm)
            agent_graph = create_deep_agent(
                model=llm,
                tools=tools,
                system_prompt=agent.system_prompt,
                backend=self._build_agent_backend(agent, thread_id),
                memory=[MEMORY_PATH] if agent.memory_enabled else None,
                middleware=[_BeforeModelHookMiddleware(), tool_error_handler],
                subagents=subagents or None,
                name=name,
            )
            # create_agent does not expose handle_tool_errors; its ToolNode swallows
            # ToolInvocationError. Disable that so middleware sees the real exception.
            self._disable_tool_node_error_handling(agent_graph)
            return agent_graph
        return create_react_agent(
            llm,
            ToolNode(
                tools,
                handle_tool_errors=False,
                awrap_tool_call=tool_error_handler.awrap_tool_call,
            ),
            # Only the root react agent puts the system prompt in messages[0], which the trimmer expects.
            prompt=agent.system_prompt if name is not None else None,
            pre_model_hook=None if name is not None else self._build_message_trimmer(llm, tools),
            name=name,
        )

    def _build_agent_backend(self, agent: Agent, thread_id: int) -> AgentStoreBackend | CompositeBackend:
        thread_backend = AgentStoreBackend(
            store=self._store,
            namespace=lambda _: agent_store_namespace(self._user_id, thread_id),
        )
        if not agent.memory_enabled:
            return thread_backend

        memory_namespace = (
            self._memory_namespace
            if agent.id == self._agent.id
            else self._memory_namespace_for_agent(agent.id)
        )
        memory_backend = AgentStoreBackend(
            store=self._store,
            namespace=lambda _: memory_namespace,
        )
        return CompositeBackend(
            default=thread_backend,
            routes={MEMORY_ROUTE: memory_backend},
        )

    def _should_stream_message(self, msg: Any, metadata: dict) -> bool:
        if isinstance(msg, ToolMessage) and msg.response_metadata.get("return_direct"):
            return True
        if not isinstance(msg, AIMessage) or metadata.get("langgraph_node") == "tools":
            return False
        # Stream only the main agent tokens
        return not metadata.get("lc_agent_name")

    def _build_message_trimmer(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
    ) -> Callable[[Any], Any]:
        def pre_model_hook(state):
            # this is mostly the same logic (but simplified) as invoking langchain trim_messages with last strategy and allow partial
            # , but if a message is too big to fit, instead of returning the last part, we return the first part of the message.
            # This way, we keep the first part of the message that we consider should be more relevan.
            # For example, if user sends text and files, then text is kept, first files are kept, and the first part of the last file that fits is kept as well.
            messages = state["messages"]
            system_message = messages[0]
            messages = messages[1:]
            token_counter = llm.get_num_tokens_from_messages

            # Reverse messages to use _first_max_tokens with reversed logic
            messages = messages[::-1]

            end_index = next(
                i
                for i, x in enumerate(messages)
                if _is_message_type(x, (HumanMessage, ToolMessage))
            )
            messages = messages[end_index:]

            tools_tokens = self._count_tools_tokens(tools, llm)
            system_message_tokens = token_counter([system_message])
            reserved_tokens = tools_tokens + system_message_tokens

            result = trim_messages_to_fit_model(
                messages,
                token_counter=token_counter,
                model=self._agent.model,
                reserved_tokens=reserved_tokens,
                end_on=HumanMessage,
            )
            # Re-reverse the messages and add back the system message
            return {"llm_input_messages": [system_message] + result[::-1]}

        return pre_model_hook

    def _count_tools_tokens(self, tools: List[BaseTool], llm: BaseChatModel) -> int:
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        tools_json = json.dumps(openai_tools)
        return llm.get_num_tokens(tools_json)

    async def _write_files_to_store(self, message: ThreadMessage) -> None:
        if self._agent.agent_type != AgentType.DEEP_AGENT:
            return
        namespace = agent_store_namespace(self._user_id, message.thread_id)
        for file_obj in message.files:
            f = file_obj.file
            if self._is_inline_image_file(f.name, f.content_type):
                continue
            if not self._store_as_text(f.name) or not f.processed_content:
                continue
            await self._store.aput(namespace, f"/{f.name}", {
                "content": f.processed_content,
                "encoding": "utf-8",
            })

    # deepagents treats these extensions as multimodal binary "file" blocks, not text.
    # Models that don't support the "file" content type (e.g. Azure OpenAI) will reject them.
    # Keep these out of the store and send extracted text inline instead.
    @staticmethod
    def _store_as_text(file_name: str) -> bool:
        return not file_name.lower().endswith(".pdf")

    @staticmethod
    def _is_inline_image_file(file_name: str, content_type: str) -> bool:
        return content_type.startswith("image/") and not file_name.lower().endswith(".svg")

    def _build_input(self, messages: List[ThreadMessage], include_system_prompt: bool) -> Any:
        use_store_for_files = self._agent.agent_type == AgentType.DEEP_AGENT
        messages_list: List[BaseMessage] = [SystemMessage(self._agent.system_prompt)] if include_system_prompt else []
        for message in messages:
            if message.origin == ThreadMessageOrigin.USER:
                content = []
                message_text = message.text

                for file_obj in message.files:
                    if self._is_inline_image_file(file_obj.file.name, file_obj.file.content_type):
                        content.append(
                            {
                                "type": "image",
                                "source_type": "base64",
                                "mime_type": file_obj.file.content_type,
                                "data": base64.b64encode(file_obj.file.content).decode("utf-8"),
                            }
                        )
                        message_text += f"\n\nAttached image: {file_obj.file.name}"
                    elif use_store_for_files and self._store_as_text(file_obj.file.name):
                        message_text += f"\n\nAttached file: /{file_obj.file.name}"
                    elif file_obj.file.processed_content:
                        message_text += "\n\n File named: " + file_obj.file.name + "\n\n" + file_obj.file.processed_content

                if message_text.strip():
                    content.append({"type": "text", "text": message_text})

                messages_list.append(HumanMessage(content=content))
            else:
                messages_list.append(AIMessage(message.text))
        return {"messages": messages_list}

    def _disable_tool_node_error_handling(self, agent: Any) -> None:
        graph = agent
        while graph is not None and not hasattr(graph, "nodes"):
            graph = getattr(graph, "bound", None)
        node = getattr(graph, "nodes", {}).get("tools")
        tool_node = getattr(node, "bound", None)
        if isinstance(tool_node, ToolNode):
            tool_node._handle_tool_errors = False


class _BeforeModelHookMiddleware(AgentMiddleware):

    async def abefore_model(self, state, runtime) -> None:
        # Registers a before_model graph node so thinking status is emitted
        # before every LLM call. Deep agents have no other before_model hook.
        return None


class _HandleToolErrorsMiddleware(AgentMiddleware):
    agent_display_name: Optional[str] = None

    def __init__(self, db: AsyncSession, agent_display_name: Optional[str] = None):
        super().__init__()
        self._db = db
        self.agent_display_name = agent_display_name

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        # deepagents built-ins have no StatusUpdateCallbackHandler
        emit_status = tool_name in {
            "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "task",
        }
        args = request.tool_call.get("args")
        if emit_status:
            self._emit_builtin_tool_status(AgentAction.EXECUTING_TOOL, tool_name, args)
        try:
            result = await handler(request)
            if isinstance(result, ToolMessage) and result.status == "error":
                report_tool_error(tool_name, result.content)
                return result
            if emit_status:
                self._emit_builtin_tool_status(
                    AgentAction.EXECUTED_TOOL, tool_name, args, _stored_builtin_result(result)
                )
            return result
        except GraphBubbleUp:
            raise
        except ToolAuthRequestException:
            raise
        except Exception as e:
            report_tool_error(tool_name, e)
            return ToolMessage(
                content=str(e),
                name=tool_name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        finally:
            await rollback_if_aborted(self._db)

    def _emit_builtin_tool_status(
        self,
        action: AgentAction,
        tool_name: str,
        args: Optional[dict] = None,
        result: Optional[Any] = None,
    ) -> None:
        get_stream_writer()(
            AgentActionEvent(
                action=action,
                tool_name=tool_name,
                args=args,
                result=result,
                agent_name=self.agent_display_name,
            )
        )


def _stored_builtin_result(result: Any) -> Optional[list]:
    if isinstance(result, Command):
        update = result.update
        if not update:
            return None
        result = update["messages"][-1]
    return stored_status_result(result.content)


_registered_harness_providers: set[str] = set()


def _ensure_deep_agent_harness(llm: BaseChatModel) -> None:
    try:
        provider = llm._get_ls_params().get("ls_provider")
        if not provider or provider in _registered_harness_providers:
            return
        register_harness_profile(
            provider,
            HarnessProfile(
                excluded_tools=frozenset({"execute"}),
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            ),
        )
        _registered_harness_providers.add(provider)
    except Exception:
        logger.exception("Failed to register deep agent harness profile")


async def build_thread_name(first_thread_message: str, message_usage: MessageUsage, db: AsyncSession) -> str:
    model = await AiModelRepository(db).find_by_id(env.internal_generator_model)
    if not model:
        raise ValueError("Internal generator model not found")
    llm = ai_factory.build_internal_generator_chat_model(model)
    system_prompt = "From the following user message generate a short (less than 80 characters) title for the chat. Do not include quoting or any special characters."
    # invoke the llm using the prompt as system prompt and the first thread message as user message
    response = await llm.ainvoke(
        [SystemMessage(system_prompt), HumanMessage(first_thread_message)]
    )
    response = cast(AIMessage, response)
    message_usage.increment_with_metadata(response.usage_metadata, model)
    return response.text[:MAX_THREAD_NAME_LENGTH].replace("\n", " ")
