import aiofiles
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import List, Any, Optional, cast, Sequence
from uuid import UUID
from enum import Enum
import tiktoken

from langchain.indexes import SQLRecordManager, aindex
from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun, AsyncCallbackManager
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownTextSplitter, CharacterTextSplitter
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import AgentToolConfig, AgentToolConfigFile
from ...agents.repos import AgentToolConfigFileRepository
from ...ai_models import ai_factory, azure_provider
from ...ai_models.domain import LlmModel
from ...ai_models.repos import AiModelRepository
from ...core.assets import solve_asset_path
from ...core.env import env
from ...files.domain import File, FileProcessor
from ...files.file_quota import FileQuota, CurrentQuota
from ...files.parser import extract_file_text
from ...usage.domain import Usage, MessageUsage, UsageType
from ...usage.repos import UsageRepository
from ...users.domain import User
from ...files.repos import FileRepository
from ...threads.domain import AgentActionEvent, AgentAction
from ..core import AgentToolWithFiles, load_schema
from .domain import DocToolFile, DocToolConfig
from .repos import DocToolFileRepository, DocToolConfigRepository

logger = logging.getLogger(__name__)
DOCS_TOOL_ID = "docs"
ADVANCED_FILE_PROCESSING = "advancedFileProcessing"


def embedding_tokens_from_text(text: str) -> int:
    embeddings_encoding = tiktoken.encoding_for_model(env.embedding_model)
    return len(embeddings_encoding.encode(text))


class DocumentUrlSolvingRetriever(VectorStoreRetriever):
    agent_id: int
    tool_id: str
    embedding_usage: Usage

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        self.embedding_usage.increment(embedding_tokens_from_text(query), env.embedding_cost_per_1k_tokens)
        ret = await super()._aget_relevant_documents(
            query, run_manager=run_manager, **kwargs
        )
        for doc in ret:
            doc.metadata["url"] = (
                f"{env.frontend_url}/agents/{self.agent_id}/tools/{self.tool_id}/files/{doc.metadata['id']}"
            )
        return ret


class DocsToolArgs(BaseModel):
    user_query: str = Field(description="The query from the user")


class DocsTool(AgentToolWithFiles):
    id: str = DOCS_TOOL_ID
    name: str = "Docs"
    description: str = (
        "Allows to use information from uploaded files in agent responses"
    )
    config_schema: dict = load_schema(__file__)
    _embedding_usage: Optional[Usage] = None

    @model_validator(mode="after")
    def remove_advanced_processing_if_not_configured(self):
        if not env.azure_doc_intelligence_endpoint or not env.azure_doc_intelligence_key:
            self.config_schema["properties"].pop(ADVANCED_FILE_PROCESSING)
            self.config_schema["required"].remove(ADVANCED_FILE_PROCESSING)
        return self

    @property
    def embedding_usage(self) -> Usage:
        if self._embedding_usage is None:
            self._embedding_usage = Usage(user_id=self.user_id, agent_id=self.agent.id, model_id=env.embedding_model, type=UsageType.EMBEDDING_TOKENS)
        return self._embedding_usage

    async def _setup_tool(
        self, prev_config: Optional[AgentToolConfig]
    ) -> Optional[dict]:
        await self._build_record_manager().acreate_schema()

    def _build_record_manager(self) -> SQLRecordManager:
        return SQLRecordManager(
            self._build_index_namespace(self.agent.id), engine=self._get_async_engine()
        )

    @staticmethod
    def _build_index_namespace(agent_id: int) -> str:
        return f"postgres/{DocsTool._build_collection_name(agent_id)}"

    @staticmethod
    def _build_collection_name(agent_id: int) -> str:
        return f"docs_{agent_id}"

    def _get_async_engine(self) -> AsyncEngine:
        return cast(AsyncEngine, self.db.bind)

    async def teardown(self):
        await aindex(
            [], self._build_record_manager(), self._build_vectorstore(), cleanup="full", key_encoder="sha256"
        )
        await DocToolFileRepository(self.db).remove_by_agent_id(self.agent.id)
        await DocToolConfigRepository(self.db).remove(self.agent.id)

    def _build_vectorstore(self):
        ai_provider = ai_factory.get_provider(env.embedding_model)
        return PGVector(embeddings=ai_provider.build_embedding(env.embedding_model), connection=self._get_async_engine(),
                        collection_name=self._build_collection_name(self.agent.id), use_jsonb=True)

    async def add_file(self, file: File, user: User):
        await self._handle_file(file, user)

    async def _handle_file(self, file: File, user: User):
        pdf_parsing_usage = None
        message_usage = None
        try:
            model = await self._find_description_model()
            pdf_parsing_usage = Usage(user_id=file.user_id, agent_id=self.agent.id, model_id=None, type=UsageType.PDF_PARSING)
            message_usage = MessageUsage(user_id=file.user_id, agent_id=self.agent.id, model_id=model.id)
            current_usage = await UsageRepository(self.db).find_current_month_user_usage_usd(file.user_id)
            file_quota = FileQuota(pdf_parsing_usage, None, CurrentQuota(current_usage, user.monthly_usd_limit))
            file.file_processor = FileProcessor.ENHANCED if self.config.get(ADVANCED_FILE_PROCESSING) else FileProcessor.BASIC
            file_doc = await self._build_document(file, file_quota)
            file.processed_content = file_doc.page_content
            await FileRepository(self.db).update(file)
            await self._update_tool_description_with_file(file, model, message_usage)
            docs = MarkdownTextSplitter.from_tiktoken_encoder(encoding_name=tiktoken.encoding_for_model(env.embedding_model).name,
                                        chunk_size=env.docs_tool_chunk_size, chunk_overlap=env.docs_tool_chunk_overlap).split_documents([file_doc])
            embeddings_tokens = sum(embedding_tokens_from_text(doc.page_content) for doc in docs)
            self.embedding_usage.increment(embeddings_tokens, env.embedding_cost_per_1k_tokens)
            await aindex(docs, self._build_record_manager(), self._build_vectorstore(), cleanup="incremental",
                            source_id_key="id", key_encoder="sha256")

        finally:
            usage_repo = UsageRepository(self.db)
            await usage_repo.add(pdf_parsing_usage)
            await usage_repo.add(message_usage)
            await usage_repo.add(self.embedding_usage)
        
    async def _update_tool_description_with_file(self, file: File, model: LlmModel, message_usage: MessageUsage):
        description = await self._generate_file_description(file, model, message_usage)
        await DocToolFileRepository(self.db).add(
            DocToolFile(file_id=file.id, description=description, agent_id=self.agent.id))
        await self._update_tool_description(model, message_usage)

    async def _find_description_model(self) -> LlmModel:
        ret = await AiModelRepository(self.db).find_by_id(env.internal_generator_model)
        if not ret:
            raise ValueError("Internal generator model not found")
        return ret

    @staticmethod
    async def _build_document(file: File, file_quota: FileQuota):
        metadata = {'id': str(file.id)}
        content = await extract_file_text(file, file_quota)
        return Document(page_content=content, metadata=metadata)

    async def _generate_file_description(self, file: File, model: LlmModel, message_usage: MessageUsage) -> str:
        async with aiofiles.open(solve_asset_path('file-description-prompt.md', __file__)) as f:
            system_prompt = await f.read()
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            model_name=model.id,
            chunk_size=env.docs_tool_description_chunk_size,
            chunk_overlap=env.docs_tool_description_chunk_overlap,
        )
        chunks = text_splitter.split_text(cast(str, file.processed_content))
        ret = "none"
        for chunk in chunks:
            prompt = system_prompt + f"Previous Description: {ret}\n\n" + f"## File contents\n\n{chunk}"
            ret = await self._generate_description(prompt, 200, model, message_usage)
        return ret

    @staticmethod
    async def _generate_description(prompt: str, max_length: int, model: LlmModel, message_usage: MessageUsage) -> str:
        llm = ai_factory.build_chat_model(model.id, env.internal_generator_temperature)
        response = await llm.ainvoke([HumanMessage(prompt)])
        response = cast(AIMessage, response)
        message_usage.increment_with_metadata(response.usage_metadata, model)
        response_msg = cast(str, response.content)
        return (
            response_msg
            if len(response_msg) <= max_length
            else response_msg[:max_length]
        )

    async def _update_tool_description(self, model: LlmModel, message_usage: MessageUsage):
        tool_files = await DocToolFileRepository(self.db).find_by_agent_id(self.agent.id)
        repo = DocToolConfigRepository(self.db)
        if not tool_files:
            await repo.remove(self.agent.id)
        else:
            tool_description = await self._generate_tool_description(tool_files, model, message_usage)
            await repo.add(DocToolConfig(agent_id=self.agent.id, description=tool_description))

    async def _generate_tool_description(self, files: List[DocToolFile], model: LlmModel, message_usage: MessageUsage) -> str:
        async with aiofiles.open(solve_asset_path('tool-description-prompt.md', __file__)) as f:
            prompt = await f.read()
        for f in files:
            prompt += f"\n- {f.description}"
        return await self._generate_description(prompt, 200, model, message_usage)

    async def update_file(self, file: File, user: User):
        # clear processed content before update to avoid partial quota-exceeded state
        await self._remove_file_processed_content(file)
        await self._handle_file(file, user)            

    async def remove_file(self, file: File):
        # langchain does not provide an abstraction to just remove one document from the index, so we built this logic
        # from index method cleanup logic
        record_manager = self._build_record_manager()
        keys = await record_manager.alist_keys(group_ids=[str(file.id)])
        vectorstore = self._build_vectorstore()
        await vectorstore.adelete(keys)
        await record_manager.adelete_keys(keys)
        await DocToolFileRepository(self.db).remove(self.agent.id, file.id)
        model = await self._find_description_model()
        message_usage = MessageUsage(user_id=file.user_id, agent_id=self.agent.id, model_id=model.id)
        try:
            await self._update_tool_description(model, message_usage)
        finally:
            await UsageRepository(self.db).add(message_usage)

    async def _remove_file_processed_content(self, file: File):
        file.processed_content = None
        await FileRepository(self.db).update(file)

    async def _run(self, user_query: str) -> str:
        async with aiofiles.open(solve_asset_path("answer-prompt.md", __file__)) as f:
            template = await f.read()
        template = (
            f"{template}\n\n{self.agent.system_prompt}"
            if self.agent.system_prompt
            else template
        )
        prompt = ChatPromptTemplate.from_template(template)
        llm = ai_factory.build_chat_model(self.agent.model.id, self.agent.model_temperature, self.agent.model_reasoning_effort)
        retriever = self._build_retriever()
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        config = ensure_config()
        if "callbacks" in config:
            cast(AsyncCallbackManager, config["callbacks"]).inheritable_handlers.append(DocsStatusUpdateCallbackHandler(self.id, self.description))
        response = await rag_chain.ainvoke(user_query, config=config)
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.EXECUTING_TOOL,
                tool_name=self.id,
                step=DocsExecutionStep.GROUNDING_RESPONSE,
            )
        )
        grounded_response = await self._ground_response(response, retriever, llm)
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.EXECUTING_TOOL,
                tool_name=self.id,
                step=DocsExecutionStep.GROUNDED_RESPONSE,
            )
        )
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.EXECUTED_TOOL,
                tool_name=self.id,
            )
        )
        await UsageRepository(self.db).add(self.embedding_usage)
        return grounded_response

    def _build_retriever(self) -> VectorStoreRetriever:
        return DocumentUrlSolvingRetriever(
            vectorstore=self._build_vectorstore(),
            search_kwargs={"k": env.docs_tool_retrieve_top},
            agent_id=self.agent.id,
            tool_id=self.id,
            embedding_usage=self.embedding_usage,
        )

    @staticmethod
    async def _ground_response(
        response: str, retriever: VectorStoreRetriever, llm: BaseChatModel
    ) -> str:
        async with aiofiles.open(
            solve_asset_path("ground-check-prompt.md", __file__)
        ) as f:
            template = await f.read()
        verification_prompt = ChatPromptTemplate.from_template(template)
        verification_chain = (
            {"context": retriever, "response": RunnablePassthrough()}
            | verification_prompt
            | llm
            | StrOutputParser()
        )
        return await verification_chain.ainvoke(response)

    @asynccontextmanager
    async def load(self) -> AsyncIterator['DocsTool']:
        yield self

    async def build_langchain_tools(self) -> List[BaseTool]:
        tool_config = await DocToolConfigRepository(self.db).find_by_agent_id(
            self.agent.id
        )
        if not tool_config:
            return []
        docs_tool = self

        return [StructuredTool(
            name=docs_tool.id,
            description=tool_config.description,
            args_schema=DocsToolArgs,
            coroutine=lambda user_query: docs_tool._run(user_query),
        )]


    async def clone(
        self,
        agent_id: int,
        cloned_agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> None:
        file_id_map = await self._clone_files(
            agent_id, cloned_agent_id, tool_id, user_id, db
        )
        await self._clone_vector_store(agent_id, cloned_agent_id, db)
        await self._clone_record_manager(agent_id, cloned_agent_id, db, file_id_map)
        await self._clone_tool_config(agent_id, cloned_agent_id, db)
        await self._clone_tool_files(agent_id, cloned_agent_id, file_id_map, db)

    async def _clone_files(
        self,
        agent_id: int,
        cloned_agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> dict:
        tool_file_repo = AgentToolConfigFileRepository(db)
        files = await tool_file_repo.find_with_content_by_agent_and_tool(
            agent_id, tool_id
        )
        file_id_map = {}

        for file in files:
            new_file = file.clone(user_id=user_id)
            new_file = await FileRepository(db).add(new_file)
            await tool_file_repo.add(
                AgentToolConfigFile(
                    agent_id=cloned_agent_id, tool_id=tool_id, file_id=new_file.id
                )
            )
            file_id_map[file.id] = new_file.id
        return file_id_map

    async def _clone_vector_store(
        self, agent_id: int, cloned_agent_id: int, db: AsyncSession
    ) -> None:
        old_collection = self._build_collection_name(agent_id)
        new_collection = self._build_collection_name(cloned_agent_id)
        result = await db.execute(
            text("""
                INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
                SELECT gen_random_uuid(), :new_collection, cmetadata
                FROM langchain_pg_collection
                WHERE name = :old_collection
                ON CONFLICT (name) DO NOTHING
                RETURNING uuid
            """),
            {"new_collection": new_collection, "old_collection": old_collection},
        )

        await db.execute(
            text("""
                INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                SELECT gen_random_uuid()::varchar, :new_collection_uuid, embedding, document, cmetadata
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :old_collection
                )
            """),
            {
                "new_collection_uuid": result.scalar_one_or_none(),
                "old_collection": old_collection,
            },
        )
        await db.commit()

    async def _clone_record_manager(
        self, agent_id: int, cloned_agent_id: int, db: AsyncSession, file_id_map: dict
    ) -> None:
        selects = []
        params = {
            "new_namespace": self._build_index_namespace(cloned_agent_id),
            "old_namespace": self._build_index_namespace(agent_id),
        }
        for idx, (old_file_id, new_file_id) in enumerate(file_id_map.items()):
            selects.append(f"""
                SELECT gen_random_uuid(), REPLACE(key, :old_file_id_{idx}, :new_file_id_{idx}),
                       :new_namespace, :new_file_id_{idx}, updated_at
                FROM upsertion_record
                WHERE namespace = :old_namespace AND group_id = :old_file_id_{idx}
            """)
            params[f"old_file_id_{idx}"] = str(old_file_id)
            params[f"new_file_id_{idx}"] = str(new_file_id)

        query = f"""
            INSERT INTO upsertion_record (uuid, key, namespace, group_id, updated_at)
            {" UNION ALL ".join(selects)}
            ON CONFLICT (key, namespace) DO NOTHING
        """
        await db.execute(text(query), params)
        await db.commit()

    async def _clone_tool_config(
        self, agent_id: int, cloned_agent_id: int, db: AsyncSession
    ) -> None:
        doc_tool_config_repo = DocToolConfigRepository(db)
        original_config = await doc_tool_config_repo.find_by_agent_id(agent_id)
        if original_config:
            cloned_config = DocToolConfig(
                agent_id=cloned_agent_id, description=original_config.description
            )
            await doc_tool_config_repo.add(cloned_config)

    async def _clone_tool_files(
        self, agent_id: int, cloned_agent_id: int, file_id_map: dict, db: AsyncSession
    ) -> None:
        doc_tool_file_repo = DocToolFileRepository(db)
        original_files = await doc_tool_file_repo.find_by_agent_id(agent_id)
        for file in original_files:
            new_file_id = file_id_map.get(file.file_id, file.file_id)
            cloned_file = DocToolFile(agent_id=cloned_agent_id, file_id=new_file_id, description=file.description)
            await doc_tool_file_repo.add(cloned_file)


class DocsExecutionStep(str, Enum):
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    RETRIEVING = "retrieving"
    RETRIEVED = "retrieved"
    GROUNDING_RESPONSE = "groundingResponse"
    GROUNDED_RESPONSE = "groundedResponse"


class DocsToolExecutionEvent(AgentActionEvent):
    step: Optional[DocsExecutionStep] = None


class DocsStatusUpdateCallbackHandler(AsyncCallbackHandler):
    def __init__(self, tool_id: str, description: str):
        self.tool_id = tool_id
        self.description = description

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.EXECUTING_TOOL,
                tool_name=self.tool_id,
                step=DocsExecutionStep.ANALYZING,
            )
        )

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.EXECUTING_TOOL,
                tool_name=self.tool_id,
                step=DocsExecutionStep.ANALYZED,
            )
        )

    async def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.TOOL_ERROR,
                tool_name=self.tool_id,
                result=str(error),
            )
        )

    async def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.TOOL_ERROR,
                tool_name=self.tool_id,
                result=str(error),
            )
        )

    async def on_retriever_start(
        self,
        serialized: Optional[dict[str, Any]],
        query: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.EXECUTING_TOOL,
                tool_name=self.tool_id,
                step=DocsExecutionStep.RETRIEVING,
                args=query,
                description=self.description,
            )
        )

    async def on_retriever_end(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> None:
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.EXECUTING_TOOL,
                tool_name=self.tool_id,
                step=DocsExecutionStep.RETRIEVED,
                result=[doc.page_content[0:150] + "..." for doc in documents],
            )
        )

    async def on_retriever_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        get_stream_writer()(
            DocsToolExecutionEvent(
                action=AgentAction.TOOL_ERROR,
                tool_name=self.tool_id,
                result=str(error),
            )
        )
