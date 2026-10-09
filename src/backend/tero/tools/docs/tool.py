import aiofiles
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from functools import cache
import json
import logging
from tokenizers import Tokenizer
from typing import List, Any, Optional, cast, Sequence
from uuid import UUID

from langchain_classic.indexes import SQLRecordManager, aindex
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun, AsyncCallbackManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import AgentToolConfig
from ...agents.repos import AgentToolConfigFileRepository
from ...ai_models import ai_factory
from ...ai_models.domain import LlmModel
from ...ai_models.repos import AiModelRepository
from ...core.assets import solve_asset_path
from ...core.env import env
from ...files.domain import File, FileProcessor
from ...files.core import FileQuota, CurrentQuota
from ...files.parser import extract_file_text
from ...files.processors.pdf import is_enhanced_pdf_processor_available
from ...files.repos import FileRepository
from ...threads.domain import AgentActionEvent, AgentAction, ModelRateLimitError
from ...usage.domain import Usage, MessageUsage, UsageType
from ...usage.repos import UsageRepository
from ...users.domain import User
from ..core import AgentToolWithFiles, UserFacingToolError, load_schema, report_tool_error
from .domain import DocToolFile, DocToolConfig
from .repos import DocToolFileRepository, DocToolConfigRepository


logger = logging.getLogger(__name__)
DOCS_TOOL_ID = "docs"
ADVANCED_FILE_PROCESSING = "advancedFileProcessing"
RESPONSE_GROUNDING = "responseGrounding"

# Description/embedding generation can hit the model provider's rate limits during
# ingestion, and the SDK defaults (~600s timeout, 2 retries) aren't enough to ride
# them out. Give ingestion more headroom so it waits/retries longer instead of aborting.
DOCS_INGESTION_TIMEOUT_SECONDS = 900
DOCS_INGESTION_MAX_RETRIES = 2


def _count_tokens(text: str) -> int:
    embedding_model = env.embedding_model
    ai_provider = ai_factory.get_provider(embedding_model)
    return ai_provider.count_tokens(text, embedding_model)


_DOCS_FULLTEXT_DDL = (
    """
    ALTER TABLE langchain_pg_embedding
      ADD COLUMN IF NOT EXISTS content_tsv tsvector
      GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED
    """,
    """
    CREATE INDEX IF NOT EXISTS langchain_pg_embedding_content_tsv_idx
      ON langchain_pg_embedding USING gin (content_tsv)
    """,
)


_HYBRID_SEARCH_SQL = """
    WITH query_tsquery AS (
        SELECT CASE
            WHEN plainto_tsquery('simple', :query_text) = ''::tsquery THEN NULL
            ELSE to_tsquery('simple', replace(plainto_tsquery('simple', :query_text)::text, ' & ', ' | '))
        END AS tsq
    ),
    vector_search AS (
        SELECT id, document, cmetadata,
               ROW_NUMBER() OVER (ORDER BY embedding <=> :query_embedding) AS rank
        FROM langchain_pg_embedding
        WHERE collection_id = :collection_id
          AND (CAST(:file_ids AS text[]) IS NULL OR cmetadata->>'id' = ANY(CAST(:file_ids AS text[])))
        ORDER BY embedding <=> :query_embedding
        LIMIT :k
    ),
    text_search AS (
        SELECT e.id, e.document, e.cmetadata,
               ROW_NUMBER() OVER (
                 ORDER BY ts_rank(e.content_tsv, q.tsq) DESC
               ) AS rank
        FROM langchain_pg_embedding e, query_tsquery q
        WHERE e.collection_id = :collection_id
          AND q.tsq IS NOT NULL
          AND e.content_tsv @@ q.tsq
          AND (CAST(:file_ids AS text[]) IS NULL OR e.cmetadata->>'id' = ANY(CAST(:file_ids AS text[])))
        ORDER BY ts_rank(e.content_tsv, q.tsq) DESC
        LIMIT :k
    )
    SELECT COALESCE(v.id, t.id) AS id,
           COALESCE(v.document, t.document) AS document,
           COALESCE(v.cmetadata, t.cmetadata) AS cmetadata,
           v.rank IS NOT NULL AS matched_vector,
           t.rank IS NOT NULL AS matched_text,
           COALESCE(1.0 / (:rrf_k + v.rank), 0) + COALESCE(1.0 / (:rrf_k + t.rank), 0) AS score
    FROM vector_search v
    FULL OUTER JOIN text_search t USING (id)
    ORDER BY score DESC
    LIMIT :k
"""


class HybridSearchRetriever(BaseRetriever):
    agent_id: int
    tool_id: str
    embedding_usage: Usage
    db: AsyncSession
    collection_name: str
    embeddings: Embeddings
    file_ids: Optional[List[str]] = None

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise NotImplementedError("HybridSearchRetriever only supports async retrieval")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        collection_id = await self._resolve_collection_id()
        if collection_id is None:
            return []
        query_embedding = await self.embeddings.aembed_query(query)
        self.embedding_usage.increment(_count_tokens(query), env.embedding_cost_per_1k_tokens)

        for statement in _DOCS_FULLTEXT_DDL:
            await self.db.execute(text(statement))

        result = await self.db.execute(
            text(_HYBRID_SEARCH_SQL),
            {
                "collection_id": collection_id,
                "query_embedding": "[" + ",".join(str(value) for value in query_embedding) + "]",
                "query_text": query,
                "file_ids": self.file_ids,
                "rrf_k": env.docs_tool_rrf_k,
                "k": env.docs_tool_retrieve_top,
            },
        )

        return [
            Document(
                page_content=row.document,
                metadata={
                    **row.cmetadata,
                    "matched_vector": row.matched_vector,
                    "matched_text": row.matched_text,
                    "url": f"{env.frontend_url}/agents/{self.agent_id}/tools/{self.tool_id}/files/{row.cmetadata['id']}"
                },
            )
            for row in result
        ]

    async def _resolve_collection_id(self) -> Optional[str]:
        result = await self.db.execute(
            text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
            {"name": self.collection_name},
        )
        row = result.one_or_none()
        return str(row.uuid) if row else None


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
        self.embedding_usage.increment(_count_tokens(query), env.embedding_cost_per_1k_tokens)
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
    files: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional. Restrict search to specific uploaded files by name (case insensitive). "
            "Only set when the user explicitly asks to search within particular file(s); "
            "leave unset for normal queries that should search across all files."
        ),
    )


class DocsTool(AgentToolWithFiles):
    id: str = DOCS_TOOL_ID
    name: str = "Docs"
    description: str = (
        "Use information from uploaded files in agent responses"
    )
    config_schema: dict = load_schema(__file__)
    _embedding_usage: Optional[Usage] = None

    @model_validator(mode="after")
    def remove_advanced_processing_if_not_configured(self):
        if not is_enhanced_pdf_processor_available():
            self.config_schema["properties"].pop(ADVANCED_FILE_PROCESSING)
            self.config_schema["required"].remove(ADVANCED_FILE_PROCESSING)
        return self

    @property
    def embedding_usage(self) -> Usage:
        if self._embedding_usage is None:
            self._embedding_usage = Usage(user_id=self.user_id, agent_id=self.agent.id, model_id=env.embedding_model, type=UsageType.EMBEDDING_TOKENS)
        return self._embedding_usage

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
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
        return PGVector(
            embeddings=ai_provider.build_embedding(
                env.embedding_model,
                request_timeout=DOCS_INGESTION_TIMEOUT_SECONDS,
                max_retries=DOCS_INGESTION_MAX_RETRIES),
            connection=self._get_async_engine(),
            collection_name=self._build_collection_name(self.agent.id),
            use_jsonb=True
        )

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
            docs = self._split_file_content(file_doc)
            self.embedding_usage.increment(sum(_count_tokens(doc.page_content) for doc in docs), env.embedding_cost_per_1k_tokens)
            try:
                await aindex(
                    docs,
                    self._build_record_manager(),
                    self._build_vectorstore(),
                    cleanup="incremental",
                    source_id_key="id",
                    key_encoder="sha256")
            except* Exception as eg:
                if all(ai_factory.get_provider(env.embedding_model).is_rate_limit_error(e) for e in eg.exceptions):
                    raise ModelRateLimitError() from eg
                raise
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

    def _split_file_content(self, file_doc: Document) -> list[Document]:
        text_splitter = MarkdownTextSplitter(
            length_function=_count_tokens,
            chunk_size=env.docs_tool_chunk_size,
            chunk_overlap=env.docs_tool_chunk_overlap)
        return text_splitter.split_documents([file_doc])

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
        llm = ai_factory.build_internal_generator_chat_model(
            model,
            request_timeout=DOCS_INGESTION_TIMEOUT_SECONDS,
            max_retries=DOCS_INGESTION_MAX_RETRIES)
        async with aiofiles.open(solve_asset_path('file-description-prompt.md', __file__)) as f:
            system_prompt = await f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            length_function=lambda text: llm.get_num_tokens(text),
            chunk_size=env.docs_tool_description_chunk_size,
            chunk_overlap=env.docs_tool_description_chunk_overlap)
        chunks = text_splitter.split_text(cast(str, file.processed_content))
        ret = "none"
        for chunk in chunks:
            prompt = system_prompt + f"Previous Description: {ret}\n\n" + f"## File contents\n\n{chunk}"
            ret = await self._generate_description(prompt, 200, llm, model, message_usage)
        return ret

    @staticmethod
    async def _generate_description(prompt: str, max_length: int, llm: BaseChatModel, model: LlmModel, message_usage: MessageUsage) -> str:
        try:
            response = await llm.ainvoke([HumanMessage(prompt)])
        except* Exception as eg:
            if all(ai_factory.get_provider(model.id).is_rate_limit_error(e) for e in eg.exceptions):
                raise ModelRateLimitError() from eg
            raise
        response = cast(AIMessage, response)
        message_usage.increment_with_metadata(response.usage_metadata, model)
        response_msg = response.text
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
        llm = ai_factory.build_internal_generator_chat_model(
            model,
            request_timeout=DOCS_INGESTION_TIMEOUT_SECONDS,
            max_retries=DOCS_INGESTION_MAX_RETRIES)
        return await self._generate_description(prompt, 200, llm, model, message_usage)

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

    async def _run(self, user_query: str, files: Optional[List[str]] = None) -> str:
        available_files = await self._find_available_files()
        file_ids = self._resolve_file_ids(files, available_files)
        if files and file_ids == []:
            available_names = ", ".join(f.name for f in available_files)
            return (
                f"None of the requested files ({', '.join(files)}) were found. "
                f"Available files: {available_names or 'none'}"
            )

        async with aiofiles.open(solve_asset_path("answer-prompt.md", __file__)) as f:
            template = await f.read()
        if self.agent.system_prompt:
            # Escape braces so they stay literal text, not template variables.
            literal_prompt = self.agent.system_prompt.replace("{", "{{").replace("}", "}}")
            template = f"{template}\n\n{literal_prompt}"
        prompt = ChatPromptTemplate.from_template(template)
        llm = ai_factory.build_chat_model(self.agent.model.id, self.agent.model_temperature, self.agent.model_reasoning_effort)
        retriever = self._build_retriever(file_ids)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        config = ensure_config()
        if "callbacks" in config:
            file_names = {str(f.id): f.name for f in available_files}
            cast(AsyncCallbackManager, config["callbacks"]).inheritable_handlers.append(
                DocsStatusUpdateCallbackHandler(self.id, self.description, files or "all", file_names)
            )
        response = await rag_chain.ainvoke(user_query, config=config)
        if self.config.get(RESPONSE_GROUNDING, True):
            get_stream_writer()(
                DocsToolExecutionEvent(
                    action=AgentAction.EXECUTING_TOOL,
                    tool_name=self.id,
                    step=DocsExecutionStep.GROUNDING_RESPONSE,
                )
            )
            response = await self._ground_response(response, retriever, llm)
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
        return response

    def _resolve_file_ids(self, file_names: Optional[List[str]], available_files: List[File]) -> Optional[List[str]]:
        if not file_names:
            return None
        names_by_lower = {f.name.lower(): str(f.id) for f in available_files}
        return [names_by_lower[name.lower()] for name in file_names if name.lower() in names_by_lower]

    def _build_retriever(self, file_ids: Optional[List[str]] = None) -> BaseRetriever:
        if not env.docs_tool_hybrid_search_enabled:
            search_kwargs: dict[str, Any] = {"k": env.docs_tool_retrieve_top}
            if file_ids is not None:
                search_kwargs["filter"] = {"id": {"$in": file_ids}}
            return DocumentUrlSolvingRetriever(
                vectorstore=self._build_vectorstore(),
                search_kwargs=search_kwargs,
                agent_id=self.agent.id,
                tool_id=self.id,
                embedding_usage=self.embedding_usage,
            )
        ai_provider = ai_factory.get_provider(env.embedding_model)
        return HybridSearchRetriever(
            db=self.db,
            collection_name=self._build_collection_name(self.agent.id),
            embeddings=ai_provider.build_embedding(env.embedding_model),
            file_ids=file_ids,
            agent_id=self.agent.id,
            tool_id=self.id,
            embedding_usage=self.embedding_usage,
        )

    @staticmethod
    async def _ground_response(
        response: str, retriever: BaseRetriever, llm: BaseChatModel
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
        available_files = await self._find_available_files()
        description = tool_config.description
        if available_files:
            file_names = ", ".join(f.name for f in available_files)
            description = (
                f"{description}. Available files: {file_names}. "
                f"The optional 'files' argument restricts search to specific files by name "
                f"(case insensitive); only use it when the user explicitly references "
                f"particular file(s), otherwise omit it to search all files."
            )

        return [StructuredTool(
            name=docs_tool.id,
            description=description,
            args_schema=DocsToolArgs,
            coroutine=lambda user_query, files=None: docs_tool._run(user_query, files),
        )]

    async def _find_available_files(self) -> List[File]:
        return await AgentToolConfigFileRepository(self.db).find_by_agent_id_and_tool_id(self.agent.id, self.id)

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
        await self._clone_vector_store(agent_id, cloned_agent_id, db, file_id_map)
        await self._clone_tool_config(agent_id, cloned_agent_id, db)
        await self._clone_tool_files(agent_id, cloned_agent_id, file_id_map, db)

    async def _clone_vector_store(
        self, agent_id: int, cloned_agent_id: int, db: AsyncSession, file_id_map: dict
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
        new_collection_uuid = result.scalar_one_or_none()
        if new_collection_uuid is None or not file_id_map:
            await db.commit()
            return

        await db.execute(
            text("""
                WITH mapped AS (
                    SELECT
                        gen_random_uuid()::varchar AS new_id,
                        e.embedding,
                        e.document,
                        jsonb_set(
                            COALESCE(e.cmetadata, '{}'::jsonb),
                            '{id}',
                            CAST(:file_id_map AS jsonb) -> (e.cmetadata->>'id')
                        ) AS new_cmetadata,
                        CAST(:file_id_map AS jsonb) ->> r.group_id AS new_group_id,
                        r.updated_at
                    FROM langchain_pg_embedding e
                    JOIN upsertion_record r
                      ON r.key = e.id AND r.namespace = :old_namespace
                    WHERE e.collection_id = (
                        SELECT uuid FROM langchain_pg_collection WHERE name = :old_collection
                    )
                      AND CAST(:file_id_map AS jsonb) ->> r.group_id IS NOT NULL
                ),
                inserted_embeddings AS (
                    INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                    SELECT new_id, :new_collection_uuid, embedding, document, new_cmetadata
                    FROM mapped
                    RETURNING id
                )
                INSERT INTO upsertion_record (uuid, key, namespace, group_id, updated_at)
                SELECT gen_random_uuid(), m.new_id, :new_namespace, m.new_group_id, m.updated_at
                FROM mapped m
                JOIN inserted_embeddings ie ON ie.id = m.new_id
                ON CONFLICT (key, namespace) DO NOTHING
            """),
            {
                "new_collection_uuid": new_collection_uuid,
                "old_collection": old_collection,
                "old_namespace": self._build_index_namespace(agent_id),
                "new_namespace": self._build_index_namespace(cloned_agent_id),
                "file_id_map": json.dumps({str(old_id): str(new_id) for old_id, new_id in file_id_map.items()}),
            },
        )
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
    def __init__(self, tool_id: str, description: str, search_files: List[str] | str = "all", file_names: Optional[dict[str, str]] = None):
        self.tool_id = tool_id
        self.description = description
        self.search_files = search_files
        self.file_names = file_names or {}

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
                args={"query": query, "files": self.search_files},
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
                result=[
                    {
                        "content": doc.page_content,
                        "title": self.file_names.get(str(doc.metadata.get("id", "")), ""),
                    }
                    for doc in documents
                ],
            )
        )
