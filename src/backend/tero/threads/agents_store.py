import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import wcmatch.glob as wcglob
from deepagents.backends import StoreBackend
from deepagents.backends.protocol import FileData, FileInfo, GlobResult, GrepResult, LsResult
from deepagents.backends.utils import _glob_search_files, _normalize_path, grep_matches_from_files
from fastapi import HTTPException, status
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PoolConfig
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.abc import QueryNoTemplate

from ..core.repos import plain_postgresql_url

_store = None

_FILE_CONTENT_SQL = "COALESCE(value->>'content', '')"
_VALID_FILE_SQL = "jsonb_typeof(value->'content') = 'string'"
MEMORY_ROUTE = "/memory/"
MEMORY_PATH = f"{MEMORY_ROUTE}AGENTS.md"
MEMORY_STORE_KEY = "/AGENTS.md"


def _user_store_namespace_prefix(user_id: int) -> tuple[str, ...]:
    return (f"user_{user_id}",)


def _build_store_namespace(user_id: int, scope: str, scope_id: int, namespace: str) -> tuple[str, ...]:
    return (*_user_store_namespace_prefix(user_id), f"{scope}_{scope_id}", namespace)


def agent_store_namespace(user_id: int, thread_id: int) -> tuple[str, ...]:
    return _build_store_namespace(user_id, "thread", thread_id, "fs")


def agent_memory_namespace(user_id: int, agent_id: int) -> tuple[str, ...]:
    return _build_store_namespace(user_id, "agent", agent_id, "memory")


def agent_test_case_memory_namespace_prefix(
    user_id: int,
    suite_run_id: int,
    test_case_id: int,
) -> tuple[str, ...]:
    return (
        *_user_store_namespace_prefix(user_id),
        f"test_suite_{suite_run_id}",
        f"test_case_{test_case_id}",
    )


def scoped_agent_memory_namespace(prefix: tuple[str, ...], agent_id: int) -> tuple[str, ...]:
    return (*prefix, f"agent_{agent_id}", "memory")


async def ensure_memory_file(store: BaseStore, namespace: tuple[str, ...]) -> None:
    if await store.aget(namespace, MEMORY_STORE_KEY) is None:
        await store.aput(
            namespace,
            MEMORY_STORE_KEY,
            {"content": "# Agent Memory\n", "encoding": "utf-8"},
        )


async def clear_store_namespace(store: BaseStore, namespace: tuple[str, ...]) -> None:
    while True:
        items = await store.asearch(namespace, limit=100)
        if not items:
            return
        for item in items:
            await store.adelete(tuple(item.namespace), item.key)


@dataclass(frozen=True)
class _FileMeta:
    key: str
    size: int
    modified_at: str


class AgentsPostgresStore(AsyncPostgresStore):
    async def setup(self) -> None:
        await super().setup()
        async with self._cursor() as cur:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            await cur.execute(
                """
                CREATE INDEX IF NOT EXISTS store_value_content_trgm_idx
                ON store USING gin ((value->>'content') gin_trgm_ops)
                """
            )


class AgentStoreBackend(StoreBackend):
    async def als(self, path: str) -> LsResult:
        store = self._get_store()
        if isinstance(store, AsyncPostgresStore):
            metas = await self._alist_file_meta(store)
            return _ls_from_meta(path, metas)
        return await asyncio.to_thread(super().ls, path)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        store = self._get_store()
        if isinstance(store, AsyncPostgresStore):
            return await self._aglob_sql(store, pattern, path)
        return await asyncio.to_thread(super().glob, pattern, path)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        store = self._get_store()
        if isinstance(store, AsyncPostgresStore):
            return await self._agrep_sql(store, pattern, path, glob, max_count=max_count)
        return await self._agrep_paginated(store, pattern, path, glob, max_count=max_count)

    async def _alist_file_meta(self, store: AsyncPostgresStore) -> list[_FileMeta]:
        sql = f"""
            SELECT key,
                   CASE
                       WHEN jsonb_typeof(value->'modified_at') = 'string' THEN value->>'modified_at'
                       ELSE ''
                   END AS modified_at,
                   length({_FILE_CONTENT_SQL}) AS size
            FROM store
            WHERE prefix = %s
              AND {_VALID_FILE_SQL}
            ORDER BY updated_at DESC
        """
        async with store._cursor() as cur:
            await cur.execute(sql, [".".join(self._get_namespace())])
            rows = await cur.fetchall()
        return [
            _FileMeta(key=row["key"], size=int(row["size"]), modified_at=row["modified_at"] or "")
            for row in rows
        ]

    async def _aglob_sql(self, store: AsyncPostgresStore, pattern: str, path: str | None) -> GlobResult:
        metas = await self._alist_file_meta(store)
        files: dict[str, FileData] = {
            meta.key: FileData(content="", encoding="utf-8", modified_at=meta.modified_at)
            for meta in metas
        }
        size_by_key = {meta.key: meta.size for meta in metas}
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return GlobResult(matches=[])
        return GlobResult(
            matches=[
                FileInfo(
                    path=file_path,
                    is_dir=False,
                    size=size_by_key.get(file_path, 0),
                    modified_at=files[file_path].get("modified_at", "") if file_path in files else "",
                )
                for file_path in result.split("\n")
            ]
        )

    async def _agrep_sql(
        self,
        store: AsyncPostgresStore,
        pattern: str,
        path: str | None,
        glob: str | None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        try:
            normalized_path = _normalize_path(path)
        except ValueError:
            return GrepResult(matches=[])

        like_pattern = _like_contains(pattern)
        conditions = [
            "prefix = %s",
            _VALID_FILE_SQL,
            "value->>'content' LIKE %s ESCAPE '\\'",
        ]
        params: list[str] = [".".join(self._get_namespace()), like_pattern]
        if normalized_path != "/":
            conditions.append("(key = %s OR key LIKE %s ESCAPE '\\')")
            params.extend([normalized_path, f"{_like_escape(normalized_path)}/%"])

        sql = f"""
            SELECT key, {_FILE_CONTENT_SQL} AS content
            FROM store
            WHERE {' AND '.join(conditions)}
        """
        async with store._cursor() as cur:
            await cur.execute(cast(QueryNoTemplate, sql), params)
            rows = await cur.fetchall()

        files: dict[str, FileData] = {}
        for row in rows:
            key = row["key"]
            if glob and not wcglob.globmatch(Path(key).name, glob, flags=wcglob.BRACE):
                continue
            files[key] = FileData(content=row["content"] or "", encoding="utf-8")
        return grep_matches_from_files(files, pattern, path, glob, max_count=max_count)

    async def _agrep_paginated(
        self,
        store: BaseStore,
        pattern: str,
        path: str | None,
        glob: str | None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        items = await super()._asearch_store_paginated(store, self._get_namespace())
        files = {}
        for item in items:
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        return grep_matches_from_files(files, pattern, path, glob, max_count=max_count)


def _ls_from_meta(path: str, metas: list[_FileMeta]) -> LsResult:
    infos: list[FileInfo] = []
    subdirs: set[str] = set()
    normalized_path = path if path.endswith("/") else path + "/"

    for meta in metas:
        if not meta.key.startswith(normalized_path):
            continue
        relative = meta.key[len(normalized_path):]
        if "/" in relative:
            subdirs.add(normalized_path + relative.split("/")[0] + "/")
            continue
        infos.append(
            {
                "path": meta.key,
                "is_dir": False,
                "size": meta.size,
                "modified_at": meta.modified_at,
            }
        )

    infos.extend(FileInfo(path=subdir, is_dir=True, size=0, modified_at="") for subdir in sorted(subdirs))
    infos.sort(key=lambda item: item.get("path", ""))
    return LsResult(entries=infos)


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _like_contains(pattern: str) -> str:
    return f"%{_like_escape(pattern)}%"


@asynccontextmanager
async def agents_store_lifespan():
    global _store
    async with AgentsPostgresStore.from_conn_string(
        plain_postgresql_url(),
        pool_config=PoolConfig(min_size=1, max_size=10),
    ) as store:
        await store.setup()
        _store = store
        try:
            yield
        finally:
            _store = None


def get_store() -> BaseStore:
    if _store is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Store is not configured",
        )
    return _store
