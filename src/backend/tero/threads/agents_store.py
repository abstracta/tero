from contextlib import asynccontextmanager

from fastapi import HTTPException, status
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PoolConfig
from langgraph.store.postgres.aio import AsyncPostgresStore

from ..core.repos import plain_postgresql_url

_store = None


@asynccontextmanager
async def agents_store_lifespan():
    global _store
    async with AsyncPostgresStore.from_conn_string(
        plain_postgresql_url(),
        pool_config=PoolConfig(min_size=1, max_size=5),
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
