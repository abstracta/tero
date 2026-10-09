from typing import Any

import pytest
from deepagents.backends import StoreBackend
from deepagents.backends.protocol import FileInfo, GrepMatch
from langgraph.store.postgres import PoolConfig

from deepagents.backends.utils import format_grep_matches

from tero.threads.agents_store import AgentStoreBackend, AgentsPostgresStore


def _store_url(postgres_container) -> str:
    return postgres_container.get_connection_url().replace("postgresql+psycopg://", "postgresql://")


def _eq_namespace(_: Any) -> tuple[str, ...]:
    return ("eq", "user_1", "thread_1", "fs")


@pytest.fixture(name="grep_store")
async def grep_store_fixture(postgres_container):
    async with AgentsPostgresStore.from_conn_string(
        _store_url(postgres_container),
        pool_config=PoolConfig(min_size=1, max_size=5),
    ) as store:
        await store.setup()
        yield store


def _entry_tuple(entry: FileInfo) -> tuple:
    return (entry.get("path"), entry.get("is_dir"), entry.get("size"), entry.get("modified_at", ""))


def _match_tuple(match: GrepMatch) -> tuple:
    return (match["path"], match["line"], match["text"])


def _paths(entries: list[FileInfo] | None) -> list[str]:
    return [entry.get("path", "") for entry in (entries or [])]


async def test_agent_store_matches_stock_ls_glob_grep(grep_store):
    stock = StoreBackend(store=grep_store, namespace=_eq_namespace)
    optimized = AgentStoreBackend(store=grep_store, namespace=_eq_namespace)
    files = {
        "/docs/readme.md": "alpha\nneedle here\nalpha\n",
        "/src/main.py": "import os\nprint('needle')\n",
        "/src/pkg/util.py": "needle\nhelper\n",
        "/data/rows.csv": "a,b\n1,2\n",
        "/empty.txt": "",
        "/specials.txt": "100%_done\nfoo_bar\n",
    }
    for path, content in files.items():
        result = await optimized.awrite(path, content)
        assert result.error is None, result.error

    await grep_store.aput(
        _eq_namespace(None),
        "/upload",
        {"content": "uploaded body\nneedle\n", "encoding": "utf-8"},
    )
    await grep_store.aput(
        _eq_namespace(None),
        "/skip-me",
        {"encoding": "utf-8"},
    )

    for path in ("/", "/src", "/src/pkg", "/missing"):
        stock_ls = await stock.als(path)
        ours_ls = await optimized.als(path)
        assert [_entry_tuple(e) for e in (stock_ls.entries or [])] == [
            _entry_tuple(e) for e in (ours_ls.entries or [])
        ]
        assert _paths(stock_ls.entries) == _paths(ours_ls.entries)

    for pattern, path in (
        ("**/*.py", "/"),
        ("*.py", "/"),
        ("*.py", "/src"),
        ("**/*.md", "/"),
        ("*.txt", "/"),
    ):
        stock_glob = await stock.aglob(pattern, path=path)
        ours_glob = await optimized.aglob(pattern, path=path)
        assert [_entry_tuple(e) for e in (stock_glob.matches or [])] == [
            _entry_tuple(e) for e in (ours_glob.matches or [])
        ]
        assert _paths(stock_glob.matches) == _paths(ours_glob.matches)

    for pattern, path, glob in (
        ("needle", "/", None),
        ("needle", "/src", "*.py"),
        ("alpha", "/", None),
        ("missing", "/", None),
        ("needle", "/src/main.py", None),
        ("%", "/", None),
        ("_", "/", None),
        ("foo_bar", "/", None),
        ("needle\nhelper", "/", None),
        ("", "/empty.txt", None),
    ):
        stock_grep = await stock.agrep(pattern, path, glob)
        ours_grep = await optimized.agrep(pattern, path, glob)
        assert sorted(_match_tuple(m) for m in (stock_grep.matches or [])) == sorted(
            _match_tuple(m) for m in (ours_grep.matches or [])
        )
        for output_mode in ("files_with_matches", "content", "count"):
            assert format_grep_matches(stock_grep.matches or [], output_mode) == format_grep_matches(
                ours_grep.matches or [], output_mode
            )

    capped = await optimized.agrep(pattern="needle", path="/", max_count=1)
    assert capped.error is None
    assert len(capped.matches or []) == 1
    assert capped.truncated is True
