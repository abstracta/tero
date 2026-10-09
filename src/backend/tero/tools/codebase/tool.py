import asyncio
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from ...agents.domain import AgentToolConfig
from ...agents.repos import AgentToolConfigRepository
from ..auth import ToolAuthRepository, ToolAuthToken, ToolAuthTokenType
from ..core import AgentTool, UserFacingToolError, load_schema, StatusUpdateCallbackHandler
from .cloner import RepoManager, RepoAuthenticationError, RepoNotFoundError, RepoCloneError, GIT_SAFE_DIRECTORY


CODEBASE_TOOL_ID = "codebase"
MAX_SEARCH_RESULTS = 50
MAX_FILE_LINES = 500


class SearchCodeArgs(BaseModel):
    query: str = Field(description="Text pattern to search for in the codebase")
    file_pattern: Optional[str] = Field(
        default=None, description="Glob pattern to filter files (e.g. '*.py', 'src/**/*.ts')"
    )


class ReadFileArgs(BaseModel):
    path: str = Field(description="Relative file path within the repository")
    start_line: Optional[int] = Field(default=None, description="Start reading from this line number (1-based)")
    end_line: Optional[int] = Field(default=None, description="Stop reading at this line number (inclusive)")


class ListFilesArgs(BaseModel):
    path: Optional[str] = Field(default="", description="Directory path relative to repository root")


class CodebaseNotSyncedError(Exception):
    pass


class CodebaseTool(AgentTool):
    id: str = CODEBASE_TOOL_ID
    name: str = "Codebase"
    description: str = "Connect to a repository"
    config_schema: dict = load_schema(__file__)
    _token_secret_key = "accessToken"

    @asynccontextmanager
    async def load(self) -> AsyncIterator['CodebaseTool']:
        yield self

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        manager = RepoManager(repo_url="")
        src = manager.get_repo_path(agent_id)
        dst = manager.get_repo_path(cloned_agent_id)
        if src.exists():
            await asyncio.to_thread(shutil.copytree, src, dst)

    async def teardown(self):
        await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)
        RepoManager(repo_url="").delete(self.agent.id)

    async def sync(self) -> Optional[dict]:
        token_record = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        token = token_record.access_token if token_record else None
        manager = RepoManager(
            repo_url=self.config.get('repositoryUrl', ''), access_token=token, ref=self.config.get('ref')
        )
        try:
            commit_info = await manager.sync(self.agent.id)
            metadata = commit_info.model_dump(by_alias=True, mode='json')
            self._tool_metadata = metadata
            await AgentToolConfigRepository(self.db).add(AgentToolConfig(
                agent_id=self.agent.id,
                tool_id=self.id,
                config=self.config,
                tool_metadata=metadata,
            ))
            return metadata
        except (RepoAuthenticationError, RepoCloneError, RepoNotFoundError) as e:
            raise ValueError(str(e)) from e

    async def build_langchain_tools(self) -> List[BaseTool]:
        manager = RepoManager(repo_url="")
        repo_path = manager.get_repo_path(self.agent.id)
        if not await manager.is_work_tree(self.agent.id):
            if self.config.get('repositoryUrl'):
                raise CodebaseNotSyncedError()
            return []

        return [
            StructuredTool(
                name="codebase_search",
                description="Search for text patterns in the repository code. Returns matching lines with file paths and line numbers.",
                args_schema=SearchCodeArgs,
                coroutine=lambda query, file_pattern=None: self._search_code(repo_path, query, file_pattern),
                callbacks=[StatusUpdateCallbackHandler("codebase_search")],
            ),
            StructuredTool(
                name="codebase_read",
                description="Read a file from the repository. Can read specific line ranges.",
                args_schema=ReadFileArgs,
                coroutine=lambda path, start_line=None, end_line=None: self._read_file(
                    repo_path, path, start_line, end_line
                ),
                callbacks=[StatusUpdateCallbackHandler("codebase_read")],
            ),
            StructuredTool(
                name="codebase_list",
                description="List files in the repository. Optionally filter by directory path.",
                args_schema=ListFilesArgs,
                coroutine=lambda path="": self._list_files(repo_path, path),
                callbacks=[StatusUpdateCallbackHandler("codebase_list")],
            ),
        ]

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        prev_token = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        provided_token = self._get_secret(self._token_secret_key)
        access_token = provided_token
        config_changed = bool(prev_config and prev_config.config != self.config)
        token_changed = bool(prev_token and access_token is not None and access_token != prev_token.access_token)
        if access_token is None and prev_token:
            access_token = prev_token.access_token

        manager = RepoManager(
            repo_url=self.config.get('repositoryUrl', ''), access_token=access_token, ref=self.config.get('ref')
        )
        try:
            if (
                manager.get_repo_path(self.agent.id).exists()
                and prev_config
                and prev_config.tool_metadata
                and not config_changed
                and not token_changed
            ):
                self._tool_metadata = prev_config.tool_metadata
                return
            commit_info = await manager.clone(self.agent.id)
            self._tool_metadata = commit_info.model_dump(by_alias=True, mode='json')
            if provided_token is not None and not provided_token:
                await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)
            else:
                await self._save_token(access_token)
        except (RepoAuthenticationError, RepoCloneError, RepoNotFoundError) as e:
            raise ValueError(str(e)) from e

    async def _save_token(self, token: Optional[str]):
        if not token:
            return
        await ToolAuthRepository(self.db).save_token(
            ToolAuthToken(
                user_id=self.user_id,
                agent_id=self.agent.id,
                tool_id=self.id,
                access_token=token,
                token_type=ToolAuthTokenType.BEARER,
                expires_in=None,
                scope=None,
                refresh_token=None,
                expires_at=None,
            )
        )

    async def _search_code(self, repo_path: Path, query: str, file_pattern: Optional[str] = None) -> str:
        cmd = ['git', '-c', GIT_SAFE_DIRECTORY, 'grep', '-i', '-n', '--no-color', '-I', '-e', query]
        if file_pattern:
            glob = file_pattern if '/' in file_pattern else f'**/{file_pattern}'
            cmd += ['--', f':(icase,glob){glob}']

        process = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(repo_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode and process.returncode > 1:
            detail = stderr.decode(errors='replace').strip()
            raise RuntimeError(detail or "git grep failed")

        if not stdout:
            return "No matches found."

        lines = stdout.decode(errors='replace').splitlines()
        if len(lines) > MAX_SEARCH_RESULTS:
            lines = lines[:MAX_SEARCH_RESULTS]
            lines.append(f"\n... truncated ({MAX_SEARCH_RESULTS} results shown)")

        return "\n".join(lines)

    async def _read_file(
        self, repo_path: Path, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
    ) -> str:
        file_path = (repo_path / path).resolve()
        try:
            file_path.relative_to(repo_path.resolve())
        except ValueError:
            raise UserFacingToolError("Path traversal not allowed.")
        if not file_path.is_file():
            raise UserFacingToolError(f"File not found: {path}")

        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            raise UserFacingToolError("Binary file, cannot read as text.")

        lines = content.splitlines()
        total = len(lines)

        if start_line or end_line:
            start = max((start_line or 1) - 1, 0)
            end = min(end_line or total, total)
            lines = lines[start:end]
            header = f"Lines {start + 1}-{end} of {total} in {path}:\n"
        elif total > MAX_FILE_LINES:
            lines = lines[:MAX_FILE_LINES]
            header = f"First {MAX_FILE_LINES} of {total} lines in {path} (use start_line/end_line to read more):\n"
        else:
            header = f"{path} ({total} lines):\n"

        numbered = [f"{i + (start_line or 1)}  {line}" for i, line in enumerate(lines)]
        return header + "\n".join(numbered)

    async def _list_files(self, repo_path: Path, path: Optional[str] = None) -> str:
        cmd = ['git', '-c', GIT_SAFE_DIRECTORY, 'ls-files']
        if path:
            cmd += ['--', path]

        process = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(repo_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            detail = stderr.decode(errors='replace').strip()
            raise RuntimeError(detail or "git ls-files failed")

        if not stdout:
            return "No files found."

        files = stdout.decode(errors='replace').splitlines()
        return "\n".join(files)
