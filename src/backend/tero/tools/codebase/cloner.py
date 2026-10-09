import asyncio
import logging
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

from ...core.domain import CamelCaseModel
from ...core.env import env

logger = logging.getLogger(__name__)

_CREDENTIALS_PATTERN = re.compile(r'://[^@]+@')
GIT_SAFE_DIRECTORY = f'safe.directory={Path(env.codebase_tool_repos_path).resolve()}/*'


class CodebaseToolConfigMetadata(CamelCaseModel):
    commit_id: str
    synced_at: datetime


class RepoManager:
    _replace_locks: dict[int, asyncio.Lock] = {}

    def __init__(self, repo_url: str, access_token: Optional[str] = None, ref: Optional[str] = None):
        self.repo_url = repo_url
        self.access_token = access_token
        self.ref = ref

    async def clone(self, agent_id: int) -> CodebaseToolConfigMetadata:
        repo_path = self.get_repo_path(agent_id)
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(tempfile.mkdtemp(prefix=f'{repo_path.name}.', dir=repo_path.parent))

        url = self._build_clone_url()
        cmd = ['clone', '--depth', '1']
        if self.ref:
            cmd += ['--branch', self.ref]
        cmd += [url, str(tmp_path)]
        try:
            await self._run_git(cmd)
            commit_info = await self._get_commit_info(tmp_path)
            await self._strip_remote_credentials(tmp_path)
            async with self._replace_lock(agent_id):
                if repo_path.exists():
                    await asyncio.to_thread(shutil.rmtree, repo_path)
                tmp_path.rename(repo_path)
            return commit_info
        except Exception:
            await asyncio.to_thread(shutil.rmtree, tmp_path, ignore_errors=True)
            raise

    def _replace_lock(self, agent_id: int) -> asyncio.Lock:
        lock = self._replace_locks.get(agent_id)
        if lock is None:
            lock = asyncio.Lock()
            self._replace_locks[agent_id] = lock
        return lock

    async def sync(self, agent_id: int) -> CodebaseToolConfigMetadata:
        repo_path = self.get_repo_path(agent_id)
        if not repo_path.exists():
            return await self.clone(agent_id)

        try:
            await self._set_remote_url(repo_path, self._build_clone_url())
            if self.ref:
                await self._run_git(
                    ['fetch', 'origin', self.ref, '--depth', '1', '--force'],
                    cwd=str(repo_path),
                )
                await self._run_git(['checkout', '--detach', 'FETCH_HEAD'], cwd=str(repo_path))
            else:
                await self._run_git(['pull', '--ff-only'], cwd=str(repo_path))
            return await self._get_commit_info(repo_path)
        except RepoCloneError:
            return await self.clone(agent_id)
        finally:
            await self._strip_remote_credentials(repo_path)

    def delete(self, agent_id: int):
        repo_path = self.get_repo_path(agent_id)
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

    def get_repo_path(self, agent_id: int) -> Path:
        return Path(env.codebase_tool_repos_path) / str(agent_id)

    async def is_work_tree(self, agent_id: int) -> bool:
        repo_path = self.get_repo_path(agent_id)
        if not repo_path.is_dir():
            return False
        try:
            await self._run_git(['rev-parse', '--git-dir'], cwd=str(repo_path))
            return True
        except RepoCloneError:
            return False

    async def _get_commit_info(self, repo_path: Path) -> CodebaseToolConfigMetadata:
        proc = await asyncio.create_subprocess_exec(
            'git', '-c', GIT_SAFE_DIRECTORY, 'log', '-1', '--format=%H',
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return CodebaseToolConfigMetadata(
            commit_id=stdout.decode().strip(),
            synced_at=datetime.now(timezone.utc),
        )

    async def _strip_remote_credentials(self, repo_path: Path):
        if not self.access_token or not repo_path.exists():
            return
        await self._set_remote_url(repo_path, self.repo_url)

    async def _set_remote_url(self, repo_path: Path, url: str):
        await self._run_git(['remote', 'set-url', 'origin', url], cwd=str(repo_path))

    async def _run_git(self, args: list[str], *, cwd: Optional[str] = None, timeout: int = 300) -> bytes:
        process = await asyncio.create_subprocess_exec(
            'git', '-c', GIT_SAFE_DIRECTORY, '-c', 'credential.helper=', '-c', 'credential.interactive=false', *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RepoCloneError("Operation timed out. The repository may be too large or the server is not responding.")

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            if 'Authentication' in error_msg or 'could not read' in error_msg or 'terminal prompts disabled' in error_msg or 'unable to get password' in error_msg:
                raise RepoAuthenticationError(
                    "Authentication failed. Verify the access token has read access to the repository."
                )
            if 'not found' in error_msg.lower() or 'does not exist' in error_msg.lower() or 'Repository not found' in error_msg:
                raise RepoNotFoundError(
                    "Repository not found. Verify the URL is correct."
                )
            sanitized = _CREDENTIALS_PATTERN.sub('://***@', error_msg)
            logger.error(f"Git operation failed: {sanitized}")
            raise RepoCloneError("Git operation failed. Verify the URL and try again.")

        return stdout

    def _build_clone_url(self) -> str:
        if not self.access_token:
            return self.repo_url
        parsed = urlparse(self.repo_url)
        authed = parsed._replace(
            netloc=f"x-access-token:{self.access_token}@{parsed.hostname}"
            + (f":{parsed.port}" if parsed.port else "")
        )
        return urlunparse(authed)


class RepoAuthenticationError(Exception):
    pass


class RepoNotFoundError(Exception):
    pass


class RepoCloneError(Exception):
    pass
