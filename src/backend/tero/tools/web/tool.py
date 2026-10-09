import ast
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import Annotated, Any, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.messages import ToolMessage
from langchain_core.tools import ArgsSchema, BaseTool, InjectedToolCallId
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults
from langchain_tavily import TavilyExtract, TavilySearch
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from ...core.env import env
from ...usage.domain import ToolUsage, UsageType
from ..core import AgentTool, AgentToolConfig, AgentToolMetadata, UserFacingToolError, preview_status_detail, stored_status_result, load_schema, StatusUpdateCallbackHandler


logger = logging.getLogger(__name__)
WEB_TOOL_ID = "web"
SEARCH_NUM_RESULTS = 5


class WebSearchToolArgs(BaseModel):
    query: str = Field(description="The query to search for")
    tool_call_id: Annotated[str, InjectedToolCallId]


def parse_result_search(result: str) -> List[dict]:
    return [
        {"content": e.get("content") or "", "title": e.get("url") or ""}
        for e in ast.literal_eval(result).get("results")
    ]


class WebSearchLangchainTool(BaseTool):
    name: str = "web_search"
    description: str = "Searches the web for the given query"
    args_schema: Optional[ArgsSchema] = WebSearchToolArgs
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description,
        response_parser=parse_result_search)]

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def google_search(self, query: str) -> Tuple[List[Union[str, dict]], ToolUsage]:
        search_api_wrapper = GoogleSearchAPIWrapper(
            google_api_key=env.web_tool_google_api_key.get_secret_value(),
            google_cse_id=env.web_tool_google_custom_search_engine_id,
        )

        search_results = GoogleSearchResults(
            api_wrapper=search_api_wrapper,
            num_results=SEARCH_NUM_RESULTS
        )
        results = await search_results.ainvoke({"query": query})
        tool_usage = ToolUsage(type=UsageType.WEB_SEARCH, quantity=1, cost_per_1k_units=env.web_tool_google_cost_per_1k_searches_usd)
        return results, tool_usage

    async def tavily_search(self, query: str) -> Tuple[List[Union[str, dict]], ToolUsage]:
        tavily_search = TavilySearch(
            tavily_api_key=env.web_tool_tavily_api_key,
            search_depth="basic",
            max_results=SEARCH_NUM_RESULTS
        )
        results = await tavily_search.ainvoke({"query": query})
        tool_usage = ToolUsage(type=UsageType.WEB_SEARCH, quantity=1, cost_per_1k_units=env.web_tool_tavily_cost_per_1k_credits_usd)
        return results, tool_usage

    async def _arun(self, *args: Any, **kwargs: Any) -> ToolMessage:
        query = kwargs.get("query")
        if not query:
            raise UserFacingToolError("Query is required")
        results, tool_usage = await self.tavily_search(query) if env.web_tool_tavily_api_key else await self.google_search(query)
        return ToolMessage(
            tool_call_id=kwargs.get("tool_call_id"),
            name=self.name,
            content=results,
            response_metadata=AgentToolMetadata(tool_usage=tool_usage).model_dump()
        )


class WebExtractToolArgs(BaseModel):
    urls: List[str] = Field(description="The URLs to extract text from")
    tool_call_id: Annotated[str, InjectedToolCallId]


def parse_result_extract(result: Any) -> Any:
    if not isinstance(result, list):
        return stored_status_result(result)
    parsed = []
    for entry in result:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url") or ""
        text = entry.get("raw_content") or entry.get("text") or entry.get("error") or ""
        if not text and not url:
            continue
        parsed.append({"content": text or url, "title": url})
    return parsed


class WebExtractLangchainTool(BaseTool):
    name: str = "web_extract"
    description: str = "Extracts text from the given URLs"
    args_schema: Optional[ArgsSchema] = WebExtractToolArgs
    callbacks: Callbacks = [StatusUpdateCallbackHandler(name, description=description,
        response_parser=parse_result_extract)]

    def __init__(self, max_extract_length: int):
        super().__init__()
        self._max_extract_length = max_extract_length

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Synchronous run not implemented.")

    async def tavily_extract(self, urls: List[str]) -> Tuple[List[Union[str, dict]], ToolUsage]:
        tool = TavilyExtract(
            tavily_api_key=env.web_tool_tavily_api_key,
            extract_depth="basic",
            include_images=False,
        )
        tool_usage = ToolUsage(type=UsageType.WEB_EXTRACT, quantity=1, cost_per_1k_units=env.web_tool_tavily_cost_per_1k_credits_usd)
        results = await tool.ainvoke({"urls": urls})
        if isinstance(results, dict):
            failed_results = results.get("failed_results", [])
            for failed in failed_results:
                if isinstance(failed, dict) and failed.get("error"):
                    failed["error"] = preview_status_detail(str(failed["error"])) or "Error"
            results = results.get("results", [])
            max_extract_length_per_result = int(self._max_extract_length / len(results)) if results else self._max_extract_length
            for result in results:
                raw_content_length = len(result["raw_content"])
                if raw_content_length > max_extract_length_per_result:
                    result["raw_content"] = result["raw_content"][:max_extract_length_per_result]
                    logger.warning(f"Web extract result truncated from {raw_content_length} to {max_extract_length_per_result} characters for {result['url']}")
            results.extend(failed_results)
        return results, tool_usage

    async def bs4_extract(self, urls: List[str]) -> Tuple[List[Union[str, dict]], Optional[ToolUsage]]:
        results = []
        async with httpx.AsyncClient(timeout=10) as client:
            for url in urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    results.append({"url": url, "text": text})
                except Exception as e:
                    results.append({"url": url, "error": self._extract_failure_message(e)})
        return results, None

    def _extract_failure_message(self, error: Exception) -> str:
        if isinstance(error, httpx.HTTPStatusError):
            return f"HTTP {error.response.status_code}"
        if isinstance(error, httpx.TimeoutException):
            return "Request timed out"
        if isinstance(error, httpx.RequestError):
            return "Site unreachable"
        return "Could not extract content"


    async def _arun(self, *args: Any, **kwargs: Any) -> ToolMessage:
        urls = kwargs.get("urls")
        if not urls:
            raise UserFacingToolError("URLs are required")

        results, tool_usage = await self.tavily_extract(urls) if env.web_tool_tavily_api_key else await self.bs4_extract(urls)

        return ToolMessage(
            tool_call_id=kwargs.get("tool_call_id"),
            name=self.name,
            content=results,
            response_metadata=AgentToolMetadata(tool_usage=tool_usage).model_dump()
        )


class WebTool(AgentTool):
    id: str = WEB_TOOL_ID
    name: str = "Web Tools"
    description: str = "Search the web and extract content from public URLs"
    config_schema: dict = load_schema(__file__)

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        pass

    async def teardown(self):
        pass

    @asynccontextmanager
    async def load(self) -> AsyncIterator['WebTool']:
        yield self

    async def clone(
        self,
        agent_id: int,
        cloned_agent_id: int,
        tool_id: str,
        user_id: int,
        db: AsyncSession,
    ) -> None:
        pass

    async def build_langchain_tools(self) -> List[BaseTool]:
        tools = []
        if env.web_tool_google_api_key and env.web_tool_google_custom_search_engine_id or env.web_tool_tavily_api_key:
            tools.append(WebSearchLangchainTool())
        # Set max characters for web extraction to half of the available input tokens to prevent overflow
        tools.append(WebExtractLangchainTool(max_extract_length=int((self.agent.model.token_limit) / 2)))
        return tools
