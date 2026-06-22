import abc
import aiofiles
import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPMethod
from typing import Any, ClassVar, Optional, cast
from urllib.parse import quote

import httpx
from httpx import HTTPStatusError
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import AnyHttpUrl
from sqlmodel.ext.asyncio.session import AsyncSession

from ..agents.domain import AgentToolConfig
from ..core.assets import solve_asset_path
from .auth import (
    AgentToolOauth,
    OAuthMetadata,
    ToolAuthCallback,
    ToolAuthCallbackError,
    ToolAuthRepository,
    ToolAuthRequestException,
    ToolAuthToken,
    ToolAuthTokenCallback,
    ToolAuthTokenRequest,
    ToolAuthTokenType,
    ToolOAuthCallback,
    ToolOAuthClientInfo,
    ToolOAuthClientInfoRepository,
)
from .core import AgentTool, StatusUpdateCallbackHandler


class OpenApiTool(AgentTool, abc.ABC):
    _BODY_LOCATION = "body"
    _body_content_types = ("application/json",)
    _api_url: str = ""

    async def build_langchain_tools(self) -> list[BaseTool]:
        api_spec = await self._load_api_spec()
        schemas = api_spec.get("components", {}).get("schemas", {})
        return [
            self._build_langchain_tool(path, method, method_spec, schemas)
            for path, path_spec in api_spec.get("paths", {}).items()
            for method, method_spec in path_spec.items() if self._should_include_operation(path, method)
        ]

    async def _load_api_spec(self) -> dict:
        tool_id = self.id.split("-", 1)[0]
        # Use local files to avoid fetching the spec every run.
        return await self._load_json(f"{tool_id}-api-spec.json")

    async def _load_json(self, filename: str) -> dict:
        async with aiofiles.open(solve_asset_path(filename, inspect.getfile(self.__class__))) as file:
            return json.loads(await file.read())

    def _should_include_operation(self, path: str, method: str) -> bool:
        return True

    def _operation_description(self, method: str, path: str, method_spec: dict) -> str:
        return (
            method_spec.get("description")
            or method_spec.get("summary")
            or f"{method.upper()} {path}"
        )

    def _build_langchain_tool(self, path: str, method: str, method_spec: dict, schemas: dict) -> BaseTool:
        body_content_type = self._find_body_content_type(method_spec)

        async def call_tool(**arguments: dict[str, Any]) -> str:
            param_type = self._find_unique_parameter_type(method_spec)
            params = {param_type: arguments} if param_type else arguments
            path_params = {key: quote(str(value)) for key, value in params.get("path", {}).items()}
            final_path = path.format(**path_params) if path_params else path
            return await self._invoke_rest_api(
                method,
                f"{self._api_url}{final_path}",
                params.get("query"),
                params.get("header"),
                params.get("body"),
                body_content_type,
            )

        name = f"{self.name}-{method_spec['operationId']}".replace(" ", "")
        description = f"{self.name} tool that {self._operation_description(method, path, method_spec)}"
        return StructuredTool(
            name=name,
            description=description,
            args_schema=self._build_args_schema(method_spec, schemas),
            coroutine=call_tool,
            callbacks=[StatusUpdateCallbackHandler(name, description=description)],
        )

    async def _invoke_rest_api(
        self,
        method: str,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        body: Optional[Any] = None,
        body_content_type: Optional[str] = None,
    ) -> Any:
        headers = headers or {}
        if body is not None:
            headers["Content-Type"] = body_content_type or "application/json"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                method,
                url,
                params=params,
                headers=await self._add_auth_headers(headers),
                content=json.dumps(body) if body is not None else None,
            )
            try:
                response.raise_for_status()
            except HTTPStatusError as e:
                if e.response.content:
                    raise HTTPStatusError(
                        f"{e}. Response body: {e.response.text}",
                        request=e.request,
                        response=e.response,
                    ) from e
                raise
            if response.status_code == 204:
                return None
            return response.json() if response.content else None

    @abc.abstractmethod
    async def _add_auth_headers(self, headers: dict) -> dict:
        raise NotImplementedError

    def _find_unique_parameter_type(self, method_spec: dict) -> Optional[str]:
        ret = None
        for param in method_spec.get("parameters", []):
            location = param["in"]
            if ret and ret != location:
                return None
            ret = location
        body_schema = self._find_body_schema(method_spec)
        if ret and body_schema:
            return None
        return ret if not body_schema else self._BODY_LOCATION

    def _find_body_content_type(self, method_spec: dict) -> Optional[str]:
        content = method_spec.get("requestBody", {}).get("content", {})
        return next((content_type for content_type in self._body_content_types if content_type in content), None)

    def _find_body_schema(self, method_spec: dict) -> Optional[dict]:
        content_type = self._find_body_content_type(method_spec)
        if not content_type:
            return None
        return method_spec.get("requestBody", {}).get("content", {}).get(content_type, {}).get("schema")

    def _build_args_schema(self, method_spec: dict, schemas: dict) -> dict[str, Any]:
        ret = self._build_params_schema(method_spec)
        body_schema = self._find_body_schema(method_spec)
        props = ret["properties"]
        if body_schema:
            props[self._BODY_LOCATION] = body_schema
        input_schemas = [schema for schema in props.values() if schema]
        ret = input_schemas[0] if len(input_schemas) == 1 else ret
        self._refactor_schema_refs(ret, schemas)
        return ret

    def _build_params_schema(self, method_spec: dict) -> dict:
        ret = self._build_empty_schema()
        props = ret["properties"]
        for param in method_spec.get("parameters", []):
            location = param["in"]
            props[location] = props.get(location, self._build_empty_schema())
            location_params = props[location]
            name = param["name"]
            param_schema = param["schema"]
            description = param.get("description")
            if description:
                param_schema["description"] = description
            location_params["properties"][name] = param_schema
            if param.get("required"):
                location_params["required"].append(name)
        return ret

    def _build_empty_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def _refactor_schema_refs(self, schema: dict, schemas: dict):
        refs = set()
        self._collect_and_refactor_schema_refs(schema, schemas, refs)
        if refs:
            schema["$defs"] = {ref: schemas[ref] for ref in refs}

    def _collect_and_refactor_schema_refs(self, schema: dict, schemas: dict, refs: set):
        ref = schema.get("$ref")
        if ref:
            self._refactor_ref(schema, ref.split("/")[-1], schemas, refs)
        self._refactor_subschemas_refs("allOf", schema, schemas, refs)
        self._refactor_subschemas_refs("anyOf", schema, schemas, refs)
        self._refactor_subschemas_refs("oneOf", schema, schemas, refs)
        schema_type = schema.get("type")
        if not schema_type:
            self._handle_schema_without_type(schema, schemas, refs)
        elif schema_type == "array":
            items = schema.get("items")
            if items:
                self._collect_and_refactor_schema_refs(items, schemas, refs)
        for value in schema.get("properties", {}).values():
            self._collect_and_refactor_schema_refs(value, schemas, refs)
        # removing additional properties to simplify schema since so far we haven't identified any use case for them when used by the llm
        if schema.get("additionalProperties"):
            del schema["additionalProperties"]

    def _handle_schema_without_type(self, schema: dict, schemas: dict, refs: set) -> None:
        # Hook for subclasses to handle special schemas lacking a type.
        return None

    def _refactor_ref(self, schema: dict, simple_ref: str, schemas: dict, refs: set):
        schema["$ref"] = f"#/$defs/{simple_ref}"
        # passing refs as parameter and modify it instead of returning it to be able to make this check to avoid infinite recursion in cyclic references
        if simple_ref not in refs:
            refs.add(simple_ref)
            self._collect_and_refactor_schema_refs(schemas[simple_ref], schemas, refs)

    def _refactor_subschemas_refs(self, subschema_key: str, schema: dict, schemas: dict, refs: set):
        for sub_schema in schema.get(subschema_key, []):
            self._collect_and_refactor_schema_refs(sub_schema, schemas, refs)


@dataclass(frozen=True)
class OAuthToolConfig:
    authority_base_url: str
    authorize_path: str
    token_path: str
    scope: str

    @property
    def metadata(self) -> OAuthMetadata:
        return OAuthMetadata(
            issuer=AnyHttpUrl(self.authority_base_url),
            authorization_endpoint=AnyHttpUrl(f"{self.authority_base_url}{self.authorize_path}"),
            token_endpoint=AnyHttpUrl(f"{self.authority_base_url}{self.token_path}"),
        )


class OAuthOpenApiTool(OpenApiTool, abc.ABC):
    _oauth: Optional[AgentToolOauth] = None
    _client_secret_config_key: ClassVar[str] = "clientSecret"
    _client_id_config_key: ClassVar[str] = "clientId"

    @abc.abstractmethod
    def _oauth_config(self) -> OAuthToolConfig:
        raise NotImplementedError

    @abc.abstractmethod
    async def _resolve_api_url(self) -> str:
        raise NotImplementedError

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        client_info_repo = ToolOAuthClientInfoRepository(self.db)
        prev_client_info = await client_info_repo.find_by_ids(self.agent.id, self.id)
        client_secret = self._get_secret(self._client_secret_config_key)
        client_id = self.config[self._client_id_config_key]
        if (
            prev_config and prev_config.config != self.config
            or prev_client_info and client_secret and prev_client_info.client_secret != client_secret
        ):
            if not client_secret and prev_client_info and prev_client_info.client_id == client_id:
                client_secret = prev_client_info.client_secret
            await self.teardown()
        if client_secret:
            await client_info_repo.save(
                ToolOAuthClientInfo(
                    agent_id=self.agent.id,
                    tool_id=self.id,
                    client_id=client_id,
                    client_secret=client_secret,
                    token_endpoint_auth_method="client_secret_post",
                    scope=self._oauth_config().scope,
                )
            )
        async with self.load():
            pass

    @asynccontextmanager
    async def load(self) -> AsyncIterator["OAuthOpenApiTool"]:
        self._oauth = await self._load_oauth()
        await self._oauth.solve_tokens()
        self._api_url = await self._resolve_api_url()
        yield self

    async def _add_auth_headers(self, headers: dict) -> dict:
        tokens = await cast(AgentToolOauth, self._oauth).solve_tokens()
        if tokens:
            headers["Authorization"] = f"Bearer {tokens.access_token}"
        return headers

    async def auth(self, auth_callback: ToolAuthCallback):
        state = await ToolAuthRepository(self.db).find_state(
            self.user_id, self.id, cast(ToolOAuthCallback, auth_callback).state
        )
        if not state:
            raise ToolAuthCallbackError("OAuth state not found")
        oauth = await self._load_oauth()
        await oauth.callback(cast(ToolOAuthCallback, auth_callback), state)
    

    async def _load_oauth(self) -> AgentToolOauth:
        oauth_config = self._oauth_config()
        client_info = await ToolOAuthClientInfoRepository(self.db).find_by_ids(self.agent.id, self.id)
        if not client_info or not client_info.scope:
            raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
        return AgentToolOauth(
            oauth_config.authority_base_url,
            oauth_config.metadata,
            cast(str, client_info.scope) + " offline_access",
            self.agent.id,
            self.id,
            self.user_id,
            self.db,
        )

    async def teardown(self):
        await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)
        await ToolOAuthClientInfoRepository(self.db).delete(self.agent.id, self.id)

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        pass


class TokenOpenApiTool(OpenApiTool, abc.ABC):
    _token_secret_key: ClassVar[str]
    _auth_check_path: ClassVar[str]
    _token: Optional[str] = None

    @abc.abstractmethod
    def _invalid_token_message(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_token_to_headers(self, headers: dict, token: str) -> dict:
        raise NotImplementedError

    async def _setup_tool(self, prev_config: Optional[AgentToolConfig]):
        self._token = self._get_secret(self._token_secret_key)
        prev_token = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        if (
            prev_config and prev_config.config != self.config
            or self._token and prev_token and self._token != prev_token.access_token
        ):
            await self.teardown()
        await self._check_auth()
        await self._save_token(self._token, self.db)

    async def teardown(self):
        await ToolAuthRepository(self.db).delete_token(self.user_id, self.agent.id, self.id)

    async def _save_token(self, token: Optional[str], db: AsyncSession):
        if not token:
            return
        await ToolAuthRepository(db).save_token(
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

    async def _check_auth(self):
        await self._invoke_rest_api(HTTPMethod.GET, f"{self._api_url.rstrip('/')}{self._auth_check_path}")

    @asynccontextmanager
    async def load(self) -> AsyncIterator["TokenOpenApiTool"]:
        try:
            await self._check_auth()
            yield self
        except HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
            raise

    async def auth(self, auth_callback: ToolAuthCallback):
        self._token = cast(ToolAuthTokenCallback, auth_callback).auth_token
        try:
            await self._check_auth()
        except HTTPStatusError as e:
            self._on_auth_callback_http_error(e)
        await self._save_token(self._token, self.db)

    def _on_auth_callback_http_error(self, e: HTTPStatusError) -> None:
        if e.response.status_code in (401, 403):
            raise ToolAuthCallbackError(self._invalid_token_message())
        raise

    async def clone(self, agent_id: int, cloned_agent_id: int, tool_id: str, user_id: int, db: AsyncSession) -> None:
        # Tokens are per-user and not copied; each user must connect their own token on the cloned agent.
        pass

    async def _add_auth_headers(self, headers: dict) -> dict:
        return self._apply_token_to_headers(headers, await self._resolve_token())

    async def _resolve_token(self) -> str:
        if self._token:
            return self._token.strip()
        token = await ToolAuthRepository(self.db).find_token(self.user_id, self.agent.id, self.id)
        if not token:
            raise ToolAuthRequestException(ToolAuthTokenRequest(tool_id=self.id, agent_id=self.agent.id))
        return cast(str, token.access_token).strip()
