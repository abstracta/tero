from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import Response
from fastapi_mcp import AuthConfig, FastApiMCP
import httpx

from .agents.mcp_tools import router as agents_mcp_tools_router
from .core.auth import get_current_user
from .core.env import env
from .external_agents.mcp_tools import router as external_agents_mcp_tools_router
from .threads.mcp_tools import router as threads_mcp_tools_router

OPENID_SCOPES = ["openid", "profile", "email", "offline_access"]

mcp_server_router = APIRouter()


@mcp_server_router.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource_metadata():
    base_url = env.frontend_url.rstrip("/")
    return {
        "resource": f"{base_url}/mcp",
        "authorization_servers": [base_url],
        "scopes_supported": OPENID_SCOPES,
        "bearer_methods_supported": ["header"],
    }


@mcp_server_router.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server_forward():
    frontend_openid_url = env.frontend_openid_url or env.openid_url
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{frontend_openid_url}/.well-known/oauth-authorization-server", follow_redirects=True)
        metadata = response.json()
    metadata["scopes_supported"] = OPENID_SCOPES
    return metadata


# This is a workaround to allow the required-action page to load the css and js files
@mcp_server_router.get("/resources/{path:path}")
async def resources_forward(path: str):
    openid_url = env.openid_url
    keycloak_base = openid_url[:openid_url.index("/realms/")]
    target_url = f"{keycloak_base}/resources/{path}"
    async with httpx.AsyncClient() as client:
        response = await client.get(target_url)
        return Response(content=response.content, status_code=response.status_code)


def setup_mcp_server(app: FastAPI):
    if not env.openid_url:
        return

    mcp = FastApiMCP(
        app,
        auth_config=AuthConfig(
            dependencies=[Depends(get_current_user)],
        ),
        headers=["authorization", "host"],
    )
    mcp.mount_http()
    app.include_router(agents_mcp_tools_router)
    app.include_router(threads_mcp_tools_router)
    app.include_router(external_agents_mcp_tools_router)
    mcp.setup_server()
    # include the mcp server router after the mcp server is setup to avoid capturing well-known or other non-tool endpoints as tools
    app.include_router(mcp_server_router)
