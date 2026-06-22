from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api_keys.api import router as api_keys_router
from .agents.api import router as agents_router
from .agents.evaluators.api import router as evaluators_router
from .agents.prompts.api import router as agents_prompts_router
from .agents.test_cases.api import router as test_cases_router
from .ai_models.api import router as ai_models_router
from .core.api import BASE_PATH
from .core.domain import CamelCaseModel
from .core.env import env
from .external_agents.api import router as external_agents_router
from .mcp_server import setup_mcp_server
from .teams.api import router as teams_router
from .threads.api import router as threads_router
from .threads.agents_store import agents_store_lifespan
from .tools.api import router as tools_router
from .usage.api import router as usage_router
from .users.api import router as users_router


def _setup_logging():
    class HealthCheckFilter(logging.Filter):
        def filter(self, record):
            if hasattr(record, 'getMessage'):
                return not 'GET /health' in record.getMessage()
            return True

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(HealthCheckFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with agents_store_lifespan():
        yield


logger = logging.getLogger(__name__)
_setup_logging()
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["Content-Disposition", "Content-Type", "Location"])
app.add_middleware(GZipMiddleware)
if env.frontend_path:
    app.mount("/assets", StaticFiles(directory=os.path.join(env.frontend_path, "assets")), name="assets")
setup_mcp_server(app)


def _should_serve_frontend(path: str) -> bool:
    api_paths = ["/api", "/assets", "/mcp", "/.well-known/", "/resources/"]
    return not any(path.startswith(prefix) for prefix in api_paths) and path not in ("/manifest.json",)


@app.middleware("http")
async def frontend_router(request: Request, call_next) -> Response:
    if env.frontend_path and _should_serve_frontend(request.url.path):
        response = FileResponse(os.path.join(env.frontend_path, "index.html"))
        response.headers["Cache-Control"] = "no-cache"
        return response
    return await call_next(request)


class HubConfig(CamelCaseModel):
    authority: str
    client_id: str
    scope: str
    contact_email: str


# this endpoint is deprecated, use manifest.json instead (stay just for this release)
@app.get(f"{BASE_PATH}/config")
async def config() -> HubConfig:
    return HubConfig(authority=env.frontend_openid_url or env.openid_url, client_id=env.openid_client_id, scope=env.openid_scope,
                     contact_email=env.contact_email)


class ManifestAuthConfig(CamelCaseModel):
    url: str
    client_id: str
    scope: str


class Manifest(CamelCaseModel):
    id: str
    contact_email: str
    auth: ManifestAuthConfig
    disable_publish_global: bool


@app.get("/manifest.json")
async def manifest() -> Manifest:
    return Manifest(
        id=env.frontend_url,
        contact_email=env.contact_email,
        auth=ManifestAuthConfig(
            url=env.frontend_openid_url or env.openid_url,
            client_id=env.openid_client_id,
            scope=env.openid_scope,
        ),
        disable_publish_global=env.disable_publish_global or False
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


for router in [
    api_keys_router,
    ai_models_router,
    agents_router,
    tools_router,
    threads_router,
    usage_router,
    agents_prompts_router,
    users_router,
    teams_router,
    external_agents_router,
    test_cases_router,
    evaluators_router
]:
    app.include_router(router)
