from tero.core.env import set_current_tenant_name

from .common import *

async def test_manifest(client: AsyncClient):
    resp = await client.get("/manifest.json")
    resp.raise_for_status()
    assert resp.json() == {
        "id": env.frontend_url,
        "contactEmail": env.contact_email,
        "auth": {
            "url": env.frontend_openid_url or env.openid_url,
            "clientId": env.openid_client_id,
            "scope": env.openid_scope
        },
        "disablePublishGlobal": env.disable_publish_global or False,
        "maxFileSizeMb": env.max_file_size_mb,
    }


async def test_manifest_frontend_url_follows_tenant_override(client: AsyncClient):
    original_frontend_url = env.frontend_url
    original_tenants = env.tenants
    original_overrides = env.tenant_overrides
    env.frontend_url = "https://primary.example.com"
    env.tenants = ["primary", "acme"]
    env.tenant_overrides = {"acme": {"frontend_url": "https://acme.example.com"}}
    try:
        resp = await client.get("/manifest.json", headers={"host": "acme.example.com"})
        resp.raise_for_status()
        assert resp.json()["id"] == "https://acme.example.com"

        resp = await client.get("/manifest.json", headers={"host": "primary.example.com"})
        resp.raise_for_status()
        assert resp.json()["id"] == "https://primary.example.com"
    finally:
        set_current_tenant_name("")
        env.frontend_url = original_frontend_url
        env.tenants = original_tenants
        env.tenant_overrides = original_overrides
