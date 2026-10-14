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
