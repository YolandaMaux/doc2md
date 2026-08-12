"""
security.py
-----------
Shared API-key security dependency for nginx-gated endpoints.

nginx validates the key upstream; FastAPI only declares the OpenAPI
security scheme so that Swagger UI exposes an **Authorize** button and
automatically injects ``X-API-Key: <key>`` into every test request.

Usage
-----
Import ``get_api_key`` and add it as a dependency:

    # app-level (covers all routes)
    app = FastAPI(dependencies=[Depends(get_api_key)])

    # router-level (covers all routes in that router)
    router = APIRouter(dependencies=[Depends(get_api_key)])
"""
from __future__ import annotations

import os

from fastapi import Security
from fastapi.security import APIKeyHeader

# Header name nginx expects.  Override via NGINX_API_KEY_HEADER env var.
_HEADER_NAME: str = os.getenv("DOC2MD_NGINX_API_KEY_HEADER", "X-API-Key")

nginx_api_key_scheme = APIKeyHeader(
    name=_HEADER_NAME,
    description=(
        "API key enforced by the **nginx** reverse proxy.  "
        "Click **Authorize**, paste your key, then confirm — "
        f"Swagger will attach `{_HEADER_NAME}: <key>` to every request automatically."
    ),
    auto_error=False,   # nginx is the gatekeeper; FastAPI just forwards the header
)


async def get_api_key(
    api_key: str | None = Security(nginx_api_key_scheme),
) -> str | None:
    """
    Declare the nginx API-key header in the OpenAPI / Swagger schema.

    Actual key validation happens upstream in nginx.  FastAPI receives
    (and transparently forwards) whatever value the Swagger client sends.
    """
    return api_key
