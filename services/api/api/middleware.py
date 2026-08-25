"""HTTP middleware: security response headers, CORS, and request tracing.

Security headers are added to every response as defense-in-depth (clickjacking,
MIME sniffing, referrer leakage, transport security). CORS is restricted to the
explicit origins from settings with credentials enabled, since auth travels in
httpOnly cookies — a wildcard origin is never allowed. Request-id tracing logs
every request generically so a turn's logs can be correlated end-to-end.
"""

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from shared.core.config import Settings

_REQUEST_ID_HEADER = "X-Request-ID"


def security_headers() -> dict[str, str]:
    """Return the security headers applied to every response."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the standard security headers to every outgoing response."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for name, value in security_headers().items():
            response.headers.setdefault(name, value)
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Bind a request-correlation id to every log line emitted for a request.

    Reuses an incoming ``X-Request-ID`` if the caller supplied one, else mints a
    fresh one; echoes it back so a client can correlate its own logs. Binding it
    via ``logger.contextualize`` (contextvars-based) means every log emitted
    anywhere during the request — including deep inside a chat turn's agent
    graph — carries it automatically, with no explicit threading. This is the
    only place that logs generic "a request happened" entries; per-turn/per-node
    detail is logged separately by :mod:`api.agent.graph`/:mod:`api.services.agent_service`.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
        started = time.monotonic()
        with logger.contextualize(request_id=request_id):
            logger.info("http.request.start", method=request.method, path=request.url.path)
            response = await call_next(request)
            duration_ms = int((time.monotonic() - started) * 1000)
            finish = logger.error if response.status_code >= 500 else logger.info
            finish(
                "http.request.finish",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


def configure_cors(app: FastAPI, settings: Settings) -> None:
    """Enable CORS for the explicit allowed origins (credentials on, no wildcard)."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
