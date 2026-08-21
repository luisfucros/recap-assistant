"""FastAPI application factory.

Assembles the core scaffolding — structured logging, CORS, security headers,
standard error handlers, and the health route — into a ready FastAPI app. Heavy
singletons (DB engine, Qdrant, Redis, embedder, LLM client, tracer, prompt
registry) are wired separately via the app lifespan; keeping that out of the
factory lets tests build a fully-functional app without external infrastructure.
"""

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from api.errors import register_exception_handlers
from api.lifespan import lifespan
from api.metrics import setup_metrics
from api.middleware import RequestIDMiddleware, SecurityHeadersMiddleware, configure_cors
from api.routers.admin import router as admin_router
from api.routers.analytics import router as analytics_router
from api.routers.auth import router as auth_router
from api.routers.chat import router as chat_router
from api.routers.documents import router as documents_router
from api.routers.evaluations import router as evaluations_router
from api.routers.health import router as health_router
from api.routers.memory import router as memory_router
from api.routers.progress import router as progress_router
from api.routers.recommendations import router as recommendations_router
from api.routers.usage import router as usage_router
from api.routers.users import router as users_router
from shared.core.config import Settings, get_settings
from shared.core.logging import configure_logging

# The OAuth login→callback handshake stores its CSRF state/nonce in this signed
# session cookie; it lives only for the brief redirect round-trip.
_OAUTH_SESSION_MAX_AGE_SECONDS = 600


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the configured FastAPI application.

    Args:
        settings: Settings to use; falls back to the cached process settings.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, quiet_loggers=("uvicorn.access",))

    app = FastAPI(
        title=settings.app_name,
        docs_url=f"{settings.api_v1_prefix}/docs",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    # Stash settings so the lifespan builds Resources from the same config.
    app.state.settings = settings

    configure_cors(app, settings)
    app.add_middleware(SecurityHeadersMiddleware)
    # Signed session cookie for the OAuth state/nonce (only used by Google
    # sign-in). Requires a secret; skipped when none is configured (OAuth is
    # unavailable in that case anyway).
    if settings.jwt_secret is not None:
        app.add_middleware(
            SessionMiddleware,
            secret_key=settings.jwt_secret.get_secret_value(),
            same_site="lax",
            https_only=settings.cookie_secure,
            max_age=_OAUTH_SESSION_MAX_AGE_SECONDS,
        )
    # Outermost middleware (added last): wraps CORS/security-headers/session
    # processing too, so it logs and times the whole request.
    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router, prefix=settings.api_v1_prefix)
    app.include_router(auth_router, prefix=settings.api_v1_prefix)
    app.include_router(users_router, prefix=settings.api_v1_prefix)
    app.include_router(documents_router, prefix=settings.api_v1_prefix)
    app.include_router(progress_router, prefix=settings.api_v1_prefix)
    app.include_router(analytics_router, prefix=settings.api_v1_prefix)
    app.include_router(chat_router, prefix=settings.api_v1_prefix)
    app.include_router(memory_router, prefix=settings.api_v1_prefix)
    app.include_router(recommendations_router, prefix=settings.api_v1_prefix)
    app.include_router(evaluations_router, prefix=settings.api_v1_prefix)
    app.include_router(usage_router, prefix=settings.api_v1_prefix)
    app.include_router(admin_router, prefix=settings.api_v1_prefix)
    # Always-on metrics: instrument HTTP + expose /metrics (root) for Prometheus.
    setup_metrics(app)
    return app
