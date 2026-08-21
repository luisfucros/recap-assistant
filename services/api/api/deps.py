"""FastAPI dependencies for reaching the app's singletons and per-request objects.

Handlers depend on these rather than touching ``app.state`` directly, so the
wiring stays in one place and is easy to override in tests.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from fastapi.security import APIKeyCookie
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import HTTPConnection, Request

from api.agent.context import ToolContext
from api.oauth import GoogleOAuthClient
from api.resources import Resources
from api.security import ACCESS_COOKIE
from api.services.agent_service import AgentService
from api.services.analytics_service import AnalyticsService
from api.services.auth_service import AuthService, InvalidTokenError
from api.services.conversation_service import ConversationService
from api.services.document_service import DocumentService
from api.services.evaluation_service import EvaluationService
from api.services.ingestion_service import IngestionService
from api.services.memory_service import MemoryService
from api.services.progress_service import ProgressService
from api.services.rate_limit_service import RateLimitService
from api.services.recommendation_service import RecommendationService
from api.services.retrieval_service import RetrievalService
from api.services.usage_service import UsageService
from shared.core.config import Settings
from shared.core.errors import AuthenticationError, AuthorizationError
from shared.models.user import User
from shared.prompt import PromptRegistry
from shared.providers import Embedder
from shared.repositories import (
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    EvaluationRunRepository,
    LongTermMemoryRepository,
    MessageRepository,
    OutboxRepository,
    ReadingEventRepository,
    ReadingProgressRepository,
    UsageEventRepository,
    UserRepository,
)


def get_resources(connection: HTTPConnection) -> Resources:
    """Return the process-wide resource container built in the lifespan.

    Typed as ``HTTPConnection`` (the ``Request``/``WebSocket`` base) rather than
    ``Request`` so this — and everything chained off :data:`ResourcesDep` — also
    resolves for the ``/chat/ws`` WebSocket route; FastAPI fills an
    ``HTTPConnection``-typed parameter for either connection kind.
    """
    return connection.app.state.resources


ResourcesDep = Annotated[Resources, Depends(get_resources)]


def get_app_settings(resources: ResourcesDep) -> Settings:
    """The settings the app was actually built with (not the global cache)."""
    return resources.settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


async def get_db_session(resources: ResourcesDep) -> AsyncIterator[AsyncSession]:
    """Yield a per-request async DB session (committed/rolled back by the caller)."""
    async with resources.sessionmaker() as session:
        yield session


def get_qdrant(resources: ResourcesDep) -> AsyncQdrantClient:
    return resources.qdrant


def get_redis(resources: ResourcesDep) -> Redis:
    return resources.redis


def get_prompt_registry(resources: ResourcesDep) -> PromptRegistry:
    return resources.prompts


def get_embedder(resources: ResourcesDep) -> Embedder:
    """The active embedder (built lazily; raises if its API key is unset)."""
    return resources.embedder


def get_auth_service(resources: ResourcesDep) -> AuthService:
    """The process-wide auth service (built lazily; needs ``jwt_secret``)."""
    return resources.auth_service


def get_google_oauth(resources: ResourcesDep) -> GoogleOAuthClient:
    """The Google OAuth client (built lazily; 404s if OAuth isn't configured)."""
    return resources.google_oauth


# Common annotated dependencies for handler signatures.
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
QdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant)]
RedisDep = Annotated[Redis, Depends(get_redis)]
PromptsDep = Annotated[PromptRegistry, Depends(get_prompt_registry)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
GoogleOAuthDep = Annotated[GoogleOAuthClient, Depends(get_google_oauth)]


def get_ingestion_service(resources: ResourcesDep) -> IngestionService:
    """The upload handoff service (built lazily; holds the storage provider)."""
    return resources.ingestion_service


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def get_document_service(resources: ResourcesDep) -> DocumentService:
    """The document lifecycle service (delete + metadata; built lazily)."""
    return resources.document_service


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


def get_user_repository(session: DbSession) -> UserRepository:
    """A ``UserRepository`` bound to the request's DB session."""
    return UserRepository(session)


UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]


def get_outbox_repository(session: DbSession) -> OutboxRepository:
    """An ``OutboxRepository`` bound to the request's DB session (not user-scoped)."""
    return OutboxRepository(session)


OutboxRepositoryDep = Annotated[OutboxRepository, Depends(get_outbox_repository)]


def get_evaluation_run_repository(session: DbSession) -> EvaluationRunRepository:
    """An ``EvaluationRunRepository`` bound to the request's session (not user-scoped)."""
    return EvaluationRunRepository(session)


EvaluationRunRepositoryDep = Annotated[
    EvaluationRunRepository, Depends(get_evaluation_run_repository)
]


async def resolve_user_from_token(
    token: str | None, auth: AuthService, users: UserRepository
) -> User:
    """Decode an access token and look up its owner, or raise :class:`AuthenticationError`.

    Shared by :func:`get_current_user` (HTTP) and the ``/chat/ws`` WebSocket
    route: a bad/missing token and one whose user no longer exists both raise
    the same error — never reveal which — and the ``user_id`` it returns comes
    only from the signed token, never from client input, upholding per-user
    isolation.
    """
    if not token:
        raise AuthenticationError("Not authenticated.")
    try:
        user_id = auth.decode_token(token, expected_type="access")
    except InvalidTokenError as exc:
        raise AuthenticationError("Invalid or expired token.") from exc
    user = await users.get_by_id(user_id)
    if user is None:
        raise AuthenticationError("Invalid or expired token.")
    return user


_access_cookie_scheme = APIKeyCookie(name=ACCESS_COOKIE, auto_error=False)


async def get_current_user(
    auth: AuthServiceDep,
    users: UserRepositoryDep,
    token: Annotated[str | None, Depends(_access_cookie_scheme)] = None,
) -> User:
    """Resolve the authenticated user from the access-token cookie.

    Set by ``POST /auth/login``; the browser attaches it automatically. Backed
    by an ``APIKeyCookie`` security scheme (``auto_error=False`` so a missing
    cookie falls through to our own 401 below rather than the scheme's own)
    so every route depending on this — transitively, ``CurrentUser``/
    ``AdminUser`` and everything built on them — is marked "protected" in the
    OpenAPI schema and shows a padlock in Swagger, instead of looking like an
    ordinary, unauthenticated route. The cookie is checked first so an
    unauthenticated request gets a clean 401 without touching the DB.

    Not used by ``/chat/ws``: raising there from a dependency (rather than from
    inside the handler) would make FastAPI try to render the 401 as an HTTP
    response over the WebSocket ASGI scope, which raises a ``RuntimeError``
    instead of closing cleanly — so that route resolves the same token by hand,
    via :func:`resolve_user_from_token`, and closes the connection itself.
    """
    return await resolve_user_from_token(token, auth, users)


CurrentUser = Annotated[User, Depends(get_current_user)]


async def require_admin(user: CurrentUser) -> User:
    """Resolve the authenticated user and require ``is_admin``, else 403.

    Layered on :func:`get_current_user` rather than replacing it, so a missing/
    invalid token still yields a 401 (unauthenticated) and only an authenticated
    non-admin gets a 403 (unauthorized) — the two failure modes stay distinct.
    """
    if not user.is_admin:
        raise AuthorizationError("This action requires an administrator.")
    return user


AdminUser = Annotated[User, Depends(require_admin)]


def get_document_repository(session: DbSession, user: CurrentUser) -> DocumentRepository:
    """A ``DocumentRepository`` scoped to the authenticated user.

    The owning ``user_id`` is taken from the resolved session user (a signed
    token), never from the request, so every document query is isolation-scoped.
    """
    return DocumentRepository(session, user.id)


DocumentRepositoryDep = Annotated[DocumentRepository, Depends(get_document_repository)]


def get_chunk_repository(session: DbSession, user: CurrentUser) -> ChunkRepository:
    """A ``ChunkRepository`` scoped to the authenticated user (retrieval hydration)."""
    return ChunkRepository(session, user.id)


ChunkRepositoryDep = Annotated[ChunkRepository, Depends(get_chunk_repository)]


def get_memory_repository(session: DbSession, user: CurrentUser) -> LongTermMemoryRepository:
    """A ``LongTermMemoryRepository`` scoped to the authenticated user."""
    return LongTermMemoryRepository(session, user.id)


MemoryRepositoryDep = Annotated[LongTermMemoryRepository, Depends(get_memory_repository)]


def get_progress_repository(session: DbSession, user: CurrentUser) -> ReadingProgressRepository:
    """A ``ReadingProgressRepository`` scoped to the authenticated user."""
    return ReadingProgressRepository(session, user.id)


ProgressRepositoryDep = Annotated[ReadingProgressRepository, Depends(get_progress_repository)]


def get_reading_event_repository(session: DbSession, user: CurrentUser) -> ReadingEventRepository:
    """A ``ReadingEventRepository`` scoped to the authenticated user (append-only)."""
    return ReadingEventRepository(session, user.id)


ReadingEventRepositoryDep = Annotated[ReadingEventRepository, Depends(get_reading_event_repository)]


def get_retrieval_service(resources: ResourcesDep) -> RetrievalService:
    """The process-wide retrieval service (built lazily; holds the embedder)."""
    return resources.retrieval_service


RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]


def get_memory_service(resources: ResourcesDep) -> MemoryService:
    """The process-wide memory service (built lazily; holds the embedder)."""
    return resources.memory_service


MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]


def get_recommendation_service(resources: ResourcesDep) -> RecommendationService:
    """The process-wide recommendation service (built lazily; holds the embedder)."""
    return resources.recommendation_service


RecommendationServiceDep = Annotated[RecommendationService, Depends(get_recommendation_service)]


def get_evaluation_service(resources: ResourcesDep) -> EvaluationService:
    """The process-wide evaluation service (built lazily; holds the agent service)."""
    return resources.evaluation_service


EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]


def get_progress_service(resources: ResourcesDep) -> ProgressService:
    """The process-wide reading-progress service (stateless; repos passed per call)."""
    return resources.progress_service


ProgressServiceDep = Annotated[ProgressService, Depends(get_progress_service)]


def get_analytics_service(resources: ResourcesDep) -> AnalyticsService:
    """The process-wide reading-analytics service (Redis-cached)."""
    return resources.analytics_service


AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]


def get_usage_repository(session: DbSession, user: CurrentUser) -> UsageEventRepository:
    """A ``UsageEventRepository`` scoped to the authenticated user (append-only)."""
    return UsageEventRepository(session, user.id)


UsageEventRepositoryDep = Annotated[UsageEventRepository, Depends(get_usage_repository)]


def get_usage_service(resources: ResourcesDep) -> UsageService:
    """The process-wide usage service (Redis-cached; per-user token spend/tool calls)."""
    return resources.usage_service


UsageServiceDep = Annotated[UsageService, Depends(get_usage_service)]


def get_rate_limit_service(resources: ResourcesDep) -> RateLimitService:
    """The process-wide rate limiter (Redis-backed; fails open on a Redis error)."""
    return resources.rate_limit_service


RateLimitServiceDep = Annotated[RateLimitService, Depends(get_rate_limit_service)]


async def enforce_auth_rate_limit(
    request: Request, rate_limit: RateLimitServiceDep, settings: SettingsDep
) -> None:
    """Throttle unauthenticated auth attempts per client IP (brute-force guard).

    There's no authenticated user yet on register/login/refresh, so the client's
    IP is the only identity available — the same reason these routes can't use
    a per-user key the way chat does. Raises ``RateLimitExceededError`` (429) via
    :meth:`RateLimitService.enforce`.
    """
    client_ip = request.client.host if request.client else "unknown"
    await rate_limit.enforce(
        key=f"ratelimit:auth:{client_ip}",
        limit=settings.rate_limit_auth_max,
        window_seconds=settings.rate_limit_auth_window_seconds,
    )


AuthRateLimit = Annotated[None, Depends(enforce_auth_rate_limit)]


async def enforce_chat_rate_limit(
    user: CurrentUser, rate_limit: RateLimitServiceDep, settings: SettingsDep
) -> None:
    """Throttle turn-triggering chat calls per authenticated user (LLM-cost guard).

    Keyed by user rather than IP: chat is already authenticated, and the thing
    worth bounding is per-account spend, not shared-network traffic. Raises
    ``RateLimitExceededError`` (429) via :meth:`RateLimitService.enforce`.
    """
    await rate_limit.enforce(
        key=f"ratelimit:chat:{user.id}",
        limit=settings.rate_limit_chat_max,
        window_seconds=settings.rate_limit_chat_window_seconds,
    )


ChatRateLimit = Annotated[None, Depends(enforce_chat_rate_limit)]


def get_conversation_repository(session: DbSession, user: CurrentUser) -> ConversationRepository:
    """A ``ConversationRepository`` scoped to the authenticated user."""
    return ConversationRepository(session, user.id)


ConversationRepositoryDep = Annotated[ConversationRepository, Depends(get_conversation_repository)]


def get_message_repository(session: DbSession, user: CurrentUser) -> MessageRepository:
    """A ``MessageRepository`` scoped to the authenticated user."""
    return MessageRepository(session, user.id)


MessageRepositoryDep = Annotated[MessageRepository, Depends(get_message_repository)]


def get_agent_service(resources: ResourcesDep) -> AgentService:
    """The process-wide agent runner (built lazily; holds models + checkpointer)."""
    return resources.agent_service


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]


def get_conversation_service(resources: ResourcesDep) -> ConversationService:
    """The process-wide chat-transcript service (stateless; repos passed per call)."""
    return resources.conversation_service


ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]


def get_tool_context(
    user: CurrentUser,
    documents: DocumentRepositoryDep,
    chunks: ChunkRepositoryDep,
    progress: ProgressRepositoryDep,
    progress_service: ProgressServiceDep,
    retrieval_service: RetrievalServiceDep,
    prompts: PromptsDep,
    resources: ResourcesDep,
    session: DbSession,
    events: ReadingEventRepositoryDep,
    memories: MemoryRepositoryDep,
    memory_service: MemoryServiceDep,
    usage: UsageEventRepositoryDep,
    usage_service: UsageServiceDep,
) -> ToolContext:
    """Assemble the per-turn, user-scoped context the agent's tools operate through.

    Every handle is derived server-side from the authenticated request — the owner
    ``user_id`` and the user's spoiler-safe default come from the resolved session
    user, never from the request body or the model — so a tool call can't widen its
    scope. The summarizer is a cheap-tier chat model built per request.
    """
    return ToolContext(
        user_id=user.id,
        documents=documents,
        chunks=chunks,
        progress_repo=progress,
        progress_service=progress_service,
        retrieval_service=retrieval_service,
        summarizer=resources.chat_model(tier="cheap"),
        prompts=prompts,
        user_spoiler_safe=user.spoiler_safe,
        session=session,
        events=events,
        memories=memories,
        memory_service=memory_service,
        recommendation_service=resources.recommendation_service,
        web_search=lambda: resources.web_search,
        usage=usage,
        usage_service=usage_service,
    )


ToolContextDep = Annotated[ToolContext, Depends(get_tool_context)]
