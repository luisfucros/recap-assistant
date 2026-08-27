"""Process-wide singletons for the API service.

Heavy/shared objects are built once and reused for the app's lifetime (the
"load heavy classes once at startup" rule), then disposed on shutdown. Two tiers:

- **Eager** (built in ``__init__``): infra clients that construct without network
  or API keys — DB engine, Redis, Qdrant — plus the prompt registry and tracer.
  Constructing them never fails, so startup is safe even with an empty ``.env``
  (they connect lazily on first use).
- **Lazy** (``cached_property``): providers that need an API key (embedder, web
  search). Building them can raise if the key is missing, so we defer until a
  request actually needs them — a missing key fails that request, not startup.

The LLM chat model is intentionally not held here: it's built per turn in
``api.llm`` (LangGraph binds tools to a fresh model), with its own resilience.
"""

from contextlib import suppress
from functools import cached_property
from typing import ClassVar
from uuid import UUID

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from api.checkpointer import build_pool
from api.evaluation.scorers.judge import EvaluationJudgment
from api.llm import build_chat_model, build_resilient_structured_model
from api.oauth import GoogleOAuthClient
from api.services.agent_service import AgentService, build_agent_models
from api.services.analytics_service import AnalyticsService
from api.services.auth_service import AuthService
from api.services.compaction_service import CompactionService
from api.services.conversation_service import ConversationService
from api.services.document_service import DocumentService
from api.services.evaluation_service import EvaluationService
from api.services.ingestion_service import IngestionService
from api.services.memory_service import MemoryService
from api.services.multimodal_service import MultimodalNormalizer
from api.services.progress_service import ProgressService
from api.services.rate_limit_service import RateLimitService
from api.services.recommendation_service import RecommendationService
from api.services.retrieval_service import RetrievalService
from api.services.scratchpad_service import ScratchpadService
from api.services.usage_service import UsageService
from shared.core.config import Settings
from shared.db.engine import create_database_engine
from shared.observability.tracing import Tracer, build_tracer
from shared.prompt import PromptRegistry, get_prompt_registry
from shared.providers import (
    Embedder,
    ImageDescriber,
    Transcriber,
    WebSearchProvider,
    build_embedder,
    build_image_describer,
    build_transcriber,
    build_web_search_provider,
)
from shared.providers.storage import S3StorageProvider, build_storage_provider
from shared.vectorstore import ChunkVectorStore, MemoryVectorStore


class Resources:
    """Container for the API's long-lived singletons."""

    # Lazy resources the startup warm-up force-builds once (best-effort), so the
    # first request doesn't pay for them and a misconfiguration surfaces at boot:
    #   - ``embedder`` — its *construction* is expensive (a local model loading
    #     weights into memory/GPU).
    #   - ``agent_service`` — builds the agent's LLM entry points; warming it means
    #     the chat model wiring is ready before the first ``/chat`` and a missing/
    #     invalid LLM key shows up in the startup log rather than on first use.
    #   - ``conversation_service`` — also builds an ``AsyncPostgresSaver`` (to
    #     delete a conversation's checkpoint thread on ``DELETE /conversations/{id}``),
    #     which MUST happen on the event loop (see ``LOOP_BOUND_RESOURCES`` below) —
    #     left lazy, its first real access would come from a plain sync dependency
    #     function run in a worker thread, which has no event loop to capture.
    # All three stay ``cached_property`` (a missing key fails a request, never boot
    # — the warm-up only logs), so non-LLM endpoints still serve with an empty key.
    HEAVY_RESOURCES: ClassVar[tuple[str, ...]] = (
        "embedder",
        "agent_service",
        "conversation_service",
    )

    # Subset of HEAVY_RESOURCES that MUST be constructed on the event loop rather
    # than a worker thread: ``agent_service``/``conversation_service`` each build an
    # ``AsyncPostgresSaver``, which captures the running loop at construction
    # (``asyncio.get_running_loop``) and raises in a thread. The rest default to a
    # worker thread because their construction blocks (a local model loading weights).
    LOOP_BOUND_RESOURCES: ClassVar[frozenset[str]] = frozenset(
        {"agent_service", "conversation_service"}
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # --- eager: construction is offline and key-free (connect lazily) ---
        self.engine: AsyncEngine = create_database_engine(settings)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self.redis: Redis = Redis.from_url(settings.redis_url)
        # check_compatibility=False: don't make a version call at construction, so
        # the app starts even when Qdrant isn't reachable yet.
        self.qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            check_compatibility=False,
        )
        self.prompts: PromptRegistry = get_prompt_registry()
        self.tracer: Tracer = build_tracer(settings)
        # Unopened psycopg pool for the durable agent checkpointer (opened in the
        # lifespan, closed on shutdown). Construction is offline, so it's eager.
        self.checkpointer_pool: AsyncConnectionPool = build_pool(settings)

    # --- lazy: key-dependent, built on first use and cached ---------------- #
    @cached_property
    def storage(self) -> S3StorageProvider:
        return build_storage_provider(self.settings)

    @cached_property
    def embedder(self) -> Embedder:
        return build_embedder(self.settings)

    @cached_property
    def auth_service(self) -> AuthService:
        """Auth (JWT + password hashing). Lazy: needs ``jwt_secret`` and builds
        the argon2 hasher once, so a missing secret fails a protected request
        rather than startup."""
        return AuthService(self.settings)

    @cached_property
    def ingestion_service(self) -> IngestionService:
        """Upload handoff (validate-store-enqueue). Lazy: it holds the storage
        provider, which is itself built on first use."""
        return IngestionService(storage=self.storage, embed_model=self.settings.embedding_model)

    @cached_property
    def document_service(self) -> DocumentService:
        """Document lifecycle (delete + metadata). Lazy: holds the storage provider
        (built on first use) and a vector store for chunk cleanup (no dim needed
        for deletion)."""
        return DocumentService(
            storage=self.storage,
            vector_store=ChunkVectorStore(
                self.qdrant, collection=self.settings.qdrant_chunks_collection
            ),
        )

    @cached_property
    def progress_service(self) -> ProgressService:
        """Reading-progress service (stateless; per-request repos passed per call)."""
        return ProgressService()

    @cached_property
    def agent_service(self) -> AgentService:
        """The LangGraph agent runner. Warm-built at startup (``HEAVY_RESOURCES``)
        so the chat model wiring is ready before the first ``/chat``. Still a
        ``cached_property``: building the models needs LLM API keys, and the
        warm-up is best-effort — a missing key is logged at boot and fails the
        chat request, never startup. Holds the pool-backed durable checkpointer
        over the (already-open) pool."""
        return AgentService(
            build_agent_models(self.settings),
            checkpointer=AsyncPostgresSaver(self.checkpointer_pool),
            tracer=self.tracer,
            scratchpad=self.scratchpad_service,
            # A factory, not the built normalizer: the transcriber/vision providers
            # need API keys, so it's built only on a turn that actually has media —
            # a text-only chat never touches (and never fails on) those keys.
            multimodal=lambda: self.multimodal_normalizer,
            compaction=self.compaction_service,
        )

    @cached_property
    def transcriber(self) -> Transcriber:
        """Speech-to-text provider (FR-19). Lazy: a hosted provider needs its key.

        When ``TRANSCRIPTION_PROVIDER=huggingface`` this loads a local Whisper model
        at construction, so the startup warm-up force-builds it (see
        ``api.lifespan._heavy_resource_names``) to keep that cost off the first
        voice-note turn; hosted transcription stays lazy (cheap to build)."""
        return build_transcriber(self.settings)

    @cached_property
    def image_describer(self) -> ImageDescriber:
        """Image-to-text provider (FR-19). Lazy: a hosted provider needs its key."""
        return build_image_describer(self.settings)

    @cached_property
    def multimodal_normalizer(self) -> MultimodalNormalizer:
        """Folds a chat turn's audio/image into text, archiving originals (FR-19).

        Lazy and only reached on a media turn (via ``agent_service``'s factory), so
        the transcription/vision keys are required only when media is actually sent.
        """
        return MultimodalNormalizer(
            transcriber=self.transcriber,
            image_describer=self.image_describer,
            storage=self.storage,
        )

    @cached_property
    def scratchpad_service(self) -> ScratchpadService:
        """The agent's turn-scoped working memory (Redis-backed, TTL'd)."""
        return ScratchpadService(redis=self.redis, ttl_seconds=self.settings.scratchpad_ttl_seconds)

    @cached_property
    def compaction_service(self) -> CompactionService:
        """Token-budget auto-compaction of the agent's checkpointed history (FR-4.1).

        Stateless (holds only settings), so it's cheap to build eagerly."""
        return CompactionService(settings=self.settings)

    @cached_property
    def rate_limit_service(self) -> RateLimitService:
        """Fixed-window request-rate limiter (Redis-backed, shared across replicas)."""
        return RateLimitService(redis=self.redis)

    @cached_property
    def usage_service(self) -> UsageService:
        """Per-user token-spend/tool-call usage service (Redis-cached, NFR-13)."""
        return UsageService(redis=self.redis, ttl_seconds=self.settings.usage_cache_ttl_seconds)

    @cached_property
    def conversation_service(self) -> ConversationService:
        """Chat-transcript service (per-request repos passed per call).

        Holds the same pool-backed checkpointer as ``agent_service`` so deleting
        a conversation also drops its LangGraph thread state.
        """
        return ConversationService(checkpointer=AsyncPostgresSaver(self.checkpointer_pool))

    @cached_property
    def analytics_service(self) -> AnalyticsService:
        """Reading-analytics service (Redis-cached; per-request repos passed per call)."""
        return AnalyticsService(
            redis=self.redis, ttl_seconds=self.settings.analytics_cache_ttl_seconds
        )

    @cached_property
    def retrieval_service(self) -> RetrievalService:
        """Semantic retrieval (embed→search→hydrate). Lazy: holds the embedder
        (built on first use) and a chunk vector store (no dim needed for search)."""
        return RetrievalService(
            embedder=self.embedder,
            vector_store=ChunkVectorStore(
                self.qdrant, collection=self.settings.qdrant_chunks_collection
            ),
            settings=self.settings,
        )

    @cached_property
    def recommendation_service(self) -> RecommendationService:
        """Explainable recommendations (FR-5). Lazy: holds the embedder (built on
        first use) and a chunk vector store (no dim needed for search) — reads
        the same ``document_chunks`` collection retrieval does, never writes."""
        return RecommendationService(
            embedder=self.embedder,
            vector_store=ChunkVectorStore(
                self.qdrant, collection=self.settings.qdrant_chunks_collection
            ),
        )

    @cached_property
    def memory_service(self) -> MemoryService:
        """Long-term memory (write/retrieve/view/delete). Lazy: holds the embedder
        (built on first use) and a memory vector store sized for upserts (``dim``
        needed to bootstrap the collection on first write)."""
        return MemoryService(
            embedder=self.embedder,
            vector_store=MemoryVectorStore(
                self.qdrant,
                collection=self.settings.qdrant_memory_collection,
                dim=self.embedder.dim,
            ),
        )

    @cached_property
    def evaluation_service(self) -> EvaluationService:
        """Dataset evaluation runner (FR-12). Lazy: holds the agent service (whose
        own build is itself lazy/key-dependent) plus a cheap-tier judge model and
        summarizer, so an eval-less deployment never builds either."""
        return EvaluationService(
            agent_service=self.agent_service,
            retrieval_service=self.retrieval_service,
            progress_service=self.progress_service,
            memory_service=self.memory_service,
            recommendation_service=self.recommendation_service,
            usage_service=self.usage_service,
            web_search=lambda: self.web_search,
            summarizer=self.chat_model(tier="cheap"),
            embedder=self.embedder,
            vector_store=ChunkVectorStore(
                self.qdrant,
                collection=self.settings.qdrant_chunks_collection,
                dim=self.embedder.dim,
            ),
            judge_model=build_resilient_structured_model(
                self.settings, EvaluationJudgment, tier="cheap"
            ),
            prompts=self.prompts,
            tracer=self.tracer,
            settings=self.settings,
            enqueue=_enqueue_evaluation_run,
        )

    @cached_property
    def google_oauth(self) -> GoogleOAuthClient:
        """Google OAuth client. Lazy: raises if OAuth isn't configured, so the
        sign-in routes 404 cleanly while the rest of the app runs unaffected."""
        return GoogleOAuthClient(self.settings)

    @cached_property
    def web_search(self) -> WebSearchProvider:
        return build_web_search_provider(self.settings)

    def chat_model(self, *, tier: str = "default"):  # noqa: ANN201 — LangChain BaseChatModel
        """Build a fresh LLM chat model for the given tier (not cached; tools bind per turn)."""
        return build_chat_model(self.settings, tier=tier)

    async def aclose(self) -> None:
        """Dispose all resources on shutdown (best-effort; never raises)."""
        await self.engine.dispose()
        await self.redis.aclose()
        await self.qdrant.close()
        with suppress(Exception):
            await self.checkpointer_pool.close()
        self.tracer.flush()


def _enqueue_evaluation_run(run_id: UUID) -> None:
    """Dispatch scoring onto the eval Celery queue (imported lazily to avoid cycles)."""
    from api.eval_worker.tasks import run_evaluation_task

    run_evaluation_task.delay(str(run_id))
