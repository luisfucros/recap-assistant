"""Per-worker singletons for the ingestion service.

Mirrors the API's ``Resources``: build heavy/shared objects once per worker
process and reuse them across tasks. The ingestion pipeline only needs the DB,
Qdrant, object storage, and the embedder (no LLM, no web search — those are the
API's concern). ``get_ingestion_resources`` caches one instance per process.

Same eager/lazy split as the API: infra clients construct offline and connect
lazily (safe to build at worker start); the embedder is key-dependent / heavy
(local models load weights) so it's built on first use and then cached.
"""

from functools import cached_property, lru_cache
from typing import ClassVar

from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from shared.core.config import Settings, get_settings
from shared.db.engine import create_database_engine
from shared.observability.tracing import Tracer, build_tracer
from shared.providers import Embedder, build_embedder
from shared.providers.storage import S3StorageProvider, build_storage_provider


class IngestionResources:
    """Container for the ingestion worker's long-lived singletons."""

    # Lazy resources whose *construction* is expensive (e.g. a local embedding
    # model loading weights). Each worker child force-builds these at process
    # start so the first task doesn't pay the cost. See ``ingestion.bootstrap``.
    HEAVY_RESOURCES: ClassVar[tuple[str, ...]] = ("embedder",)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine = create_database_engine(settings)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self.qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            ),
            check_compatibility=False,
        )
        # Optional per-phase tracing (no-op when Langfuse isn't configured).
        self.tracer: Tracer = build_tracer(settings)

    @cached_property
    def storage(self) -> S3StorageProvider:
        return build_storage_provider(self.settings)

    @cached_property
    def embedder(self) -> Embedder:
        # Built once per worker (loads local model weights when using HuggingFace).
        return build_embedder(self.settings)

    async def aclose(self) -> None:
        """Dispose infra clients on worker shutdown."""
        await self.engine.dispose()
        await self.qdrant.close()


@lru_cache
def get_ingestion_resources() -> IngestionResources:
    """Return the process-wide ingestion resources (built once, cached)."""
    return IngestionResources(get_settings())
