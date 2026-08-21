"""Application lifespan: build singletons at startup, dispose at shutdown.

FastAPI runs this once around the app's life. Resources are stashed on
``app.state`` so request handlers reach them via the dependencies in ``api.deps``.

Startup optionally *warms up* (``warm_up_on_start``): it force-builds the heavy
lazy singletons and opens the infra connection pools, so the cost of loading a
local embedding model or completing a DB/Redis/Qdrant handshake is paid once here
rather than by whichever request happens to arrive first. Every warm-up step is
best-effort — failures are logged, never raised — because infra may not be
reachable yet and a genuine config error should surface on the first real request
exactly as it would without the warm-up.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from api.resources import Resources
from shared.core.config import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Construct :class:`Resources` on startup and dispose them on shutdown."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    resources = Resources(settings)
    app.state.resources = resources
    # Open the checkpointer pool regardless of the warm-up flag — it's required
    # infra for the agent, not an optimization. Non-blocking and defensive: if
    # Postgres isn't up yet, connections fill in the background and the first
    # chat request pays the wait, exactly like the other lazy connections.
    await _open_checkpointer(resources)
    if settings.warm_up_on_start:
        await _warm_up(resources)
    try:
        yield
    finally:
        await resources.aclose()


async def _open_checkpointer(resources: Resources) -> None:
    """Open the agent's durable-checkpointer pool (best-effort, non-blocking)."""
    try:
        await resources.checkpointer_pool.open(wait=False)
        logger.info("Opened the agent checkpointer pool at startup")
    except Exception:
        logger.exception("Opening the checkpointer pool failed; will connect lazily")


async def _warm_up(resources: Resources) -> None:
    """Force-build heavy singletons and open infra connection pools at startup."""
    await _warm_up_heavy_constructions(resources)
    await _warm_up_connections(resources)


def _heavy_resource_names(resources: Resources) -> tuple[str, ...]:
    """The heavy singletons to warm for *this* configuration.

    The base ``HEAVY_RESOURCES``, plus the local HuggingFace transcriber only when
    it's the selected provider: constructing it loads the Whisper model (heavy,
    one-time), so warming it here means the first voice-note turn skips the load —
    the same rationale as the local embedder. Hosted transcription and both vision
    describers build cheaply (just an httpx client), so they stay lazy, and a
    text-only or fully-hosted deployment loads no media model at boot (nor needs
    its key). Guarded with ``getattr`` so lightweight test doubles without
    ``settings`` still work.
    """
    names = type(resources).HEAVY_RESOURCES
    settings = getattr(resources, "settings", None)
    if getattr(settings, "transcription_provider", None) == "huggingface":
        names = (*names, "transcriber")
    return names


async def _warm_up_heavy_constructions(resources: Resources) -> None:
    """Force-build the declared heavy lazy resources once, best-effort.

    Two build contexts, because they conflict:

    * **Off-loop (default):** a local model loading weights (the embedder, or the
      local HuggingFace transcriber) is sync/blocking, so it builds in a worker
      thread to keep the event loop free.
    * **On-loop (``LOOP_BOUND_RESOURCES``):** the agent service constructs an
      ``AsyncPostgresSaver``, which captures the running event loop at
      construction and raises in a worker thread — so it must build on the loop
      (its construction is cheap and non-blocking, so that's fine).

    All stay ``cached_property`` (a missing key fails a request, not boot); we
    merely pay a successful build once here. Failures are logged, never raised.
    """
    loop_bound = getattr(type(resources), "LOOP_BOUND_RESOURCES", frozenset())
    for name in _heavy_resource_names(resources):
        try:
            if name in loop_bound:
                getattr(resources, name)  # on the event loop (loop-capturing ctor)
            else:
                await asyncio.to_thread(getattr, resources, name)  # blocking → off-loop
            logger.info("Warmed up resources.{} at startup", name)
        except Exception:
            logger.exception("Startup warm-up of resources.{} failed; will retry lazily", name)


async def _warm_up_connections(resources: Resources) -> None:
    """Open the DB/Redis/Qdrant pools with a cheap no-op so the first request
    doesn't pay the connection handshake. Probed concurrently, each defensive."""
    await asyncio.gather(
        _probe("postgres", _ping_db(resources.engine)),
        _probe("redis", resources.redis.ping()),
        _probe("qdrant", resources.qdrant.get_collections()),
    )


async def _ping_db(engine: AsyncEngine) -> None:
    """Open a pooled connection and run ``SELECT 1`` to complete the handshake."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def _probe(label: str, awaitable: Awaitable[object]) -> None:
    """Await a connection warm-up, logging success/failure without raising."""
    try:
        await awaitable
        logger.info("Warmed up {} connection at startup", label)
    except Exception:
        logger.exception("Startup connection warm-up for {} failed; will connect lazily", label)
