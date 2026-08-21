"""Fixed-window request-rate limiting, backed by Redis (M8 hardening).

Two surfaces are worth throttling: auth (register/login/refresh — a credential-
stuffing/brute-force target) and chat (a turn-triggering route — unbounded LLM
cost per account). A plain in-process counter would work on one replica but
silently under-count once the API scales past one, since each replica would
keep its own window; Redis gives one counter shared across every replica, the
same role it already plays for the outbox relay and the agent scratchpad.

Uses a fixed window (``INCR`` then ``EXPIRE`` on the first hit) rather than a
sliding one: one round trip per hit, and "up to 2x burst at a window boundary"
is an acceptable looseness for abuse protection — this is not a billing-grade
limiter.

Fails open on a Redis error: a soft control sitting on top of a soft dependency
(mirrors ``ScratchpadService``/``AnalyticsService``) — a Redis outage should
degrade the app, not take auth or chat down with it.
"""

from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import RedisError

from shared.core.errors import RateLimitExceededError


class RateLimitService:
    """Fixed-window per-key request counter, shared across replicas via Redis."""

    def __init__(self, *, redis: Redis) -> None:
        """Wire the service to Redis (no policy state — limits are per call)."""
        self._redis = redis

    async def enforce(self, *, key: str, limit: int, window_seconds: int) -> None:
        """Count one hit for ``key``; raise once more than ``limit`` land in the window.

        Args:
            key: The full, already-namespaced Redis key. The caller picks the
                scope (e.g. per-client-IP for anonymous routes, per-user for
                authenticated ones) — this method only counts.
            limit: Max hits allowed within ``window_seconds``.
            window_seconds: The fixed window's length, (re)started on its first hit.

        Raises:
            RateLimitExceededError: ``key``'s count exceeded ``limit`` this window.
        """
        try:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, window_seconds)
        except RedisError:
            logger.warning("Rate limiter Redis error; failing open for key={}", key)
            return
        if count > limit:
            raise RateLimitExceededError()
