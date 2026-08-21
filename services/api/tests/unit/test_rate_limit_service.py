"""Unit tests for RateLimitService (fixed-window counting + fail-open)."""

import pytest
from api.services.rate_limit_service import RateLimitService
from redis.exceptions import RedisError

from shared.core.errors import RateLimitExceededError

pytestmark = pytest.mark.unit


class _FakeRedis:
    """In-memory stand-in for the Redis client; can be made to fail on demand."""

    def __init__(self, *, fail: bool = False) -> None:
        self.counts: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self._fail = fail

    async def incr(self, key: str) -> int:
        if self._fail:
            raise RedisError("boom")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        self.expire_calls.append((key, seconds))


async def test_enforce_allows_hits_up_to_the_limit() -> None:
    redis = _FakeRedis()
    service = RateLimitService(redis=redis)  # type: ignore[arg-type]

    for _ in range(3):
        await service.enforce(key="k", limit=3, window_seconds=60)

    assert redis.counts["k"] == 3


async def test_enforce_raises_once_the_limit_is_exceeded() -> None:
    redis = _FakeRedis()
    service = RateLimitService(redis=redis)  # type: ignore[arg-type]

    for _ in range(2):
        await service.enforce(key="k", limit=2, window_seconds=60)

    with pytest.raises(RateLimitExceededError):
        await service.enforce(key="k", limit=2, window_seconds=60)


async def test_enforce_sets_the_window_expiry_only_on_the_first_hit() -> None:
    redis = _FakeRedis()
    service = RateLimitService(redis=redis)  # type: ignore[arg-type]

    for _ in range(3):
        await service.enforce(key="k", limit=10, window_seconds=60)

    assert redis.expire_calls == [("k", 60)]


async def test_enforce_scopes_counts_independently_per_key() -> None:
    redis = _FakeRedis()
    service = RateLimitService(redis=redis)  # type: ignore[arg-type]

    await service.enforce(key="a", limit=1, window_seconds=60)
    await service.enforce(key="b", limit=1, window_seconds=60)  # distinct key, not exceeded

    with pytest.raises(RateLimitExceededError):
        await service.enforce(key="a", limit=1, window_seconds=60)


async def test_enforce_fails_open_on_a_redis_error() -> None:
    redis = _FakeRedis(fail=True)
    service = RateLimitService(redis=redis)  # type: ignore[arg-type]

    # A Redis outage must never block auth/chat — no exception, request proceeds.
    await service.enforce(key="k", limit=0, window_seconds=60)
