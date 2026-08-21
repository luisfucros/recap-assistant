"""Unit tests for UsageService (token/tool-call aggregation + caching + writes)."""

import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from api.services.usage_service import UsageService

from shared.core.enums import UsageEventType
from shared.models.usage import UsageEvent

pytestmark = pytest.mark.unit

_TODAY = date(2026, 8, 11)


def _event(
    *,
    type: UsageEventType,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    tool_name: str | None = None,
    occurred_at: datetime | None = None,
) -> UsageEvent:
    return UsageEvent(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        type=type,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_name=tool_name,
        occurred_at=occurred_at or datetime.combine(_TODAY, datetime.min.time(), tzinfo=UTC),
    )


# --- _compute (pure) ----------------------------------------------------- #


def test_compute_sums_token_usage_events() -> None:
    events = [
        _event(type=UsageEventType.TOKEN_USAGE, prompt_tokens=100, completion_tokens=20),
        _event(type=UsageEventType.TOKEN_USAGE, prompt_tokens=50, completion_tokens=10),
    ]
    summary = UsageService._compute(events, window_days=30)

    assert summary.prompt_tokens == 150
    assert summary.completion_tokens == 30
    assert summary.total_tokens == 180
    assert summary.tool_calls == 0
    assert summary.tool_calls_by_tool == []


def test_compute_counts_tool_calls_by_name() -> None:
    events = [
        _event(type=UsageEventType.TOOL_CALL, tool_name="retrieve_chunks"),
        _event(type=UsageEventType.TOOL_CALL, tool_name="retrieve_chunks"),
        _event(type=UsageEventType.TOOL_CALL, tool_name="web_search"),
    ]
    summary = UsageService._compute(events, window_days=30)

    assert summary.tool_calls == 3
    assert [c.model_dump() for c in summary.tool_calls_by_tool] == [
        {"tool": "retrieve_chunks", "count": 2},
        {"tool": "web_search", "count": 1},
    ]


def test_compute_empty_history_is_all_zero() -> None:
    summary = UsageService._compute([], window_days=30)
    assert summary.prompt_tokens == 0
    assert summary.completion_tokens == 0
    assert summary.total_tokens == 0
    assert summary.tool_calls == 0
    assert summary.tool_calls_by_tool == []


# --- writes --------------------------------------------------------------- #


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _FakeUsageRepo:
    def __init__(self, user_id: uuid.UUID) -> None:
        self.user_id = user_id
        self.added: list[UsageEvent] = []

    async def add(self, entity: UsageEvent) -> UsageEvent:
        self.added.append(entity)
        return entity

    async def list_since(self, since: datetime, *, limit: int = 10_000) -> list[UsageEvent]:
        return [e for e in self.added if e.occurred_at >= since]


class _FakeRedis:
    def __init__(self, value: bytes | None = None) -> None:
        self._preset = value
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, Any]] = []

    async def get(self, key: str) -> Any:
        self.get_calls.append(key)
        return self._preset

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self.set_calls.append((key, ex))


async def test_record_token_usage_writes_and_commits() -> None:
    session = _FakeSession()
    repo = _FakeUsageRepo(uuid.uuid4())
    service = UsageService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]

    await service.record_token_usage(
        session=session,
        usage=repo,
        prompt_tokens=42,
        completion_tokens=7,  # type: ignore[arg-type]
    )

    assert len(repo.added) == 1
    assert repo.added[0].type is UsageEventType.TOKEN_USAGE
    assert repo.added[0].prompt_tokens == 42
    assert repo.added[0].completion_tokens == 7
    assert repo.added[0].user_id == repo.user_id
    assert session.commits == 1


async def test_record_token_usage_is_a_noop_when_both_counts_are_zero() -> None:
    session = _FakeSession()
    repo = _FakeUsageRepo(uuid.uuid4())
    service = UsageService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]

    await service.record_token_usage(
        session=session,
        usage=repo,
        prompt_tokens=0,
        completion_tokens=0,  # type: ignore[arg-type]
    )

    assert repo.added == []
    assert session.commits == 0


async def test_record_tool_call_writes_and_commits() -> None:
    session = _FakeSession()
    repo = _FakeUsageRepo(uuid.uuid4())
    service = UsageService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]

    await service.record_tool_call(session=session, usage=repo, tool_name="summarize")  # type: ignore[arg-type]

    assert len(repo.added) == 1
    assert repo.added[0].type is UsageEventType.TOOL_CALL
    assert repo.added[0].tool_name == "summarize"
    assert session.commits == 1


# --- get_usage (caching) --------------------------------------------------- #


async def test_get_usage_computes_and_caches_on_miss() -> None:
    redis = _FakeRedis(value=None)
    service = UsageService(redis=redis, ttl_seconds=300)  # type: ignore[arg-type]
    user_id = uuid.uuid4()
    repo = _FakeUsageRepo(user_id)
    repo.added.append(
        _event(
            type=UsageEventType.TOKEN_USAGE,
            prompt_tokens=10,
            completion_tokens=5,
            occurred_at=datetime.combine(_TODAY, datetime.min.time(), tzinfo=UTC),
        )
    )

    summary = await service.get_usage(
        user_id=user_id,
        usage=repo,
        today=_TODAY,
        window_days=30,  # type: ignore[arg-type]
    )

    assert summary.total_tokens == 15
    assert redis.set_calls == [(f"usage:{user_id}:30", 300)]


async def test_get_usage_serves_from_cache_on_hit() -> None:
    cached = UsageService._compute(
        [_event(type=UsageEventType.TOKEN_USAGE, prompt_tokens=99, completion_tokens=1)],
        window_days=30,
    )
    redis = _FakeRedis(value=cached.model_dump_json().encode())
    service = UsageService(redis=redis, ttl_seconds=300)  # type: ignore[arg-type]
    # The repo would only be touched on a cache miss.
    repo = _FakeUsageRepo(uuid.uuid4())

    summary = await service.get_usage(
        user_id=uuid.uuid4(),
        usage=repo,
        today=_TODAY,
        window_days=30,  # type: ignore[arg-type]
    )

    assert summary.total_tokens == 100
    assert redis.set_calls == []


async def test_get_usage_window_excludes_events_before_the_window_start() -> None:
    redis = _FakeRedis(value=None)
    service = UsageService(redis=redis, ttl_seconds=300)  # type: ignore[arg-type]
    user_id = uuid.uuid4()
    repo = _FakeUsageRepo(user_id)
    old = _event(
        type=UsageEventType.TOKEN_USAGE,
        prompt_tokens=1000,
        completion_tokens=0,
        occurred_at=datetime.combine(_TODAY, datetime.min.time(), tzinfo=UTC) - timedelta(days=60),
    )
    recent = _event(
        type=UsageEventType.TOKEN_USAGE,
        prompt_tokens=10,
        completion_tokens=0,
        occurred_at=datetime.combine(_TODAY, datetime.min.time(), tzinfo=UTC),
    )
    repo.added.extend([old, recent])

    summary = await service.get_usage(
        user_id=user_id,
        usage=repo,
        today=_TODAY,
        window_days=30,  # type: ignore[arg-type]
    )

    assert summary.prompt_tokens == 10
