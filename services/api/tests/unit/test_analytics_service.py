"""Unit tests for AnalyticsService (pace/streak/pages-over-time math + caching)."""

import uuid
from datetime import UTC, date, datetime, time
from typing import Any

import pytest
from api.services.analytics_service import AnalyticsService

from shared.core.enums import ReadingEventType, ReadingStatus
from shared.models.reading import ReadingEvent

pytestmark = pytest.mark.unit

_TODAY = date(2026, 8, 3)


def _ev(
    day: date,
    *,
    from_page: int | None = None,
    to_page: int | None = None,
    type: ReadingEventType = ReadingEventType.POSITION_ADVANCED,
) -> ReadingEvent:
    return ReadingEvent(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        type=type,
        from_page=from_page,
        to_page=to_page,
        occurred_at=datetime.combine(day, time(12, 0), tzinfo=UTC),
    )


def _counts(reading: int = 0, completed: int = 0, cancelled: int = 0) -> dict[ReadingStatus, int]:
    return {
        ReadingStatus.READING: reading,
        ReadingStatus.COMPLETED: completed,
        ReadingStatus.CANCELLED: cancelled,
    }


# --- _compute: pages, pace, status counts -------------------------------- #


def test_compute_sums_forward_pages_and_computes_pace() -> None:
    events = [
        _ev(date(2026, 8, 3), from_page=0, to_page=10),
        _ev(date(2026, 8, 2), from_page=10, to_page=30),
        _ev(date(2026, 8, 1), from_page=30, to_page=40),
    ]
    summary = AnalyticsService._compute(
        events, _counts(reading=2, completed=1), today=_TODAY, window_days=30
    )

    assert summary.pages_read == 40  # 10 + 20 + 10
    assert summary.active_days == 3
    assert summary.pace_pages_per_day == round(40 / 3, 2)
    assert summary.documents_started == 2
    assert summary.documents_completed == 1
    assert summary.documents_cancelled == 0


def test_compute_ignores_backward_and_non_position_events() -> None:
    events = [
        _ev(date(2026, 8, 3), from_page=0, to_page=10),
        _ev(date(2026, 8, 3), from_page=50, to_page=40),  # backward → ignored
        _ev(date(2026, 8, 3), type=ReadingEventType.STATUS_CHANGED, to_page=10),  # not a move
        _ev(date(2026, 8, 3), type=ReadingEventType.COMPLETED, to_page=10),
    ]
    summary = AnalyticsService._compute(events, _counts(), today=_TODAY, window_days=30)

    assert summary.pages_read == 10
    assert summary.active_days == 1


def test_compute_empty_history_is_all_zero() -> None:
    summary = AnalyticsService._compute([], _counts(), today=_TODAY, window_days=30)
    assert summary.pages_read == 0
    assert summary.active_days == 0
    assert summary.pace_pages_per_day == 0.0
    assert summary.current_streak_days == 0
    assert summary.longest_streak_days == 0
    assert summary.pages_over_time == []


# --- streaks ------------------------------------------------------------- #


def test_current_streak_counts_back_from_today() -> None:
    events = [
        _ev(date(2026, 8, 3), from_page=0, to_page=10),
        _ev(date(2026, 8, 2), from_page=10, to_page=20),
        _ev(date(2026, 8, 1), from_page=20, to_page=30),
        # gap on 7/31
        _ev(date(2026, 7, 29), from_page=30, to_page=40),
        _ev(date(2026, 7, 28), from_page=40, to_page=50),
    ]
    summary = AnalyticsService._compute(events, _counts(), today=_TODAY, window_days=30)

    assert summary.current_streak_days == 3  # 8/1, 8/2, 8/3
    assert summary.longest_streak_days == 3


def test_current_streak_zero_when_no_reading_today() -> None:
    events = [
        _ev(date(2026, 8, 2), from_page=0, to_page=10),
        _ev(date(2026, 8, 1), from_page=10, to_page=20),
    ]
    summary = AnalyticsService._compute(events, _counts(), today=_TODAY, window_days=30)

    assert summary.current_streak_days == 0
    assert summary.longest_streak_days == 2


def test_pages_over_time_is_sorted_ascending() -> None:
    events = [
        _ev(date(2026, 8, 3), from_page=0, to_page=5),
        _ev(date(2026, 8, 1), from_page=5, to_page=10),
    ]
    summary = AnalyticsService._compute(events, _counts(), today=_TODAY, window_days=30)

    assert [(p.day, p.pages) for p in summary.pages_over_time] == [
        (date(2026, 8, 1), 5),
        (date(2026, 8, 3), 5),
    ]


# --- caching ------------------------------------------------------------- #


class _FakeRedis:
    def __init__(self, value: bytes | None = None) -> None:
        self.store: dict[str, str] = {}
        self._preset = value
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, Any]] = []

    async def get(self, key: str) -> Any:
        self.get_calls.append(key)
        return self._preset

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self.set_calls.append((key, ex))
        self.store[key] = value


class _FakeEventsRepo:
    def __init__(self, events: list[ReadingEvent]) -> None:
        self._events = events
        self.since: Any = None

    async def list_since(self, since, *, limit: int = 10_000):
        self.since = since
        return self._events


class _FakeProgressRepo:
    def __init__(self, counts: dict[ReadingStatus, int]) -> None:
        self._counts = counts

    async def count_by_status(self, status: ReadingStatus) -> int:
        return self._counts.get(status, 0)


async def test_get_analytics_computes_and_caches_on_miss() -> None:
    redis = _FakeRedis(value=None)
    service = AnalyticsService(redis=redis, ttl_seconds=300)  # type: ignore[arg-type]
    events = _FakeEventsRepo([_ev(date(2026, 8, 3), from_page=0, to_page=10)])
    progress = _FakeProgressRepo(_counts(reading=1))
    user_id = uuid.uuid4()

    summary = await service.get_analytics(
        user_id=user_id, events=events, progress=progress, today=_TODAY, window_days=30
    )

    assert summary.pages_read == 10
    assert redis.set_calls == [(f"analytics:{user_id}:30", 300)]  # cached with TTL


async def test_get_analytics_serves_from_cache_on_hit() -> None:
    cached = AnalyticsService._compute(
        [_ev(date(2026, 8, 3), from_page=0, to_page=99)], _counts(), today=_TODAY, window_days=30
    )
    redis = _FakeRedis(value=cached.model_dump_json().encode())
    service = AnalyticsService(redis=redis, ttl_seconds=300)  # type: ignore[arg-type]
    # Repos would raise if touched — a cache hit must not recompute.
    events = _FakeEventsRepo([])
    progress = _FakeProgressRepo(_counts())

    summary = await service.get_analytics(
        user_id=uuid.uuid4(), events=events, progress=progress, today=_TODAY, window_days=30
    )

    assert summary.pages_read == 99
    assert redis.set_calls == []  # nothing re-cached
    assert events.since is None  # events never fetched
