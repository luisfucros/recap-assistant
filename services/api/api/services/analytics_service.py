"""Reading analytics derived from the append-only reading-event trail (FR-17).

:class:`AnalyticsService` turns a user's :class:`~shared.models.reading.ReadingEvent`
history (plus current status counts) into pace, streaks, and a pages-over-time
series. The event trail — not just the mutable current page — is what makes these
computable and auditable; the results are **cached in Redis** (TTL-refreshed) so
the hot read/update path stays cheap.

Everything is user-isolated: the caller passes user-scoped repositories and the
``user_id`` (from the authenticated context) that keys the cache — analytics for
one user can never fold in another's events.
"""

import uuid
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise

from loguru import logger
from redis.asyncio import Redis

from api.schemas import AnalyticsSummary, PagesOnDay
from shared.core.enums import ReadingEventType, ReadingStatus
from shared.models.reading import ReadingEvent
from shared.repositories import ReadingEventRepository, ReadingProgressRepository


class AnalyticsService:
    """Compute (and cache) a user's reading pace, streaks, and history."""

    def __init__(self, *, redis: Redis, ttl_seconds: int) -> None:
        """Wire the service to Redis and the cache TTL (seconds)."""
        self._redis = redis
        self._ttl = ttl_seconds

    async def get_analytics(
        self,
        *,
        user_id: uuid.UUID,
        events: ReadingEventRepository,
        progress: ReadingProgressRepository,
        today: date,
        window_days: int = 30,
    ) -> AnalyticsSummary:
        """Return the user's analytics over the trailing ``window_days``.

        Served from the Redis cache when warm; otherwise the event trail since the
        window start is aggregated, the result cached for ``ttl_seconds``, and
        returned. ``today`` is injected (not read from the clock here) so the
        window and streak are deterministic and testable.
        """
        cache_key = f"analytics:{user_id}:{window_days}"
        cached = await self._redis.get(cache_key)
        if cached is not None:
            logger.debug("analytics.get: cache hit (window_days={})", window_days)
            return AnalyticsSummary.model_validate_json(cached)

        window_start = datetime.combine(
            today - timedelta(days=window_days - 1), time.min, tzinfo=UTC
        )
        window_events = await events.list_since(window_start)
        status_counts = {
            status: await progress.count_by_status(status)
            for status in (ReadingStatus.READING, ReadingStatus.COMPLETED, ReadingStatus.CANCELLED)
        }
        summary = self._compute(window_events, status_counts, today=today, window_days=window_days)
        await self._redis.set(cache_key, summary.model_dump_json(), ex=self._ttl)
        logger.info(
            "analytics.get: computed (window_days={}, pages_read={}, events={})",
            window_days,
            summary.pages_read,
            len(window_events),
        )
        return summary

    @classmethod
    def _compute(
        cls,
        events: list[ReadingEvent],
        status_counts: dict[ReadingStatus, int],
        *,
        today: date,
        window_days: int,
    ) -> AnalyticsSummary:
        """Aggregate events + status counts into an :class:`AnalyticsSummary` (pure)."""
        pages_by_day = cls._pages_by_day(events)
        pages_read = sum(pages_by_day.values())
        active_days = len(pages_by_day)
        current_streak, longest_streak = cls._streaks(set(pages_by_day), today=today)
        return AnalyticsSummary(
            window_days=window_days,
            pages_read=pages_read,
            active_days=active_days,
            pace_pages_per_day=round(pages_read / active_days, 2) if active_days else 0.0,
            current_streak_days=current_streak,
            longest_streak_days=longest_streak,
            documents_started=status_counts.get(ReadingStatus.READING, 0),
            documents_completed=status_counts.get(ReadingStatus.COMPLETED, 0),
            documents_cancelled=status_counts.get(ReadingStatus.CANCELLED, 0),
            pages_over_time=[
                PagesOnDay(day=day, pages=pages) for day, pages in sorted(pages_by_day.items())
            ],
        )

    @staticmethod
    def _pages_by_day(events: list[ReadingEvent]) -> dict[date, int]:
        """Sum forward page movement per calendar day (pace ignores re-reads back).

        Only ``POSITION_ADVANCED`` events with a positive page delta count, so
        jumping backwards or a bare status change never inflates pages read.
        """
        totals: dict[date, int] = defaultdict(int)
        for event in events:
            if event.type is not ReadingEventType.POSITION_ADVANCED:
                continue
            if event.from_page is None or event.to_page is None:
                continue
            gained = event.to_page - event.from_page
            if gained > 0:
                totals[event.occurred_at.date()] += gained
        return dict(totals)

    @staticmethod
    def _streaks(active_days: set[date], *, today: date) -> tuple[int, int]:
        """Return ``(current_streak, longest_streak)`` in consecutive active days.

        The current streak counts back from ``today`` while each prior day is
        active (0 if the user hasn't read today); the longest streak is the
        longest run of consecutive active days anywhere in the window.
        """
        if not active_days:
            return (0, 0)

        ordered = sorted(active_days)
        longest = run = 1
        for previous, current in pairwise(ordered):
            run = run + 1 if (current - previous).days == 1 else 1
            longest = max(longest, run)

        current_streak = 0
        cursor = today
        while cursor in active_days:
            current_streak += 1
            cursor -= timedelta(days=1)
        return (current_streak, longest)
