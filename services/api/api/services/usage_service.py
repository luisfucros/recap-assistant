"""Per-user token spend and tool-call counts, derived from the usage-event trail (NFR-13).

Prometheus's ``recap_llm_tokens_total``/``recap_operation_seconds`` are the
always-on SLI layer, but they're deliberately low-cardinality — no ``user_id``
label, or one time series per user would blow up the metric's cardinality.
:class:`UsageService` is the durable, per-user counterpart: it records raw
events (one row per answer-model LLM call's token counts, one per executed
tool call) and aggregates them into a cached :class:`~api.schemas.UsageSummary`,
mirroring how :class:`~api.services.analytics_service.AnalyticsService` turns
the reading-event trail into cached reading analytics.

Everything is user-isolated: the caller passes a user-scoped
:class:`~shared.repositories.usage_repository.UsageEventRepository` (its own
``user_id`` sources every write and read), so one user's usage can never fold
into another's.
"""

import uuid
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ToolCallCount, UsageSummary
from shared.core.enums import UsageEventType
from shared.models.usage import UsageEvent
from shared.repositories import UsageEventRepository


class UsageService:
    """Record per-user usage events and compute (and cache) their aggregate."""

    def __init__(self, *, redis: Redis, ttl_seconds: int) -> None:
        """Wire the service to Redis and the cache TTL (seconds)."""
        self._redis = redis
        self._ttl = ttl_seconds

    async def record_token_usage(
        self,
        *,
        session: AsyncSession,
        usage: UsageEventRepository,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Append one token-usage event for an answer-model LLM call.

        A no-op when both counts are zero (a fake/provider that didn't report
        usage) — the trail should only ever carry real spend.
        """
        if not prompt_tokens and not completion_tokens:
            return
        await usage.add(
            UsageEvent(
                user_id=usage.user_id,
                type=UsageEventType.TOKEN_USAGE,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        await session.commit()

    async def record_tool_call(
        self, *, session: AsyncSession, usage: UsageEventRepository, tool_name: str
    ) -> None:
        """Append one tool-call event for an executed tool."""
        await usage.add(
            UsageEvent(user_id=usage.user_id, type=UsageEventType.TOOL_CALL, tool_name=tool_name)
        )
        await session.commit()

    async def get_usage(
        self,
        *,
        user_id: uuid.UUID,
        usage: UsageEventRepository,
        today: date,
        window_days: int = 30,
    ) -> UsageSummary:
        """Return the user's token spend and tool-call counts over the trailing window.

        Served from the Redis cache when warm; otherwise the event trail since
        the window start is aggregated, the result cached for ``ttl_seconds``,
        and returned. ``today`` is injected (not read from the clock here) so
        the window is deterministic and testable.
        """
        cache_key = f"usage:{user_id}:{window_days}"
        cached = await self._redis.get(cache_key)
        if cached is not None:
            return UsageSummary.model_validate_json(cached)

        window_start = datetime.combine(
            today - timedelta(days=window_days - 1), time.min, tzinfo=UTC
        )
        window_events = await usage.list_since(window_start)
        summary = self._compute(window_events, window_days=window_days)
        await self._redis.set(cache_key, summary.model_dump_json(), ex=self._ttl)
        return summary

    @classmethod
    def _compute(cls, events: list[UsageEvent], *, window_days: int) -> UsageSummary:
        """Aggregate usage events into a :class:`~api.schemas.UsageSummary` (pure)."""
        prompt_tokens = 0
        completion_tokens = 0
        tool_counts: dict[str, int] = defaultdict(int)
        for event in events:
            if event.type is UsageEventType.TOKEN_USAGE:
                prompt_tokens += event.prompt_tokens or 0
                completion_tokens += event.completion_tokens or 0
            elif event.type is UsageEventType.TOOL_CALL and event.tool_name:
                tool_counts[event.tool_name] += 1
        return UsageSummary(
            window_days=window_days,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tool_calls=sum(tool_counts.values()),
            tool_calls_by_tool=[
                ToolCallCount(tool=tool, count=count) for tool, count in sorted(tool_counts.items())
            ],
        )
