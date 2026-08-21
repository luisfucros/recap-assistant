"""Integration tests for UsageService against real Postgres (NFR-13).

Exercises the per-user usage-event write path (token spend, tool-call counts)
over real SQL, and confirms aggregation correctness — including under a
larger volume of events, standing in for "verify metrics are correct under
load" without needing an actual load-testing harness.

Redis (the usage cache) is faked at the boundary — always a cache miss — so
every ``get_usage`` call recomputes from the real event trail; every other
store is real.
"""

import uuid
from datetime import UTC, datetime

import pytest
from api.services.usage_service import UsageService

from shared.models.user import User
from shared.repositories import UsageEventRepository, UserRepository

pytestmark = pytest.mark.integration

_TODAY = datetime.now(UTC).date()


class _FakeRedis:
    """Cache miss always → usage is recomputed from real SQL on every call."""

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        return None


async def _seed_user(db_sessionmaker, email: str) -> uuid.UUID:
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email=email))
        await session.commit()
        return user.id


async def test_record_and_aggregate_token_usage_and_tool_calls(db_sessionmaker) -> None:
    user_id = await _seed_user(db_sessionmaker, "usage-basic@example.com")
    service = UsageService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]

    async with db_sessionmaker() as session:
        repo = UsageEventRepository(session, user_id)
        await service.record_token_usage(
            session=session, usage=repo, prompt_tokens=100, completion_tokens=20
        )
        await service.record_token_usage(
            session=session, usage=repo, prompt_tokens=50, completion_tokens=10
        )
        await service.record_tool_call(session=session, usage=repo, tool_name="retrieve_chunks")
        await service.record_tool_call(session=session, usage=repo, tool_name="retrieve_chunks")
        await service.record_tool_call(session=session, usage=repo, tool_name="web_search")

    async with db_sessionmaker() as session:
        summary = await service.get_usage(
            user_id=user_id,
            usage=UsageEventRepository(session, user_id),
            today=_TODAY,
            window_days=30,
        )

    assert summary.prompt_tokens == 150
    assert summary.completion_tokens == 30
    assert summary.total_tokens == 180
    assert summary.tool_calls == 3
    assert {c.tool: c.count for c in summary.tool_calls_by_tool} == {
        "retrieve_chunks": 2,
        "web_search": 1,
    }


async def test_usage_is_isolated_between_users(db_sessionmaker) -> None:
    user_a = await _seed_user(db_sessionmaker, "usage-a@example.com")
    user_b = await _seed_user(db_sessionmaker, "usage-b@example.com")
    service = UsageService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]

    async with db_sessionmaker() as session:
        await service.record_token_usage(
            session=session,
            usage=UsageEventRepository(session, user_a),
            prompt_tokens=1000,
            completion_tokens=1000,
        )
        await service.record_tool_call(
            session=session, usage=UsageEventRepository(session, user_a), tool_name="web_search"
        )

    async with db_sessionmaker() as session:
        summary_b = await service.get_usage(
            user_id=user_b,
            usage=UsageEventRepository(session, user_b),
            today=_TODAY,
            window_days=30,
        )

    assert summary_b.total_tokens == 0
    assert summary_b.tool_calls == 0


async def test_usage_aggregates_correctly_under_a_larger_volume_of_events(db_sessionmaker) -> None:
    """Stands in for "verify metrics are correct under load" (NFR-13)."""
    user_id = await _seed_user(db_sessionmaker, "usage-load@example.com")
    service = UsageService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]

    tool_names = ["retrieve_chunks", "summarize", "web_search"]
    async with db_sessionmaker() as session:
        repo = UsageEventRepository(session, user_id)
        for _ in range(50):
            await service.record_token_usage(
                session=session, usage=repo, prompt_tokens=10, completion_tokens=2
            )
        for index in range(90):
            await service.record_tool_call(
                session=session, usage=repo, tool_name=tool_names[index % len(tool_names)]
            )

    async with db_sessionmaker() as session:
        summary = await service.get_usage(
            user_id=user_id,
            usage=UsageEventRepository(session, user_id),
            today=_TODAY,
            window_days=30,
        )

    assert summary.prompt_tokens == 500
    assert summary.completion_tokens == 100
    assert summary.total_tokens == 600
    assert summary.tool_calls == 90
    assert {c.tool: c.count for c in summary.tool_calls_by_tool} == {
        "retrieve_chunks": 30,
        "summarize": 30,
        "web_search": 30,
    }
