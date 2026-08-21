"""Integration tests for ProgressService + AnalyticsService against real Postgres.

Exercises the reading-state write path and analytics aggregation over real SQL:

* Updating a position through :class:`ProgressService` writes a ``reading_progress``
  row **and** appends ``reading_events`` (the append-only analytics trail).
* The event trail and analytics are **user-isolated** — one user's events never
  surface in another's repository reads or computed metrics.

Redis (the analytics cache) is faked at the boundary so the computation runs
against the real event/progress SQL; every other store is real.
"""

import uuid
from datetime import UTC, datetime

import pytest
from api.services.analytics_service import AnalyticsService
from api.services.progress_service import ProgressService

from shared.core.enums import DocumentFormat, ReadingEventType, ReadingStatus
from shared.models.document import Document
from shared.models.user import User
from shared.repositories import (
    DocumentRepository,
    ReadingEventRepository,
    ReadingProgressRepository,
    UserRepository,
)

pytestmark = pytest.mark.integration


class _FakeRedis:
    """Cache miss always → the analytics service computes from real SQL each call."""

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        return None


async def _seed_user_and_doc(db_sessionmaker, email: str) -> tuple[uuid.UUID, uuid.UUID]:
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email=email))
        await session.commit()
        user_id = user.id
    doc = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/k.pdf",
        content_sha256=uuid.uuid4().hex,
        format=DocumentFormat.PDF,
        embed_model="m",
        page_count=100,
    )
    async with db_sessionmaker() as session:
        await DocumentRepository(session, user_id).add(doc)
        await session.commit()
    return user_id, doc.id


async def test_update_writes_progress_row_and_events(db_sessionmaker) -> None:
    user_id, doc_id = await _seed_user_and_doc(db_sessionmaker, "reader@example.com")
    service = ProgressService()

    async with db_sessionmaker() as session:
        await service.update_progress(
            session=session,
            documents=DocumentRepository(session, user_id),
            progress=ReadingProgressRepository(session, user_id),
            events=ReadingEventRepository(session, user_id),
            document_id=doc_id,
            current_page=25,
        )

    # A progress row exists at page 25, auto-promoted to reading.
    async with db_sessionmaker() as session:
        row = await ReadingProgressRepository(session, user_id).get_by_document(doc_id)
        assert row is not None
        assert row.current_page == 25
        assert row.status is ReadingStatus.READING

    # Events were appended: a forward move and the status change.
    async with db_sessionmaker() as session:
        events = await ReadingEventRepository(session, user_id).list_since(
            datetime(2000, 1, 1, tzinfo=UTC)
        )
    types = {e.type for e in events}
    assert ReadingEventType.POSITION_ADVANCED in types
    assert ReadingEventType.STATUS_CHANGED in types
    advanced = next(e for e in events if e.type is ReadingEventType.POSITION_ADVANCED)
    assert (advanced.from_page, advanced.to_page) == (0, 25)


async def test_events_and_analytics_are_user_isolated(db_sessionmaker) -> None:
    user_a, doc_a = await _seed_user_and_doc(db_sessionmaker, "a@example.com")
    user_b, doc_b = await _seed_user_and_doc(db_sessionmaker, "b@example.com")
    service = ProgressService()

    # A reads 40 pages; B reads 5.
    async with db_sessionmaker() as session:
        await service.update_progress(
            session=session,
            documents=DocumentRepository(session, user_a),
            progress=ReadingProgressRepository(session, user_a),
            events=ReadingEventRepository(session, user_a),
            document_id=doc_a,
            current_page=40,
        )
    async with db_sessionmaker() as session:
        await service.update_progress(
            session=session,
            documents=DocumentRepository(session, user_b),
            progress=ReadingProgressRepository(session, user_b),
            events=ReadingEventRepository(session, user_b),
            document_id=doc_b,
            current_page=5,
        )

    # B's event repo never sees A's events.
    async with db_sessionmaker() as session:
        b_events = await ReadingEventRepository(session, user_b).list_since(
            datetime(2000, 1, 1, tzinfo=UTC)
        )
    assert all(e.document_id == doc_b for e in b_events)

    # Analytics reflect only the caller's own reading.
    analytics = AnalyticsService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]
    today = datetime.now(UTC).date()
    async with db_sessionmaker() as session:
        summary_a = await analytics.get_analytics(
            user_id=user_a,
            events=ReadingEventRepository(session, user_a),
            progress=ReadingProgressRepository(session, user_a),
            today=today,
        )
    async with db_sessionmaker() as session:
        summary_b = await analytics.get_analytics(
            user_id=user_b,
            events=ReadingEventRepository(session, user_b),
            progress=ReadingProgressRepository(session, user_b),
            today=today,
        )

    assert summary_a.pages_read == 40
    assert summary_b.pages_read == 5
    assert summary_a.documents_started == 1
    assert summary_b.documents_started == 1
