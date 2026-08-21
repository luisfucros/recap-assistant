"""Integration tests for the stuck-document sweep against real Postgres.

``test_sweep.py`` (unit) fakes ``find_stuck_documents`` entirely to test the
fan-out logic in isolation; here the real query runs against a real
``updated_at`` column, verifying the part that actually matters — that only
``pending``/``processing`` rows older than the threshold come back, and every
other status/age combination is correctly excluded.
"""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from ingestion.sweep import find_stuck_documents, sweep_stuck_documents

from shared.core.enums import DocumentFormat, DocumentStatus
from shared.models.document import Document
from shared.models.user import User
from shared.repositories import DocumentRepository, UserRepository

pytestmark = pytest.mark.integration


async def _make_user(sessionmaker) -> uuid.UUID:
    async with sessionmaker() as session:
        user = await UserRepository(session).add(User(email="reader@example.com"))
        await session.commit()
        return user.id


async def _make_document(
    sessionmaker,
    user_id: uuid.UUID,
    *,
    status: DocumentStatus,
    updated_at: datetime,
    content_sha256: str,
) -> uuid.UUID:
    """Insert a document with an explicit ``updated_at``, bypassing the server default."""
    async with sessionmaker() as session:
        document = Document(
            user_id=user_id,
            filename="book.pdf",
            object_key=f"k-{content_sha256}",
            content_sha256=content_sha256,
            format=DocumentFormat.PDF,
            embed_model="m",
            status=status,
            updated_at=updated_at,
        )
        await DocumentRepository(session, user_id).add(document)
        await session.commit()
        return document.id


async def test_find_stuck_documents_returns_only_stale_pending_and_processing(
    db_sessionmaker,
) -> None:
    resources = SimpleNamespace(sessionmaker=db_sessionmaker)
    user_id = await _make_user(db_sessionmaker)
    now = datetime.now(UTC)

    stuck_pending = await _make_document(
        db_sessionmaker,
        user_id,
        status=DocumentStatus.PENDING,
        updated_at=now - timedelta(seconds=2000),
        content_sha256="stuck-pending",
    )
    stuck_processing = await _make_document(
        db_sessionmaker,
        user_id,
        status=DocumentStatus.PROCESSING,
        updated_at=now - timedelta(seconds=2000),
        content_sha256="stuck-processing",
    )
    # Not stale yet — still well within a healthy run's expected duration.
    await _make_document(
        db_sessionmaker,
        user_id,
        status=DocumentStatus.PENDING,
        updated_at=now - timedelta(seconds=5),
        content_sha256="fresh-pending",
    )
    # Old, but terminal — nothing left to re-enqueue.
    await _make_document(
        db_sessionmaker,
        user_id,
        status=DocumentStatus.INDEXED,
        updated_at=now - timedelta(seconds=2000),
        content_sha256="old-indexed",
    )
    await _make_document(
        db_sessionmaker,
        user_id,
        status=DocumentStatus.FAILED,
        updated_at=now - timedelta(seconds=2000),
        content_sha256="old-failed",
    )

    stuck = await find_stuck_documents(resources, now=now, stuck_after_seconds=900)

    assert {document_id for document_id, _ in stuck} == {stuck_pending, stuck_processing}
    assert all(u == user_id for _, u in stuck)


async def test_sweep_stuck_documents_dispatches_only_the_stale_rows(db_sessionmaker) -> None:
    resources = SimpleNamespace(sessionmaker=db_sessionmaker)
    user_id = await _make_user(db_sessionmaker)
    now = datetime.now(UTC)
    stuck_id = await _make_document(
        db_sessionmaker,
        user_id,
        status=DocumentStatus.PROCESSING,
        updated_at=now - timedelta(seconds=2000),
        content_sha256="stuck",
    )
    await _make_document(
        db_sessionmaker,
        user_id,
        status=DocumentStatus.PENDING,
        updated_at=now,
        content_sha256="fresh",
    )
    dispatched: list[tuple[str, str]] = []

    count = await sweep_stuck_documents(
        resources,
        now=now,
        stuck_after_seconds=900,
        dispatch=lambda document_id, uid: dispatched.append((document_id, uid)),
    )

    assert count == 1
    assert dispatched == [(str(stuck_id), str(user_id))]
