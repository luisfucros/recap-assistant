"""Integration tests for ``IngestionService.upload``/``.retry`` against real Postgres + MinIO.

Verifies the upload handoff end-to-end at the data layer: the original is stored
content-addressed, a ``pending`` row and a ``document.uploaded`` outbox event
commit together, and duplicate rejection is race-safe under genuine concurrency
(the unit tier can only simulate the ``IntegrityError``). External APIs aren't
involved — no parsing/embedding happens on this path. ``.retry`` is covered here
too since it shares the same real repositories and outbox.
"""

import asyncio

import pytest
from api.services.ingestion_service import (
    DocumentNotFailedError,
    DuplicateDocumentError,
    IngestionService,
)

from shared.core.enums import DocumentFormat, DocumentStatus
from shared.core.events import DOCUMENT_UPLOADED
from shared.ingestion_core.content_address import object_key, sha256_hexdigest
from shared.models.document import Document
from shared.models.user import User
from shared.repositories import DocumentRepository, OutboxRepository, UserRepository

pytestmark = pytest.mark.integration

PDF_BYTES = b"%PDF-1.7\nfake pdf body\n%%EOF"


async def _make_user(sessionmaker, email: str = "reader@example.com") -> User:
    async with sessionmaker() as session:
        user = await UserRepository(session).add(User(email=email))
        await session.commit()
        return user


async def _upload(service, sessionmaker, user_id, *, data=PDF_BYTES, filename="book.pdf"):
    async with sessionmaker() as session:
        return await service.upload(
            session=session,
            documents=DocumentRepository(session, user_id),
            outbox=OutboxRepository(session),
            user_id=user_id,
            filename=filename,
            content_type="application/pdf",
            document_format=DocumentFormat.PDF,
            data=data,
        )


async def test_upload_persists_row_object_and_event(db_sessionmaker, storage) -> None:
    user = await _make_user(db_sessionmaker)
    service = IngestionService(storage=storage, embed_model="test-model")

    doc = await _upload(service, db_sessionmaker, user.id)

    sha = sha256_hexdigest(PDF_BYTES)
    async with db_sessionmaker() as session:
        stored = await DocumentRepository(session, user.id).get(doc.id)
        assert stored is not None
        assert stored.status is DocumentStatus.PENDING
        assert stored.content_sha256 == sha
        assert stored.object_key == object_key(user.id, sha, "pdf")
        # The original bytes are in object storage, content-addressed.
        assert await storage.get(stored.object_key) == PDF_BYTES
        # Exactly one matching outbox event committed alongside the row.
        events = await OutboxRepository(session).fetch_unprocessed(limit=10)
        matching = [e for e in events if e.aggregate_id == doc.id]
        assert len(matching) == 1
        assert matching[0].event_type == DOCUMENT_UPLOADED
        assert matching[0].payload == {"document_id": str(doc.id), "user_id": str(user.id)}


async def test_reupload_same_content_is_duplicate(db_sessionmaker, storage) -> None:
    user = await _make_user(db_sessionmaker)
    service = IngestionService(storage=storage, embed_model="test-model")

    first = await _upload(service, db_sessionmaker, user.id)
    with pytest.raises(DuplicateDocumentError) as excinfo:
        await _upload(service, db_sessionmaker, user.id, filename="again.pdf")
    assert excinfo.value.existing_id == first.id


async def test_concurrent_identical_uploads_are_race_safe(db_sessionmaker, storage) -> None:
    user = await _make_user(db_sessionmaker)
    service = IngestionService(storage=storage, embed_model="test-model")

    # Two genuinely concurrent uploads of identical content (separate sessions).
    results = await asyncio.gather(
        _upload(service, db_sessionmaker, user.id),
        _upload(service, db_sessionmaker, user.id),
        return_exceptions=True,
    )
    successes = [r for r in results if isinstance(r, Document)]
    duplicates = [r for r in results if isinstance(r, DuplicateDocumentError)]

    # Exactly one wins; the other is rejected as a duplicate pointing at the winner.
    assert len(successes) == 1
    assert len(duplicates) == 1
    assert duplicates[0].existing_id == successes[0].id
    # And the DB holds exactly one row for that content.
    async with db_sessionmaker() as session:
        assert await DocumentRepository(session, user.id).count() == 1


async def test_same_content_two_users_ingest_independently(db_sessionmaker, storage) -> None:
    alice = await _make_user(db_sessionmaker, "alice@example.com")
    bob = await _make_user(db_sessionmaker, "bob@example.com")
    service = IngestionService(storage=storage, embed_model="test-model")

    doc_a = await _upload(service, db_sessionmaker, alice.id)
    doc_b = await _upload(service, db_sessionmaker, bob.id)

    # Two independent documents for identical bytes — isolation over dedup.
    assert doc_a.id != doc_b.id
    assert doc_a.object_key != doc_b.object_key  # keys namespaced by user
    async with db_sessionmaker() as session:
        assert await DocumentRepository(session, alice.id).count() == 1
        assert await DocumentRepository(session, bob.id).count() == 1


async def test_retry_reenqueues_a_failed_document_without_reuploading(
    db_sessionmaker, storage
) -> None:
    user = await _make_user(db_sessionmaker)
    service = IngestionService(storage=storage, embed_model="test-model")
    doc = await _upload(service, db_sessionmaker, user.id)

    async with db_sessionmaker() as session:
        documents = DocumentRepository(session, user.id)
        failed = await documents.get(doc.id)
        failed.status = DocumentStatus.FAILED
        failed.failure_reason = "parse failed: corrupt bytes"
        await session.commit()

    async with db_sessionmaker() as session:
        retried = await service.retry(
            session=session,
            documents=DocumentRepository(session, user.id),
            outbox=OutboxRepository(session),
            document_id=doc.id,
        )

    assert retried.status is DocumentStatus.PENDING
    assert retried.failure_reason is None
    async with db_sessionmaker() as session:
        stored = await DocumentRepository(session, user.id).get(doc.id)
        assert stored.status is DocumentStatus.PENDING
        assert stored.failure_reason is None
        # The original object is untouched — a retry never re-uploads.
        assert await storage.get(stored.object_key) == PDF_BYTES
        # A fresh ingestion event was enqueued alongside the original upload's
        # (neither has been relayed/marked processed in this test).
        events = await OutboxRepository(session).fetch_unprocessed(limit=10)
        matching = [e for e in events if e.aggregate_id == doc.id]
        assert len(matching) == 2
        assert all(e.event_type == DOCUMENT_UPLOADED for e in matching)


async def test_retry_rejects_a_non_failed_document(db_sessionmaker, storage) -> None:
    user = await _make_user(db_sessionmaker)
    service = IngestionService(storage=storage, embed_model="test-model")
    doc = await _upload(service, db_sessionmaker, user.id)  # still pending

    async with db_sessionmaker() as session:
        with pytest.raises(DocumentNotFailedError):
            await service.retry(
                session=session,
                documents=DocumentRepository(session, user.id),
                outbox=OutboxRepository(session),
                document_id=doc.id,
            )
