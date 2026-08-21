"""Integration tests for the ingestion pipeline against real Postgres+Qdrant+MinIO.

Runs ``run_ingestion`` end-to-end with a deterministic in-process embedder (the
only mocked boundary — embeddings are an external API). Verifies the design's
load-bearing guarantees against real infra: a document becomes ``indexed`` with
its chunks persisted and vectors upserted, the run is idempotent, and a mid-run
Qdrant failure leaves the document retryable (never ``indexed`` with missing
vectors, no ``document.indexed`` event). Also exercises the outbox relay's real
fetch/mark-processed SQL.
"""

import uuid

import pytest
from api.services.ingestion_service import IngestionService
from ingestion.outbox_relay import drain_outbox
from ingestion.pipeline import run_ingestion

from shared.core.enums import DocumentFormat, DocumentStatus, Language
from shared.core.events import DOCUMENT_INDEXED, DOCUMENT_UPLOADED
from shared.models.user import User
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    OutboxRepository,
    UserRepository,
)

pytestmark = pytest.mark.integration


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal multi-page PDF whose pages contain the given text.

    Hand-assembled (no rendering library available) but valid enough for pypdf to
    extract per-page text — enough to drive parse→chunk with real page structure.
    """
    parts: list[str] = ["%PDF-1.4\n"]
    offsets: dict[int, int] = {}

    def add(num: int, body: str) -> None:
        offsets[num] = sum(len(p.encode("latin-1")) for p in parts)
        parts.append(f"{num} 0 obj\n{body}\nendobj\n")

    n = len(pages)
    page_objs = [3 + 2 * i for i in range(n)]
    font_obj = 3 + 2 * n
    kids = " ".join(f"{pid} 0 R" for pid in page_objs)
    add(1, "<</Type/Catalog/Pages 2 0 R>>")
    add(2, f"<</Type/Pages/Kids[{kids}]/Count {n}>>")
    for i, text in enumerate(pages):
        page_obj, content_obj = page_objs[i], page_objs[i] + 1
        stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET"
        add(
            page_obj,
            f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            f"/Contents {content_obj} 0 R/Resources<</Font<</F1 {font_obj} 0 R>>>>>>",
        )
        add(content_obj, f"<</Length {len(stream)}>>\nstream\n{stream}\nendstream")
    add(font_obj, "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")

    xref_pos = sum(len(p.encode("latin-1")) for p in parts)
    xref = ["xref\n", f"0 {font_obj + 1}\n", "0000000000 65535 f \n"]
    xref += [f"{offsets[i]:010d} 00000 n \n" for i in range(1, font_obj + 1)]
    parts.append("".join(xref))
    parts.append(f"trailer\n<</Size {font_obj + 1}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF")
    return "".join(parts).encode("latin-1")


PDF = make_pdf(
    [
        "This is the first page of an English book about the sea and whales and ships.",
        "The second page continues the English narrative about the ocean voyage.",
    ]
)


@pytest.fixture
async def resources(test_settings, fake_embedder, storage, qdrant_client, db_engine):
    """Ingestion resources wired to the test infra, with the embedder faked.

    Depends on ``storage``/``qdrant_client`` (bucket + collection setup and
    skip-if-down) and ``db_engine`` (schema + per-test truncation).
    """
    from ingestion.resources import IngestionResources

    res = IngestionResources(test_settings)
    res.embedder = fake_embedder  # shadow the cached_property with a deterministic embedder
    try:
        yield res
    finally:
        await res.aclose()


async def _pending_document(resources, sessionmaker, *, data: bytes = PDF):
    """Create a user and upload a pending document (object + row + outbox event)."""
    async with sessionmaker() as session:
        user = await UserRepository(session).add(User(email="reader@example.com"))
        await session.commit()
        user_id = user.id
    service = IngestionService(
        storage=resources.storage, embed_model=resources.settings.embedding_model
    )
    async with sessionmaker() as session:
        doc = await service.upload(
            session=session,
            documents=DocumentRepository(session, user_id),
            outbox=OutboxRepository(session),
            user_id=user_id,
            filename="book.pdf",
            content_type="application/pdf",
            document_format=DocumentFormat.PDF,
            data=data,
        )
    return user_id, doc.id


async def test_pipeline_indexes_document_with_chunks_and_vectors(
    resources, db_sessionmaker
) -> None:
    user_id, document_id = await _pending_document(resources, db_sessionmaker)

    await run_ingestion(resources, document_id=document_id, user_id=user_id)

    async with db_sessionmaker() as session:
        doc = await DocumentRepository(session, user_id).get(document_id)
        assert doc.status is DocumentStatus.INDEXED
        assert doc.page_count == 2
        assert doc.indexed_at is not None
        assert doc.language is Language.EN  # detected from the English text

        chunks = await ChunkRepository(session, user_id).list_by_document(document_id)
        assert len(chunks) >= 1
        assert all(c.vector_id for c in chunks)
        # The chunk spans both pages of the short document.
        assert chunks[0].page_start == 1
        assert chunks[-1].page_end == 2

        events = await OutboxRepository(session).fetch_unprocessed(limit=50)
        indexed = [
            e for e in events if e.aggregate_id == document_id and e.event_type == DOCUMENT_INDEXED
        ]
        assert len(indexed) == 1
        assert indexed[0].payload["chunk_count"] == len(chunks)

    # Vectors are actually in Qdrant, one per chunk.
    count = await resources.qdrant.count(
        collection_name=resources.settings.qdrant_chunks_collection
    )
    assert count.count == len(chunks)


async def test_pipeline_is_idempotent(resources, db_sessionmaker) -> None:
    user_id, document_id = await _pending_document(resources, db_sessionmaker)

    await run_ingestion(resources, document_id=document_id, user_id=user_id)
    async with db_sessionmaker() as session:
        first = await ChunkRepository(session, user_id).list_by_document(document_id)

    # A second run replaces (not duplicates) chunks and vectors.
    await run_ingestion(resources, document_id=document_id, user_id=user_id)
    async with db_sessionmaker() as session:
        second = await ChunkRepository(session, user_id).list_by_document(document_id)

    assert len(second) == len(first)
    count = await resources.qdrant.count(
        collection_name=resources.settings.qdrant_chunks_collection
    )
    assert count.count == len(second)


async def test_qdrant_failure_leaves_document_retryable(
    resources, db_sessionmaker, monkeypatch
) -> None:
    user_id, document_id = await _pending_document(resources, db_sessionmaker)

    async def _boom(*args, **kwargs):
        raise ConnectionError("qdrant is down")

    # Fail the vector upsert mid-run (after the PROCESSING commit, before terminal).
    monkeypatch.setattr(resources.qdrant, "upsert", _boom)
    with pytest.raises(ConnectionError):
        await run_ingestion(resources, document_id=document_id, user_id=user_id)

    async with db_sessionmaker() as session:
        doc = await DocumentRepository(session, user_id).get(document_id)
        # Never indexed on a partial run; left processing so a retry can finish it.
        assert doc.status is DocumentStatus.PROCESSING
        assert await ChunkRepository(session, user_id).list_by_document(document_id) == []
        events = await OutboxRepository(session).fetch_unprocessed(limit=50)
        assert not any(e.event_type == DOCUMENT_INDEXED for e in events)

    # Recovery: with Qdrant back, a re-run completes cleanly.
    monkeypatch.undo()
    await run_ingestion(resources, document_id=document_id, user_id=user_id)
    async with db_sessionmaker() as session:
        doc = await DocumentRepository(session, user_id).get(document_id)
        assert doc.status is DocumentStatus.INDEXED


async def test_outbox_relay_dispatches_and_marks_processed(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        await OutboxRepository(session).add(
            event_type=DOCUMENT_UPLOADED,
            aggregate_id=uuid.uuid4(),
            payload={"document_id": str(uuid.uuid4()), "user_id": str(uuid.uuid4())},
        )
        await session.commit()

    dispatched: list[str] = []
    async with db_sessionmaker() as session:
        count = await drain_outbox(
            OutboxRepository(session),
            batch_size=100,
            dispatch=lambda event_type, _payload: dispatched.append(event_type),
        )
        await session.commit()

    assert count == 1
    assert dispatched == [DOCUMENT_UPLOADED]
    async with db_sessionmaker() as session:
        assert await OutboxRepository(session).fetch_unprocessed(limit=10) == []
