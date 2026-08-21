"""Integration tests for RetrievalService against real Postgres + Qdrant.

Seeds two users, each with an indexed document (chunk rows in Postgres + vectors
in Qdrant), then drives :class:`RetrievalService` to verify the load-bearing
guarantees against real infrastructure:

* **Isolation** — a search for user A never returns user B's chunks, because the
  ``user_id`` payload filter is injected by the store from the passed context.
* **Read-range** — with a recorded position, results default to
  ``page_end <= current_page``; ``include_unread`` lifts that bound.
* **Spoiler-safe** — when on, the read-range bound is hard (``include_unread`` is
  ignored).
"""

import uuid

import pytest
from api.services.retrieval_service import RetrievalService

from shared.core.enums import DocumentFormat, ReadingStatus
from shared.models.document import Chunk, Document
from shared.models.reading import ReadingProgress
from shared.models.user import User
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    ReadingProgressRepository,
    UserRepository,
)
from shared.vectorstore import ChunkVectorStore, build_chunk_payload, chunk_point_id

pytestmark = pytest.mark.integration

_DIM = 8


class _FakeEmbedder:
    """Deterministic embedder: same vector for every text (ranking is irrelevant;
    these tests assert the *filter*, not similarity ordering)."""

    @property
    def dim(self) -> int:
        return _DIM

    async def embed(self, texts, *, batch_size=None) -> list[list[float]]:
        return [[0.5] * _DIM for _ in texts]


async def _seed_user_with_pages(db_sessionmaker, store, email: str, pages: list[int]):
    """Create a user + one document + one chunk per page (with a vector each)."""
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email=email))
        await session.commit()
        user_id = user.id

    doc = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/sha256/{email}.pdf",
        content_sha256=uuid.uuid4().hex,
        format=DocumentFormat.PDF,
        embed_model="test-model",
        page_count=max(pages),
    )
    async with db_sessionmaker() as session:
        await DocumentRepository(session, user_id).add(doc)
        await session.commit()

    chunks = [
        Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            user_id=user_id,
            ordinal=i,
            page_start=page,
            page_end=page,
            text=f"{email} page {page}",
            content_hash=uuid.uuid4().hex,
            vector_id=None,
        )
        for i, page in enumerate(pages)
    ]
    for chunk in chunks:
        chunk.vector_id = chunk_point_id(chunk.id)
    async with db_sessionmaker() as session:
        await ChunkRepository(session, user_id).add_many(chunks)
        await session.commit()

    await store.upsert(
        ids=[c.vector_id for c in chunks],
        vectors=[[0.5] * _DIM for _ in chunks],
        payloads=[build_chunk_payload(c, title="Book", author="A", language=None) for c in chunks],
    )
    return user_id, doc


async def _record_page(db_sessionmaker, user_id: uuid.UUID, document_id: uuid.UUID, page: int):
    """Insert a reading_progress row at ``page`` for read-range tests."""
    async with db_sessionmaker() as session:
        session.add(
            ReadingProgress(
                user_id=user_id,
                document_id=document_id,
                current_page=page,
                last_summarized_page=0,
                status=ReadingStatus.READING,
            )
        )
        await session.commit()


def _service(qdrant_client, test_settings) -> RetrievalService:
    return RetrievalService(
        embedder=_FakeEmbedder(),
        vector_store=ChunkVectorStore(
            qdrant_client, collection=test_settings.qdrant_chunks_collection
        ),
        settings=test_settings,
    )


async def test_search_is_isolated_to_the_user(
    db_sessionmaker, qdrant_client, test_settings
) -> None:
    collection = test_settings.qdrant_chunks_collection
    store = ChunkVectorStore(qdrant_client, collection=collection, dim=_DIM)
    await store.ensure_collection()

    user_a, _ = await _seed_user_with_pages(db_sessionmaker, store, "a@example.com", [1, 2, 3])
    user_b, _ = await _seed_user_with_pages(db_sessionmaker, store, "b@example.com", [1, 2, 3])

    service = _service(qdrant_client, test_settings)
    async with db_sessionmaker() as session:
        result = await service.retrieve(
            query="page",
            user_id=user_a,
            progress=ReadingProgressRepository(session, user_a),
            chunks=ChunkRepository(session, user_a),
            include_unread=True,
        )

    assert result.chunks  # A has hits
    # Every returned chunk is A's (its text is prefixed with A's email); B's
    # vectors sit in the same collection but are filtered out by the payload
    # user_id — even though the query and vectors are identical across users.
    assert all(c.text.startswith("a@example.com") for c in result.chunks)
    assert user_b != user_a


async def test_read_range_bounds_to_current_page(
    db_sessionmaker, qdrant_client, test_settings
) -> None:
    collection = test_settings.qdrant_chunks_collection
    store = ChunkVectorStore(qdrant_client, collection=collection, dim=_DIM)
    await store.ensure_collection()

    user_id, doc = await _seed_user_with_pages(
        db_sessionmaker, store, "reader@example.com", [1, 2, 5, 8, 10]
    )
    await _record_page(db_sessionmaker, user_id, doc.id, page=5)

    service = _service(qdrant_client, test_settings)

    # Default: bounded to page_end <= 5.
    async with db_sessionmaker() as session:
        bounded = await service.retrieve(
            query="page",
            user_id=user_id,
            progress=ReadingProgressRepository(session, user_id),
            chunks=ChunkRepository(session, user_id),
            document_id=doc.id,
        )
    assert bounded.chunks
    assert all(c.page_end <= 5 for c in bounded.chunks)

    # Opt-in: unread pages (8, 10) become retrievable.
    async with db_sessionmaker() as session:
        unbounded = await service.retrieve(
            query="page",
            user_id=user_id,
            progress=ReadingProgressRepository(session, user_id),
            chunks=ChunkRepository(session, user_id),
            document_id=doc.id,
            include_unread=True,
        )
    assert max(c.page_end for c in unbounded.chunks) > 5


async def test_spoiler_safe_hard_bounds_even_with_include_unread(
    db_sessionmaker, qdrant_client, test_settings
) -> None:
    collection = test_settings.qdrant_chunks_collection
    store = ChunkVectorStore(qdrant_client, collection=collection, dim=_DIM)
    await store.ensure_collection()

    user_id, doc = await _seed_user_with_pages(
        db_sessionmaker, store, "reader@example.com", [1, 3, 7, 9]
    )
    await _record_page(db_sessionmaker, user_id, doc.id, page=3)

    service = _service(qdrant_client, test_settings)
    async with db_sessionmaker() as session:
        result = await service.retrieve(
            query="page",
            user_id=user_id,
            progress=ReadingProgressRepository(session, user_id),
            chunks=ChunkRepository(session, user_id),
            document_id=doc.id,
            include_unread=True,  # ignored under spoiler-safe
            user_spoiler_safe=True,
        )

    assert result.chunks
    assert all(c.page_end <= 3 for c in result.chunks)
