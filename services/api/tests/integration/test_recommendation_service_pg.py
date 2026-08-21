"""Integration tests for RecommendationService against real Postgres + Qdrant.

Seeds a real reading-history document (completed, via a real ``ReadingProgress``
row) and a real candidate document with a real chunk vector in Qdrant, then
drives :meth:`RecommendationService.recommend_from_library` to verify what a
unit test (everything faked) can only approximate: the seed-to-candidate
similarity search actually round-trips through Postgres (documents, progress)
and Qdrant (``document_chunks``), and — the load-bearing isolation invariant —
one user's recommendations never surface another user's library, even when
both hold a document with the same title and an identical embedding.
"""

import uuid
from types import SimpleNamespace

import pytest
from api.services.memory_service import MemoryService
from api.services.progress_service import ProgressService
from api.services.recommendation_service import RecommendationService

from shared.core.enums import DocumentFormat, ReadingStatus
from shared.models.document import Chunk, Document
from shared.models.reading import ReadingProgress
from shared.models.user import User
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    LongTermMemoryRepository,
    ReadingProgressRepository,
    UserRepository,
)
from shared.vectorstore import ChunkVectorStore, build_chunk_payload, chunk_point_id

pytestmark = pytest.mark.integration

_DIM = 8
_V_MATCH = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_V_OTHER = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _KeyedEmbedder:
    """Returns a chosen vector per known text, so similarity is fully controlled."""

    def __init__(self, mapping: dict[str, list[float]], default: list[float]) -> None:
        self._mapping = mapping
        self._default = default

    async def embed(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
        return [self._mapping.get(t, self._default) for t in texts]


async def _make_user(db_sessionmaker, email: str) -> User:
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email=email))
        await session.commit()
        return user


async def _seed_document_with_chunk(
    db_sessionmaker, store: ChunkVectorStore, user_id: uuid.UUID, *, title: str, vector: list[float]
) -> Document:
    """Create a document with one chunk, its vector upserted into Qdrant."""
    doc = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/sha256/{uuid.uuid4().hex}.pdf",
        content_sha256=uuid.uuid4().hex,
        format=DocumentFormat.PDF,
        embed_model="test-model",
        title=title,
    )
    async with db_sessionmaker() as session:
        await DocumentRepository(session, user_id).add(doc)
        await session.commit()

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        user_id=user_id,
        ordinal=0,
        page_start=1,
        page_end=1,
        text=f"a page of {title}",
        content_hash=uuid.uuid4().hex,
        vector_id=None,
    )
    chunk.vector_id = chunk_point_id(chunk.id)
    async with db_sessionmaker() as session:
        await ChunkRepository(session, user_id).add_many([chunk])
        await session.commit()

    await store.upsert(
        ids=[chunk.vector_id],
        vectors=[vector],
        payloads=[build_chunk_payload(chunk, title=title, author=None, language=None)],
    )
    return doc


async def _mark_completed(db_sessionmaker, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
    async with db_sessionmaker() as session:
        session.add(
            ReadingProgress(
                user_id=user_id,
                document_id=document_id,
                current_page=1,
                last_summarized_page=0,
                status=ReadingStatus.COMPLETED,
            )
        )
        await session.commit()


def _service(qdrant_client, test_settings, embedder) -> RecommendationService:
    return RecommendationService(
        embedder=embedder,
        vector_store=ChunkVectorStore(
            qdrant_client, collection=test_settings.qdrant_chunks_collection
        ),
    )


async def test_recommends_a_similar_library_document_against_real_infra(
    db_sessionmaker, qdrant_client, test_settings
) -> None:
    collection = test_settings.qdrant_chunks_collection
    store = ChunkVectorStore(qdrant_client, collection=collection, dim=_DIM)
    await store.ensure_collection()

    user = await _make_user(db_sessionmaker, "reader@example.com")
    seed = await _seed_document_with_chunk(
        db_sessionmaker, store, user.id, title="The Odyssey", vector=_V_OTHER
    )
    candidate = await _seed_document_with_chunk(
        db_sessionmaker, store, user.id, title="The Iliad", vector=_V_MATCH
    )
    await _mark_completed(db_sessionmaker, user.id, seed.id)

    embedder = _KeyedEmbedder({"The Odyssey": _V_MATCH}, default=_V_OTHER)
    service = _service(qdrant_client, test_settings, embedder)
    memory_service = MemoryService(embedder=embedder, vector_store=SimpleNamespace())

    async with db_sessionmaker() as session:
        recs = await service.recommend_from_library(
            user_id=user.id,
            documents=DocumentRepository(session, user.id),
            progress_repo=ReadingProgressRepository(session, user.id),
            progress_service=ProgressService(),
            memories=LongTermMemoryRepository(session, user.id),
            memory_service=memory_service,
        )

    assert len(recs) == 1
    assert recs[0].document_id == candidate.id
    assert recs[0].title == "The Iliad"
    assert recs[0].reason == "Because you completed The Odyssey"


async def test_recommendations_are_isolated_to_the_users_own_library(
    db_sessionmaker, qdrant_client, test_settings
) -> None:
    collection = test_settings.qdrant_chunks_collection
    store = ChunkVectorStore(qdrant_client, collection=collection, dim=_DIM)
    await store.ensure_collection()

    alice = await _make_user(db_sessionmaker, "alice@example.com")
    bob = await _make_user(db_sessionmaker, "bob@example.com")

    alice_seed = await _seed_document_with_chunk(
        db_sessionmaker, store, alice.id, title="The Odyssey", vector=_V_OTHER
    )
    alice_candidate = await _seed_document_with_chunk(
        db_sessionmaker, store, alice.id, title="The Iliad", vector=_V_MATCH
    )
    await _mark_completed(db_sessionmaker, alice.id, alice_seed.id)

    # Bob holds a document with the exact same title and embedding as Alice's
    # candidate — only the server-side user_id filter can keep it out of her
    # recommendations, since nothing about the content differs.
    await _seed_document_with_chunk(
        db_sessionmaker, store, bob.id, title="The Iliad", vector=_V_MATCH
    )

    embedder = _KeyedEmbedder({"The Odyssey": _V_MATCH}, default=_V_OTHER)
    service = _service(qdrant_client, test_settings, embedder)
    memory_service = MemoryService(embedder=embedder, vector_store=SimpleNamespace())

    async with db_sessionmaker() as session:
        recs = await service.recommend_from_library(
            user_id=alice.id,
            documents=DocumentRepository(session, alice.id),
            progress_repo=ReadingProgressRepository(session, alice.id),
            progress_service=ProgressService(),
            memories=LongTermMemoryRepository(session, alice.id),
            memory_service=memory_service,
        )

    assert len(recs) == 1
    assert recs[0].document_id == alice_candidate.id
