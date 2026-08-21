"""Integration tests for the re-embed maintenance job against real infra.

Seeds an indexed document with chunks + vectors at one dimension, then re-embeds
with a different-dimension embedder — asserting the collection is rebuilt at the
new dimension, points are preserved, and ``documents.embed_model`` is updated.
Embeddings are the only mocked boundary (a deterministic in-process embedder).
"""

import uuid
from collections.abc import Callable

import pytest
from api.services.ingestion_service import IngestionService
from ingestion.reembed import reembed_all, run_reembed
from ingestion.resources import IngestionResources

from shared.core.enums import DocumentFormat, DocumentStatus
from shared.models.document import Chunk
from shared.models.user import User
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    OutboxRepository,
    UserRepository,
)
from shared.vectorstore import ChunkVectorStore, build_chunk_payload, chunk_point_id

pytestmark = pytest.mark.integration

PDF_BYTES = b"%PDF-1.7\nfake\n%%EOF"


class _FakeEmbedder:
    """Deterministic embedder with a configurable dimension."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts, *, batch_size=None):
        return [[float(index + 1)] * self._dim for index, _ in enumerate(texts)]


@pytest.fixture
async def make_resources(test_settings, storage, qdrant_client, db_engine):
    """Factory for ingestion resources wired to test infra with a given embed dim."""
    created: list[IngestionResources] = []

    def _make(dim: int) -> IngestionResources:
        res = IngestionResources(test_settings)
        res.embedder = _FakeEmbedder(dim)
        created.append(res)
        return res

    yield _make
    for res in created:
        await res.aclose()


async def _seed_indexed_document(db_sessionmaker, storage, qdrant_client, collection, *, dim: int):
    """Create an indexed document with two chunks + vectors at ``dim``."""
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email="reader@example.com"))
        await session.commit()
        user_id = user.id

    ingestion = IngestionService(storage=storage, embed_model="old-model")
    async with db_sessionmaker() as session:
        doc = await ingestion.upload(
            session=session,
            documents=DocumentRepository(session, user_id),
            outbox=OutboxRepository(session),
            user_id=user_id,
            filename="book.pdf",
            content_type="application/pdf",
            document_format=DocumentFormat.PDF,
            data=PDF_BYTES,
        )

    chunks = [
        Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            user_id=user_id,
            ordinal=i,
            page_start=i + 1,
            page_end=i + 1,
            text=f"chunk {i} text",
            content_hash=f"h{i}",
        )
        for i in range(2)
    ]
    for chunk in chunks:
        chunk.vector_id = chunk_point_id(chunk.id)
    async with db_sessionmaker() as session:
        await ChunkRepository(session, user_id).add_many(chunks)
        document = await DocumentRepository(session, user_id).get(doc.id)
        document.status = DocumentStatus.INDEXED
        document.embed_model = "old-model"
        await session.commit()

    store = ChunkVectorStore(qdrant_client, collection=collection, dim=dim)
    await store.ensure_collection()
    await store.upsert(
        ids=[c.vector_id for c in chunks],
        vectors=[[0.1] * dim for _ in chunks],
        payloads=[build_chunk_payload(c, title=None, author=None, language=None) for c in chunks],
    )
    return user_id, doc.id


async def test_reembed_all_swaps_dimension_and_updates_model(
    make_resources: Callable[[int], IngestionResources],
    db_sessionmaker,
    storage,
    qdrant_client,
    test_settings,
) -> None:
    collection = test_settings.qdrant_chunks_collection
    user_id, document_id = await _seed_indexed_document(
        db_sessionmaker, storage, qdrant_client, collection, dim=8
    )

    # Switch to a 16-dim embedder and re-embed the whole corpus.
    count = await reembed_all(make_resources(16))
    assert count == 1

    # Collection rebuilt at the new dimension, with the points preserved.
    info = await qdrant_client.get_collection(collection)
    assert info.config.params.vectors.size == 16
    assert (await qdrant_client.count(collection_name=collection)).count == 2

    # embed_model now reflects the active model, not the old one.
    async with db_sessionmaker() as session:
        document = await DocumentRepository(session, user_id).get(document_id)
        assert document.embed_model == test_settings.embedding_model
        assert document.embed_model != "old-model"


async def test_run_reembed_replaces_points_same_dim(
    make_resources: Callable[[int], IngestionResources],
    db_sessionmaker,
    storage,
    qdrant_client,
    test_settings,
) -> None:
    collection = test_settings.qdrant_chunks_collection
    user_id, document_id = await _seed_indexed_document(
        db_sessionmaker, storage, qdrant_client, collection, dim=8
    )

    await run_reembed(make_resources(8), document_id=document_id, user_id=user_id)

    # Same two points (ids are the chunk ids), model updated.
    assert (await qdrant_client.count(collection_name=collection)).count == 2
    async with db_sessionmaker() as session:
        document = await DocumentRepository(session, user_id).get(document_id)
        assert document.embed_model == test_settings.embedding_model
