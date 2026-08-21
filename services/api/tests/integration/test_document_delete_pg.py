"""Integration test for document deletion against real Postgres+Qdrant+MinIO.

Seeds a document with its stored original, chunk rows, and Qdrant vectors, then
deletes it through ``DocumentService`` and verifies all three stores are cleaned:
the object is gone from MinIO, the vectors from Qdrant, and the row (with its
chunks, via the DB cascade) from Postgres.
"""

import uuid

import pytest
from api.services.document_service import DocumentService
from api.services.ingestion_service import IngestionService
from botocore.exceptions import ClientError

from shared.core.enums import DocumentFormat
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

PDF_BYTES = b"%PDF-1.7\nfake pdf body\n%%EOF"
_DIM = 8


async def _seed_indexed_document(db_sessionmaker, storage, qdrant_client, collection):
    """Create a user + stored document + two chunks + their Qdrant vectors."""
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email="reader@example.com"))
        await session.commit()
        user_id = user.id

    # Store the original + a pending row via the real upload path.
    ingestion = IngestionService(storage=storage, embed_model="test-model")
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

    # Persist two chunk rows.
    chunks = [
        Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            user_id=user_id,
            ordinal=i,
            page_start=i + 1,
            page_end=i + 1,
            text=f"chunk {i}",
            content_hash=f"h{i}",
        )
        for i in range(2)
    ]
    for chunk in chunks:
        chunk.vector_id = chunk_point_id(chunk.id)
    async with db_sessionmaker() as session:
        await ChunkRepository(session, user_id).add_many(chunks)
        await session.commit()

    # Upsert their vectors.
    store = ChunkVectorStore(qdrant_client, collection=collection, dim=_DIM)
    await store.ensure_collection()
    await store.upsert(
        ids=[c.vector_id for c in chunks],
        vectors=[[0.1] * _DIM for _ in chunks],
        payloads=[build_chunk_payload(c, title=None, author=None, language=None) for c in chunks],
    )
    return user_id, doc


async def test_delete_removes_row_chunks_vectors_and_object(
    db_sessionmaker, storage, qdrant_client, test_settings
) -> None:
    collection = test_settings.qdrant_chunks_collection
    user_id, doc = await _seed_indexed_document(db_sessionmaker, storage, qdrant_client, collection)

    # Preconditions: everything is present.
    assert await storage.get(doc.object_key) == PDF_BYTES
    assert (await qdrant_client.count(collection_name=collection)).count == 2

    service = DocumentService(
        storage=storage,
        vector_store=ChunkVectorStore(qdrant_client, collection=collection),
    )
    async with db_sessionmaker() as session:
        await service.delete(
            session=session,
            documents=DocumentRepository(session, user_id),
            user_id=user_id,
            document_id=doc.id,
        )

    # Row + chunks gone from Postgres.
    async with db_sessionmaker() as session:
        assert await DocumentRepository(session, user_id).get(doc.id) is None
        assert await ChunkRepository(session, user_id).list_by_document(doc.id) == []
    # Vectors gone from Qdrant.
    assert (await qdrant_client.count(collection_name=collection)).count == 0
    # Object gone from MinIO.
    with pytest.raises(ClientError):
        await storage.get(doc.object_key)
