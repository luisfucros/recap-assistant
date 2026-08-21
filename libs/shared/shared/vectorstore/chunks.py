"""The ``document_chunks`` Qdrant collection: payloads, point ids, and access.

Two pure helpers (:func:`build_chunk_payload`, :func:`chunk_point_id`) build the
payload and point id, and :class:`ChunkVectorStore` performs the I/O
(collection bootstrap, upsert, delete). Every payload carries ``user_id`` and
every delete is filtered by it, so per-user isolation is enforced here rather
than trusted to each caller.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from shared.core.enums import Language
from shared.models.document import Chunk


@dataclass(slots=True)
class ScoredChunk:
    """A vector-search hit: the chunk's point id, similarity score, and payload.

    The payload is the one written by :func:`build_chunk_payload` (page range,
    document/bibliographic labels, content hash, language) — it deliberately does
    **not** carry the chunk text, which stays in Postgres as the source of truth
    and is hydrated separately by the retrieval service.
    """

    id: str
    score: float
    payload: dict[str, Any]


def chunk_point_id(chunk_id: uuid.UUID) -> str:
    """The Qdrant point id for a chunk — its own UUID (kept in ``Chunk.vector_id``)."""
    return str(chunk_id)


def build_chunk_payload(
    chunk: Chunk,
    *,
    title: str | None,
    author: str | None,
    language: Language | None,
) -> dict[str, Any]:
    """Build the Qdrant payload for a chunk.

    Carries everything retrieval filters or displays: the owner (``user_id``, for
    isolation), the source document + bibliographic labels, the page range and
    structure (for read-range scoping), the content hash (for de-duplication),
    and the document language (for optional cross-lingual filtering/labeling).
    Ids are stored as strings so they match cleanly in Qdrant keyword filters.
    """
    return {
        "user_id": str(chunk.user_id),
        "document_id": str(chunk.document_id),
        "ordinal": chunk.ordinal,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "title": title,
        "author": author,
        "content_hash": chunk.content_hash,
        "language": language.value if language else None,
    }


class ChunkVectorStore:
    """Read/write access to the ``document_chunks`` collection in Qdrant."""

    def __init__(
        self, client: AsyncQdrantClient, *, collection: str, dim: int | None = None
    ) -> None:
        """Bind to a Qdrant client, the collection name, and (for writes) the dim.

        ``dim`` is only needed to *create* the collection (upsert path); delete
        and search don't need it, so it may be omitted when the store is used
        purely for cleanup (e.g. document deletion).
        """
        self._client = client
        self._collection = collection
        self._dim = dim

    async def ensure_collection(self) -> None:
        """Create the collection if it does not exist (idempotent bootstrap).

        Sized to the active embedder's ``dim`` with cosine distance (the metric
        the supported embedding models are trained for).
        """
        if self._dim is None:
            raise ValueError("dim is required to create the collection")
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
        )

    async def recreate(self) -> None:
        """Drop and recreate the collection at ``dim`` (for an embedder/dim change).

        A provider switch can change the vector dimension, which a single Qdrant
        collection can't mix — so the collection is rebuilt empty and every
        document must then be re-embedded into it. Destructive by design; used
        only by the re-embed maintenance job.
        """
        if self._dim is None:
            raise ValueError("dim is required to (re)create the collection")
        if await self._client.collection_exists(self._collection):
            await self._client.delete_collection(self._collection)
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
        )

    async def upsert(
        self, *, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]
    ) -> None:
        """Upsert chunk points (id/vector/payload zipped positionally)."""
        points = [
            models.PointStruct(id=point_id, vector=list(vector), payload=payload)
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        if points:
            await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        *,
        user_id: uuid.UUID,
        query_vector: Sequence[float],
        limit: int = 8,
        document_id: uuid.UUID | None = None,
        max_page_end: int | None = None,
        chapter: str | None = None,
        section: str | None = None,
        language: str | None = None,
    ) -> list[ScoredChunk]:
        """Semantic search over the collection, **always** filtered by ``user_id``.

        The ``user_id`` condition is added here unconditionally, so no caller can
        issue an unscoped search — the per-user isolation invariant is enforced at
        the store, not trusted to each call site. Optional conditions narrow the
        search further:

        * ``document_id`` — restrict to one document.
        * ``max_page_end`` — read-range upper bound: only chunks whose
          ``page_end <= max_page_end`` (so retrieval doesn't surface unread pages).
        * ``chapter`` / ``section`` — structural filters.
        * ``language`` — restrict to one document language.

        Returns hits ordered by descending similarity; text is not included
        (it lives in Postgres and is hydrated by the caller).
        """
        must: list[models.Condition] = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))
        ]
        if document_id is not None:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=str(document_id))
                )
            )
        if max_page_end is not None:
            # page_end <= max_page_end. Chunks with a null page_end (page-less
            # formats) don't match a range filter and are excluded, which is the
            # intended read-range behavior.
            must.append(models.FieldCondition(key="page_end", range=models.Range(lte=max_page_end)))
        if chapter is not None:
            must.append(
                models.FieldCondition(key="chapter", match=models.MatchValue(value=chapter))
            )
        if section is not None:
            must.append(
                models.FieldCondition(key="section", match=models.MatchValue(value=section))
            )
        if language is not None:
            must.append(
                models.FieldCondition(key="language", match=models.MatchValue(value=language))
            )

        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(query_vector),
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [
            ScoredChunk(id=str(point.id), score=point.score, payload=point.payload or {})
            for point in response.points
        ]

    async def delete_by_document(self, *, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
        """Delete all of a user's points for a document (idempotent re-ingest).

        Filtered by both ``user_id`` and ``document_id`` so a re-run never
        touches another user's vectors even if ids were to collide.
        """
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=str(user_id))
                        ),
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=str(document_id))
                        ),
                    ]
                )
            ),
        )
