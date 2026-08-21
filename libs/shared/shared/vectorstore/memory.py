"""The ``long_term_memory`` Qdrant collection: payloads, point ids, and access.

Mirrors :mod:`shared.vectorstore.chunks` — the same shape, one payload/store per
collection so per-user isolation lives in one place per collection. This one
never carries the memory's text in the payload either: content stays in
Postgres (:class:`~shared.models.memory.LongTermMemory`) as the source of
truth, hydrated by :class:`~api.services.memory_service.MemoryService`.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from shared.core.enums import MemoryType
from shared.models.memory import LongTermMemory


@dataclass(slots=True)
class ScoredMemory:
    """A vector-search hit: the memory's point id, similarity score, and payload."""

    id: str
    score: float
    payload: dict[str, Any]


def memory_point_id(memory_id: uuid.UUID) -> str:
    """The Qdrant point id for a memory — its own UUID (kept in ``embedding_id``)."""
    return str(memory_id)


def build_memory_payload(memory: LongTermMemory) -> dict[str, Any]:
    """Build the Qdrant payload for a memory: owner, type, and page-range keying.

    Carries only what recall filters on — the owner (``user_id``, for isolation),
    the memory ``type``, and the document/page-range a summary memory is tied to
    (null for user-level facts). Ids are stored as strings so they match cleanly
    in Qdrant keyword filters.
    """
    return {
        "user_id": str(memory.user_id),
        "type": memory.type.value,
        "document_id": str(memory.document_id) if memory.document_id else None,
        "page_start": memory.page_start,
        "page_end": memory.page_end,
    }


class MemoryVectorStore:
    """Read/write access to the ``long_term_memory`` collection in Qdrant."""

    def __init__(
        self, client: AsyncQdrantClient, *, collection: str, dim: int | None = None
    ) -> None:
        """Bind to a Qdrant client, the collection name, and (for writes) the dim.

        ``dim`` is only needed to *create* the collection (upsert path); search
        and delete don't need it.
        """
        self._client = client
        self._collection = collection
        self._dim = dim

    async def ensure_collection(self) -> None:
        """Create the collection if it does not exist (idempotent bootstrap).

        Sized to the active embedder's ``dim`` with cosine distance, same as
        ``document_chunks`` — memory embeddings reuse the same embedder.
        """
        if self._dim is None:
            raise ValueError("dim is required to create the collection")
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
        )

    async def upsert(
        self, *, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]
    ) -> None:
        """Upsert memory points (id/vector/payload zipped positionally)."""
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
        type: MemoryType | None = None,
        document_id: uuid.UUID | None = None,
        max_page_end: int | None = None,
    ) -> list[ScoredMemory]:
        """Semantic search over the collection, **always** filtered by ``user_id``.

        The ``user_id`` condition is added here unconditionally, so no caller can
        issue an unscoped search — the per-user isolation invariant is enforced at
        the store, not trusted to each call site. Optional conditions narrow it:

        * ``type`` — restrict to one memory type (e.g. only ``summary``).
        * ``document_id`` — restrict to one document's memories.
        * ``max_page_end`` — spoiler-safe bound: only summaries whose
          ``page_end <= max_page_end`` (never surface an unread recap).

        Returns hits ordered by descending similarity; content is not included
        (it lives in Postgres and is hydrated by the caller).
        """
        must: list[models.Condition] = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))
        ]
        if type is not None:
            must.append(
                models.FieldCondition(key="type", match=models.MatchValue(value=type.value))
            )
        if document_id is not None:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=str(document_id))
                )
            )
        if max_page_end is not None:
            # page_end <= max_page_end. A memory with a null page_end (a
            # user-level fact, not a summary) doesn't match a range filter and
            # is excluded — this bound only narrows summary-type recall.
            must.append(models.FieldCondition(key="page_end", range=models.Range(lte=max_page_end)))

        response = await self._client.query_points(
            collection_name=self._collection,
            query=list(query_vector),
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [
            ScoredMemory(id=str(point.id), score=point.score, payload=point.payload or {})
            for point in response.points
        ]

    async def delete(self, *, user_id: uuid.UUID, memory_id: uuid.UUID) -> None:
        """Delete one of a user's memory points (privacy: user-initiated delete).

        Filtered by both ``user_id`` and the point id so a caller can never
        delete another user's vector even if ids were to collide.
        """
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=str(user_id))
                        ),
                        models.HasIdCondition(has_id=[memory_point_id(memory_id)]),
                    ]
                )
            ),
        )
