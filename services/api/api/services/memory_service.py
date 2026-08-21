"""The agent's long-term-memory vector source: write, retrieve, view, delete.

:class:`MemoryService` is the single place that joins the two long-term-memory
stores — Postgres (:class:`~shared.models.memory.LongTermMemory`, the source of
truth for content) and the ``long_term_memory`` Qdrant collection
(:class:`~shared.vectorstore.MemoryVectorStore`, embeddings + filter metadata
only, never content). Two invariants shape it, mirroring
:class:`~api.services.retrieval_service.RetrievalService`:

* **Per-user isolation** — the owning ``user_id`` comes only from the
  ``LongTermMemoryRepository`` passed in (itself bound to the authenticated
  user at construction) and is injected server-side into every vector-store
  call; it is never accepted as a method parameter here, so a caller/tool
  cannot smuggle another user's id into a search or delete.
* **Reading position drives memory** — a summary memory is always keyed to
  ``(document_id, page_start, page_end)``, and spoiler-safe recall bounds
  retrieval to ``page_end <= current_page`` (FR-18.3) via the vector store's
  ``max_page_end`` filter, not a post-hoc client-side check.
"""

import uuid
from collections.abc import Sequence

from loguru import logger

from shared.core.enums import MemoryType
from shared.core.errors import InvalidInputError
from shared.models.memory import LongTermMemory
from shared.providers.base import Embedder
from shared.repositories import LongTermMemoryRepository
from shared.vectorstore import (
    MemoryVectorStore,
    ScoredMemory,
    build_memory_payload,
    memory_point_id,
)


class MemoryService:
    """Write salient/summary memories; semantic, typed, page-range retrieval; view/delete."""

    # Cosine-similarity floor (Qdrant's COSINE distance score) above which a
    # freshly-classified personal memory is treated as a restatement of one
    # already saved, not a new fact — see write_memory. Chosen well above the
    # scores a merely related-but-distinct fact tends to score (e.g. "likes
    # fantasy" vs. "likes sci-fi" lands well under this), so it only catches
    # genuine near-duplicates/paraphrases, not merely similar preferences.
    _DEDUP_SIMILARITY_THRESHOLD = 0.93

    def __init__(self, *, embedder: Embedder, vector_store: MemoryVectorStore) -> None:
        """Wire the service to the active embedder and the memory vector store."""
        self._embedder = embedder
        self._vectors = vector_store

    async def write_memory(
        self,
        *,
        memories: LongTermMemoryRepository,
        session,  # noqa: ANN001 — AsyncSession; kept import-light at this layer
        type: MemoryType,
        content: str,
    ) -> LongTermMemory:
        """Save a salient, user-level memory (preference, fact, habit, or FAQ).

        Idempotent by *meaning*, not just by exact text: the agent's salience
        classifier (FR-7.9) re-runs on every turn with no memory of what's
        already stored, so a reader restating something they've already said
        ("I love fantasy novels" today, "I really love fantasy" next week)
        would otherwise pile up a near-duplicate row each time. Before
        inserting, this embeds the new content and searches the caller's own
        memories of the same ``type`` for a near-duplicate
        (:attr:`_DEDUP_SIMILARITY_THRESHOLD`); a hit is refreshed in place
        (its content updated to the latest phrasing, its vector re-indexed)
        rather than duplicated. A genuinely new fact — even one merely related
        to an existing memory — still inserts a new row.

        Raises:
            InvalidInputError: ``type`` is ``SUMMARY`` — summaries are keyed to a
                document and page range, so they're saved via :meth:`write_summary`.
        """
        if type is MemoryType.SUMMARY:
            raise InvalidInputError("Summary memories must be saved via write_summary().")
        await self._vectors.ensure_collection()
        vector = (await self._embedder.embed([content]))[0]
        existing = await self._find_duplicate(memories=memories, type=type, vector=vector)
        if existing is not None:
            existing.content = content
            await self._upsert(existing, vector)
            await session.commit()
            logger.info("memory.write: merged restatement into existing {} memory", type.value)
            return existing
        memory = await memories.add(
            LongTermMemory(user_id=memories.user_id, type=type, content=content)
        )
        await self._upsert(memory, vector)
        await session.commit()
        logger.info("memory.write: saved new {} memory", type.value)
        return memory

    async def _find_duplicate(
        self,
        *,
        memories: LongTermMemoryRepository,
        type: MemoryType,
        vector: Sequence[float],
    ) -> LongTermMemory | None:
        """Return the caller's existing same-type memory nearest ``vector``, if close enough.

        Restricted to ``type`` so a fact never collapses into an unrelated
        habit/preference that merely embeds similarly. ``None`` when the
        nearest hit (if any) scores below the dedup threshold, or when the
        hit's row is missing (e.g. deleted between index and search) — either
        way, the caller proceeds to insert a new memory.
        """
        hits = await self._vectors.search(
            user_id=memories.user_id, query_vector=vector, type=type, limit=1
        )
        if not hits or hits[0].score < self._DEDUP_SIMILARITY_THRESHOLD:
            return None
        rows = await memories.list_by_ids([uuid.UUID(hits[0].id)])
        return rows[0] if rows else None

    async def write_summary(
        self,
        *,
        memories: LongTermMemoryRepository,
        session,  # noqa: ANN001 — AsyncSession; kept import-light at this layer
        document_id: uuid.UUID,
        page_start: int,
        page_end: int,
        content: str,
    ) -> LongTermMemory:
        """Save a page-range summary memory, keyed to ``(document_id, page_range)``.

        The caller (the M5 page-range-confirmation interrupt) owns proposing and
        confirming the range; this method only validates it's well-formed.

        Raises:
            InvalidInputError: the range is empty or inverted (``page_start`` must
                be at least 1 and no greater than ``page_end``).
        """
        if page_start < 1 or page_end < page_start:
            raise InvalidInputError("page_start must be >= 1 and <= page_end.")
        memory = await memories.add(
            LongTermMemory(
                user_id=memories.user_id,
                document_id=document_id,
                type=MemoryType.SUMMARY,
                content=content,
                page_start=page_start,
                page_end=page_end,
            )
        )
        await self._embed_and_index(memory)
        await session.commit()
        logger.info(
            "memory.write_summary: saved summary for document {} pages {}-{}",
            document_id,
            page_start,
            page_end,
        )
        return memory

    async def retrieve(
        self,
        *,
        memories: LongTermMemoryRepository,
        query: str,
        type: MemoryType | None = None,
        document_id: uuid.UUID | None = None,
        max_page_end: int | None = None,
        limit: int = 8,
    ) -> list[LongTermMemory]:
        """Semantic search over the caller's memories, ranked by similarity.

        ``memories`` supplies the owning ``user_id`` (bound at construction); it
        is passed to the vector store explicitly on every call, so recall can
        never cross a user boundary. ``type``/``document_id`` narrow to one kind
        or one document; ``max_page_end`` is the spoiler-safe bound — when the
        reader's spoiler-safe setting is on, pass their ``current_page`` here so
        a summary reaching past it never surfaces.
        """
        query_vector = (await self._embedder.embed([query]))[0]
        hits = await self._vectors.search(
            user_id=memories.user_id,
            query_vector=query_vector,
            limit=limit,
            type=type,
            document_id=document_id,
            max_page_end=max_page_end,
        )
        results = await self._hydrate(hits, memories)
        logger.debug(
            "memory.retrieve: {} hits (type={}, document_id={}, max_page_end={})",
            len(results),
            type,
            document_id,
            max_page_end,
        )
        return results

    async def list_memories(
        self,
        *,
        memories: LongTermMemoryRepository,
        type: MemoryType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[LongTermMemory]:
        """List the caller's stored memories, newest first — backs the memory panel."""
        if type is not None:
            return await memories.list_by_type(type, limit=limit, offset=offset)
        return await memories.list_recent(limit=limit, offset=offset)

    async def delete_memory(
        self,
        *,
        memories: LongTermMemoryRepository,
        session,  # noqa: ANN001 — AsyncSession; kept import-light at this layer
        memory_id: uuid.UUID,
    ) -> None:
        """Delete a memory and its vector point (privacy: FR-4.5 view/delete).

        The vector point is deleted first (idempotent — a retry after a crash
        between the two deletes finds nothing to remove); the Postgres row,
        the authority, is deleted last.

        Raises:
            NotFoundError: the memory doesn't exist or isn't the caller's.
        """
        memory = await memories.get_or_404(memory_id)
        await self._vectors.delete(user_id=memories.user_id, memory_id=memory.id)
        await memories.delete(memory)
        await session.commit()
        logger.info("memory.delete: removed {} memory {}", memory.type.value, memory_id)

    async def _embed_and_index(self, memory: LongTermMemory) -> None:
        """Embed a freshly-added memory's content and upsert its vector point.

        Used by :meth:`write_summary`, which has no dedup step of its own —
        summaries are already keyed to a unique ``(document_id, page_range)``
        confirmed via the HITL page-range-confirm flow, so a repeat save isn't
        the same silent-pileup risk :meth:`write_memory` guards against.
        """
        await self._vectors.ensure_collection()
        vector = (await self._embedder.embed([memory.content]))[0]
        await self._upsert(memory, vector)

    async def _upsert(self, memory: LongTermMemory, vector: Sequence[float]) -> None:
        """Upsert one memory's vector point and set ``embedding_id`` on the row.

        Sets ``embedding_id`` on the (already session-tracked) row so it's
        persisted alongside the row on the caller's subsequent commit.
        """
        point_id = memory_point_id(memory.id)
        await self._vectors.upsert(
            ids=[point_id], vectors=[vector], payloads=[build_memory_payload(memory)]
        )
        memory.embedding_id = point_id

    async def _hydrate(
        self, hits: Sequence[ScoredMemory], memories: LongTermMemoryRepository
    ) -> list[LongTermMemory]:
        """Attach Postgres-held content to each hit, preserving search (score) order.

        Hits whose row is missing (e.g. deleted between index and query) are
        dropped rather than returned content-less.
        """
        by_id = {
            row.id: row for row in await memories.list_by_ids([uuid.UUID(hit.id) for hit in hits])
        }
        results: list[LongTermMemory] = []
        for hit in hits:
            row = by_id.get(uuid.UUID(hit.id))
            if row is not None:
                results.append(row)
        return results
