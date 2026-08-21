"""Position-aware, user-isolated semantic retrieval over document chunks.

:class:`RetrievalService` is the single entry point for "find passages relevant
to this query" (the agent's ``retrieve_chunks`` tool, Source 3). It embeds the
query, searches the ``document_chunks`` collection, hydrates the matched text
from Postgres (the source of truth — vectors carry only metadata), collapses
near-duplicates, and returns chunks with citations. Two invariants shape it:

* **Per-user isolation** — the ``user_id`` filter is injected server-side from the
  authenticated context (a method argument the caller controls only via the
  request, never via LLM/tool arguments) and passed to the vector store, which
  adds it unconditionally. There is no code path that searches without it.
* **Reading position drives retrieval** — for a targeted document, results
  default to the read range (``page_end <= current_page``) so answers don't leak
  unread content; ``include_unread`` opts out. (Spoiler-safe mode later turns this
  default into a hard, non-optional bound.)
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from shared.core.config import Settings
from shared.core.spoiler import resolve_spoiler_safe
from shared.models.reading import ReadingProgress
from shared.observability.metrics import time_operation
from shared.providers import Embedder
from shared.repositories import ChunkRepository, ReadingProgressRepository
from shared.vectorstore import ChunkVectorStore, ScoredChunk


@dataclass(slots=True)
class Citation:
    """A source pointer for a retrieved passage (document + page span + labels)."""

    document_id: uuid.UUID
    title: str | None
    author: str | None
    page_start: int | None
    page_end: int | None


@dataclass(slots=True)
class RetrievedChunk:
    """A passage returned by retrieval: its text, location, score, and citation."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    score: float
    page_start: int | None
    page_end: int | None
    chapter: str | None
    section: str | None
    citation: Citation


@dataclass(slots=True)
class RetrievalResult:
    """The outcome of a retrieval: ranked chunks and their de-duplicated citations."""

    chunks: list[RetrievedChunk]
    citations: list[Citation]


class RetrievalService:
    """Embed a query, search chunks (user-scoped, read-range aware), return passages."""

    def __init__(
        self, *, embedder: Embedder, vector_store: ChunkVectorStore, settings: Settings
    ) -> None:
        """Wire the service to the active embedder, chunk vector store, and settings."""
        self._embedder = embedder
        self._vectors = vector_store
        self._settings = settings

    async def retrieve(
        self,
        *,
        query: str,
        user_id: uuid.UUID,
        progress: ReadingProgressRepository,
        chunks: ChunkRepository,
        document_id: uuid.UUID | None = None,
        include_unread: bool = False,
        chapter: str | None = None,
        section: str | None = None,
        limit: int | None = None,
        user_spoiler_safe: bool = False,
        spoiler_safe_override: bool | None = None,
    ) -> RetrievalResult:
        """Return passages relevant to ``query`` for one user.

        ``user_id`` comes from the authenticated context and is the only owner the
        search will ever see. When ``document_id`` is given and ``include_unread``
        is false, results are bounded to the document's read range
        (``page_end <= current_page``). **Spoiler-safe** (FR-18) makes that bound
        *hard*: when it resolves on, ``include_unread`` is ignored and unread pages
        are never returned for a targeted document. Near-duplicate passages (same
        ``content_hash``) are collapsed, keeping the highest-scoring occurrence.

        Args:
            query: The natural-language search text.
            user_id: The owning user (server-supplied; never from tool arguments).
            progress: Reading-progress repository (to resolve the read range).
            chunks: Chunk repository (to hydrate text from Postgres).
            document_id: Restrict to one document (enables read-range bounding).
            include_unread: Skip the read-range bound and search unread pages too
                (ignored when spoiler-safe resolves on).
            chapter: Restrict to a chapter label.
            section: Restrict to a section label.
            limit: Max hits to request (defaults to ``retrieval_top_k``).
            user_spoiler_safe: The user's global spoiler-safe default.
            spoiler_safe_override: A per-query spoiler-safe override (wins over the
                per-document and user settings).
        """
        top_k = limit or self._settings.retrieval_top_k
        row = await progress.get_by_document(document_id) if document_id is not None else None
        spoiler_on = resolve_spoiler_safe(
            per_query=spoiler_safe_override,
            per_document=row.spoiler_safe if row is not None else None,
            user_default=user_spoiler_safe,
        )
        max_page_end = self._read_range_bound(
            row, document_id=document_id, include_unread=include_unread, spoiler_on=spoiler_on
        )

        with time_operation("embedding"):
            query_vector = (await self._embedder.embed([query]))[0]

        with time_operation("retrieval"):
            hits = await self._vectors.search(
                user_id=user_id,
                query_vector=query_vector,
                limit=top_k,
                document_id=document_id,
                max_page_end=max_page_end,
                chapter=chapter,
                section=section,
            )

        deduped = self._collapse_duplicates(hits)
        retrieved = await self._hydrate(deduped, chunks)
        logger.debug(
            "retrieval: {} hits -> {} after dedup (document_id={}, max_page_end={})",
            len(hits),
            len(deduped),
            document_id,
            max_page_end,
        )
        return RetrievalResult(chunks=retrieved, citations=[chunk.citation for chunk in retrieved])

    @staticmethod
    def _read_range_bound(
        row: ReadingProgress | None,
        *,
        document_id: uuid.UUID | None,
        include_unread: bool,
        spoiler_on: bool,
    ) -> int | None:
        """Resolve the read-range upper bound (``current_page``) or ``None``.

        A library-wide search (no ``document_id``) is never page-bounded here —
        per-document positions can't be expressed in one query, so the output-side
        spoiler check (M4) is the backstop and the agent scopes to a document when
        spoiler-safety must hold. For a targeted document: ``include_unread`` lifts
        the bound *unless* spoiler-safe is on, which forces it. With no progress row
        yet the bound is ``0`` — nothing read, so nothing surfaces until a position
        is recorded (the seam the ask-which-pages HITL fills in M5).
        """
        if document_id is None:
            return None
        current_page = row.current_page if row is not None else 0
        if include_unread and not spoiler_on:
            return None
        return current_page

    async def _hydrate(
        self, hits: Sequence[ScoredChunk], chunks: ChunkRepository
    ) -> list[RetrievedChunk]:
        """Attach Postgres-held text to each hit, preserving search (score) order.

        Hits whose chunk row is missing (e.g. deleted between index and query) are
        dropped rather than returned text-less.
        """
        by_id = {
            row.id: row for row in await chunks.list_by_ids([uuid.UUID(hit.id) for hit in hits])
        }
        results: list[RetrievedChunk] = []
        for hit in hits:
            row = by_id.get(uuid.UUID(hit.id))
            if row is None:
                continue
            results.append(self._to_retrieved(hit, row.text))
        return results

    @staticmethod
    def _to_retrieved(hit: ScoredChunk, text: str) -> RetrievedChunk:
        """Build a :class:`RetrievedChunk` from a scored hit and its hydrated text."""
        payload = hit.payload
        document_id = uuid.UUID(payload["document_id"])
        citation = Citation(
            document_id=document_id,
            title=payload.get("title"),
            author=payload.get("author"),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
        )
        return RetrievedChunk(
            chunk_id=uuid.UUID(hit.id),
            document_id=document_id,
            text=text,
            score=hit.score,
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
            chapter=payload.get("chapter"),
            section=payload.get("section"),
            citation=citation,
        )

    @staticmethod
    def _collapse_duplicates(hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Collapse near-duplicate hits by ``content_hash``, keeping the top-scored.

        Overlapping chunks or re-uploaded content can surface the same passage
        multiple times; since hits arrive in descending-score order, keeping the
        first occurrence of each ``content_hash`` preserves the best-ranked copy
        and avoids hydrating duplicate text (FR-1.12). Hits without a
        ``content_hash`` payload are never collapsed (treated as distinct).
        """
        seen: set[str] = set()
        collapsed: list[ScoredChunk] = []
        for hit in hits:
            content_hash = hit.payload.get("content_hash")
            if content_hash is not None and content_hash in seen:
                continue
            if content_hash is not None:
                seen.add(content_hash)
            collapsed.append(hit)
        return collapsed
