"""Explainable reading recommendations combining internal + external signals (FR-5).

Two paths, deliberately kept separate by how far they reach:

* **Internal** (:meth:`RecommendationService.recommend_from_library`) — combines
  three signals that touch only the reader's own stored data: reading history
  (completed/in-progress documents), semantic similarity across the reader's
  own library (searching ``document_chunks`` with a query built from those
  documents), and stated long-term-memory preferences/habits (used as extra
  search seeds). Never reaches outside the reader's data, so the agent's
  ``recommend`` tool runs this ungated and the ``/recommendations`` route calls
  it directly.
* **External** (:meth:`RecommendationService.recommend_from_web`) — a
  :class:`~shared.providers.base.WebSearchProvider` query for suggestions
  beyond the reader's own library. Reaches a third party, so the caller (the
  ``recommend`` tool) must gate this behind the reader's approval before
  calling it.

Both return the same :class:`Recommendation` shape (``document_id=None`` marks
an external/web suggestion) so callers don't need to know which path produced
a given item.
"""

import uuid
from dataclasses import dataclass

from loguru import logger

from api.services.memory_service import MemoryService
from api.services.progress_service import ProgressService
from shared.core.enums import MemoryType, ReadingStatus
from shared.providers.base import Embedder, SearchResult, WebSearchProvider
from shared.repositories import (
    DocumentRepository,
    LongTermMemoryRepository,
    ReadingProgressRepository,
)
from shared.vectorstore import ChunkVectorStore

# Bounds on how many embed/search calls one recommend() run makes — a personal
# library is small, so a handful of seeds is plenty of signal without the
# latency of embedding every completed/preferred item the reader has.
_HISTORY_SEED_LIMIT = 3
_MEMORY_SEED_LIMIT = 2
_CANDIDATES_PER_SEED = 20


@dataclass(slots=True)
class Recommendation:
    """One explainable suggestion — from the reader's own library, or the web."""

    title: str
    reason: str
    document_id: uuid.UUID | None = None
    author: str | None = None
    url: str | None = None
    score: float | None = None


@dataclass(slots=True)
class _HistoryItem:
    """One reading-history row resolved to a display title (a similarity seed)."""

    document_id: uuid.UUID
    title: str
    status: ReadingStatus


@dataclass(slots=True)
class _Seed:
    """A search seed feeding the internal similarity signal, with its own explanation."""

    query: str
    reason: str


class RecommendationService:
    """Combine reading history + library similarity + memory (+ optionally web)."""

    def __init__(self, *, embedder: Embedder, vector_store: ChunkVectorStore) -> None:
        """Wire the service to the active embedder and the (read-only) chunk vector store."""
        self._embedder = embedder
        self._vectors = vector_store

    async def recommend_from_library(
        self,
        *,
        user_id: uuid.UUID,
        documents: DocumentRepository,
        progress_repo: ReadingProgressRepository,
        progress_service: ProgressService,
        memories: LongTermMemoryRepository,
        memory_service: MemoryService,
        limit: int = 5,
    ) -> list[Recommendation]:
        """Recommend other library documents similar to the reader's history/preferences.

        Never reaches outside the reader's own stored data (no external call),
        so this path is never HITL-gated. Returns an empty list when there's no
        reading history and no stated preference/habit to build a search seed
        from — a genuine "nothing to recommend from yet", not a guess.
        """
        history = await self._history_seeds(
            documents=documents, progress_service=progress_service, progress_repo=progress_repo
        )
        seeds = await self._seeds(history=history, memories=memories, memory_service=memory_service)
        if not seeds:
            logger.debug("recommend.library: no seeds available (no history or preferences)")
            return []
        exclude_ids = {item.document_id for item in history}
        ranked = await self._rank_candidates(user_id=user_id, seeds=seeds, exclude_ids=exclude_ids)

        recommendations: list[Recommendation] = []
        for document_id, (score, reason) in ranked[:limit]:
            document = await documents.get(document_id)
            if document is None:
                continue
            recommendations.append(
                Recommendation(
                    document_id=document_id,
                    title=document.title or "Untitled document",
                    author=document.author,
                    reason=reason,
                    score=score,
                )
            )
        logger.info(
            "recommend.library: {} seeds ({} history, ranked to {} recommendations)",
            len(seeds),
            len(history),
            len(recommendations),
        )
        return recommendations

    async def recommend_from_web(
        self, *, web_search: WebSearchProvider, query: str, limit: int = 5
    ) -> list[Recommendation]:
        """Recommend further reading via a web search — external; the caller must gate this."""
        logger.info("recommend.web: search started (query_chars={}, limit={})", len(query), limit)
        hits: list[SearchResult] = await web_search.search(query, count=limit)
        logger.info("recommend.web: {} results", len(hits))
        return [
            Recommendation(
                title=hit.title or hit.url,
                reason=f'From a web search for "{query}"',
                url=hit.url,
                score=hit.score,
            )
            for hit in hits
        ]

    async def default_web_query(
        self,
        *,
        documents: DocumentRepository,
        progress_service: ProgressService,
        progress_repo: ReadingProgressRepository,
    ) -> str | None:
        """A web-search query built from the reader's most recent history, or ``None``.

        Backs the ``recommend`` tool's external branch when the model didn't
        supply an explicit query — ``None`` means there's no history to build
        one from, and the caller should ask for a topic instead of guessing.
        """
        history = await self._history_seeds(
            documents=documents, progress_service=progress_service, progress_repo=progress_repo
        )
        if not history:
            return None
        return f"books similar to {history[0].title}"

    async def _history_seeds(
        self,
        *,
        documents: DocumentRepository,
        progress_service: ProgressService,
        progress_repo: ReadingProgressRepository,
    ) -> list[_HistoryItem]:
        """The reader's most-recent completed/in-progress documents, titled."""
        grouped = await progress_service.reading_list(progress=progress_repo)
        rows = [*grouped.get(ReadingStatus.COMPLETED, []), *grouped.get(ReadingStatus.READING, [])]
        items: list[_HistoryItem] = []
        for row in rows[:_HISTORY_SEED_LIMIT]:
            document = await documents.get(row.document_id)
            title = document.title if document and document.title else "a document in your library"
            items.append(_HistoryItem(document_id=row.document_id, title=title, status=row.status))
        return items

    async def _seeds(
        self,
        *,
        history: list[_HistoryItem],
        memories: LongTermMemoryRepository,
        memory_service: MemoryService,
    ) -> list[_Seed]:
        """Combine the history and long-term-memory signals into search seeds."""
        seeds = [
            _Seed(
                query=item.title,
                reason=f"Because you {'completed' if item.status is ReadingStatus.COMPLETED else 'are reading'} "
                f"{item.title}",
            )
            for item in history
        ]
        for memory_type in (MemoryType.PREFERENCE, MemoryType.HABIT):
            hits = await memory_service.list_memories(
                memories=memories, type=memory_type, limit=_MEMORY_SEED_LIMIT
            )
            seeds.extend(
                _Seed(query=memory.content, reason=f'You mentioned: "{memory.content}"')
                for memory in hits
            )
        return seeds

    async def _rank_candidates(
        self, *, user_id: uuid.UUID, seeds: list[_Seed], exclude_ids: set[uuid.UUID]
    ) -> list[tuple[uuid.UUID, tuple[float, str]]]:
        """Search the library per seed, keeping each candidate's best-scoring seed.

        A document can match more than one seed; only the highest-scoring
        (score, reason) pair survives, so the explanation always names the
        strongest signal rather than an arbitrary one.
        """
        best: dict[uuid.UUID, tuple[float, str]] = {}
        for seed in seeds:
            query_vector = (await self._embedder.embed([seed.query]))[0]
            hits = await self._vectors.search(
                user_id=user_id, query_vector=query_vector, limit=_CANDIDATES_PER_SEED
            )
            for hit in hits:
                document_id = uuid.UUID(hit.payload["document_id"])
                if document_id in exclude_ids:
                    continue
                if document_id not in best or hit.score > best[document_id][0]:
                    best[document_id] = (hit.score, seed.reason)
        return sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
