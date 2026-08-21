"""Unit tests for RecommendationService (ranking/explanation assembly, deps mocked).

The embedder, chunk vector store, document/memory repos, and progress/memory
services are faked at the boundary; the seed-building, ranking, and
explanation logic under test is real.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from api.services.recommendation_service import Recommendation, RecommendationService

from shared.core.enums import MemoryType, ReadingStatus
from shared.models.reading import ReadingProgress
from shared.providers.base import SearchResult
from shared.vectorstore import ScoredChunk

pytestmark = pytest.mark.unit

USER_ID = uuid.uuid4()


class _FakeEmbedder:
    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeChunkVectorStore:
    """Returns one queued batch of hits per ``search()`` call, in order."""

    def __init__(self, batches: list[list[ScoredChunk]] | None = None) -> None:
        self._batches = list(batches or [])
        self.search_calls: list[dict[str, Any]] = []

    async def search(self, **kwargs: Any) -> list[ScoredChunk]:
        self.search_calls.append(kwargs)
        return self._batches.pop(0) if self._batches else []


class _FakeDocuments:
    def __init__(self, rows: dict[uuid.UUID, Any]) -> None:
        self._rows = rows

    async def get(self, document_id: uuid.UUID) -> Any:
        return self._rows.get(document_id)


class _FakeProgressService:
    def __init__(self, grouped: dict[ReadingStatus, list[ReadingProgress]] | None = None) -> None:
        self._grouped = grouped or {}

    async def reading_list(self, *, progress: Any) -> dict:
        return self._grouped


class _FakeMemoryService:
    def __init__(self, by_type: dict[MemoryType, list[Any]] | None = None) -> None:
        self._by_type = by_type or {}

    async def list_memories(self, *, memories: Any, type: MemoryType, limit: int) -> list[Any]:
        return self._by_type.get(type, [])[:limit]


def _hit(document_id: uuid.UUID, *, score: float) -> ScoredChunk:
    return ScoredChunk(id=str(uuid.uuid4()), score=score, payload={"document_id": str(document_id)})


def _document(title: str, author: str | None = None) -> Any:
    return SimpleNamespace(title=title, author=author)


def _progress_row(document_id: uuid.UUID, *, status: ReadingStatus) -> ReadingProgress:
    return ReadingProgress(
        user_id=USER_ID,
        document_id=document_id,
        current_page=10,
        last_summarized_page=0,
        status=status,
    )


def _service(vector_store: _FakeChunkVectorStore) -> tuple[RecommendationService, _FakeEmbedder]:
    embedder = _FakeEmbedder()
    return RecommendationService(embedder=embedder, vector_store=vector_store), embedder


# --- recommend_from_library: no signal to recommend from --------------------- #


async def test_returns_empty_without_history_or_memory() -> None:
    service, _ = _service(_FakeChunkVectorStore())

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=_FakeDocuments({}),
        progress_repo=SimpleNamespace(),
        progress_service=_FakeProgressService(),
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
    )

    assert recs == []


# --- history signal ----------------------------------------------------------- #


async def test_recommends_a_similar_library_document_explained_by_history() -> None:
    seed_id, candidate_id = uuid.uuid4(), uuid.uuid4()
    documents = _FakeDocuments(
        {seed_id: _document("The Odyssey"), candidate_id: _document("The Iliad", "Homer")}
    )
    progress_service = _FakeProgressService(
        {ReadingStatus.COMPLETED: [_progress_row(seed_id, status=ReadingStatus.COMPLETED)]}
    )
    vector_store = _FakeChunkVectorStore([[_hit(candidate_id, score=0.8)]])
    service, embedder = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=progress_service,
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
    )

    assert len(recs) == 1
    assert recs[0] == Recommendation(
        document_id=candidate_id,
        title="The Iliad",
        author="Homer",
        reason="Because you completed The Odyssey",
        score=0.8,
    )
    assert "The Odyssey" in embedder.embedded
    assert vector_store.search_calls[0]["user_id"] == USER_ID


async def test_reading_status_reason_says_are_reading() -> None:
    seed_id, candidate_id = uuid.uuid4(), uuid.uuid4()
    documents = _FakeDocuments({seed_id: _document("Dune"), candidate_id: _document("Foundation")})
    progress_service = _FakeProgressService(
        {ReadingStatus.READING: [_progress_row(seed_id, status=ReadingStatus.READING)]}
    )
    vector_store = _FakeChunkVectorStore([[_hit(candidate_id, score=0.5)]])
    service, _ = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=progress_service,
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
    )

    assert recs[0].reason == "Because you are reading Dune"


async def test_excludes_the_seed_documents_themselves() -> None:
    seed_id = uuid.uuid4()
    documents = _FakeDocuments({seed_id: _document("The Odyssey")})
    progress_service = _FakeProgressService(
        {ReadingStatus.COMPLETED: [_progress_row(seed_id, status=ReadingStatus.COMPLETED)]}
    )
    # The similarity search echoes back the seed's own document — must be dropped.
    vector_store = _FakeChunkVectorStore([[_hit(seed_id, score=0.99)]])
    service, _ = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=progress_service,
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
    )

    assert recs == []


async def test_ranks_multiple_candidates_by_score_descending() -> None:
    seed_id, low_id, high_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    documents = _FakeDocuments(
        {
            seed_id: _document("Seed"),
            low_id: _document("Low match"),
            high_id: _document("High match"),
        }
    )
    progress_service = _FakeProgressService(
        {ReadingStatus.COMPLETED: [_progress_row(seed_id, status=ReadingStatus.COMPLETED)]}
    )
    vector_store = _FakeChunkVectorStore([[_hit(low_id, score=0.3), _hit(high_id, score=0.9)]])
    service, _ = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=progress_service,
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
    )

    assert [r.title for r in recs] == ["High match", "Low match"]


async def test_limit_bounds_the_returned_recommendations() -> None:
    seed_id = uuid.uuid4()
    candidates = [uuid.uuid4() for _ in range(3)]
    documents = _FakeDocuments(
        {seed_id: _document("Seed"), **{c: _document(str(c)) for c in candidates}}
    )
    progress_service = _FakeProgressService(
        {ReadingStatus.COMPLETED: [_progress_row(seed_id, status=ReadingStatus.COMPLETED)]}
    )
    vector_store = _FakeChunkVectorStore(
        [[_hit(c, score=0.1 * i) for i, c in enumerate(candidates, start=1)]]
    )
    service, _ = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=progress_service,
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
        limit=2,
    )

    assert len(recs) == 2


async def test_skips_a_candidate_whose_document_row_is_missing() -> None:
    seed_id, missing_id = uuid.uuid4(), uuid.uuid4()
    documents = _FakeDocuments({seed_id: _document("Seed")})  # missing_id absent
    progress_service = _FakeProgressService(
        {ReadingStatus.COMPLETED: [_progress_row(seed_id, status=ReadingStatus.COMPLETED)]}
    )
    vector_store = _FakeChunkVectorStore([[_hit(missing_id, score=0.7)]])
    service, _ = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=progress_service,
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
    )

    assert recs == []


# --- long-term-memory signal --------------------------------------------------- #


async def test_uses_a_stated_preference_as_a_search_seed() -> None:
    candidate_id = uuid.uuid4()
    documents = _FakeDocuments({candidate_id: _document("Fast-paced thriller")})
    memory_service = _FakeMemoryService(
        {MemoryType.PREFERENCE: [SimpleNamespace(content="fast-paced thrillers")]}
    )
    vector_store = _FakeChunkVectorStore([[_hit(candidate_id, score=0.6)]])
    service, embedder = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=_FakeProgressService(),
        memories=SimpleNamespace(),
        memory_service=memory_service,
    )

    assert recs[0].reason == 'You mentioned: "fast-paced thrillers"'
    assert "fast-paced thrillers" in embedder.embedded


async def test_keeps_the_best_scoring_seed_reason_for_a_shared_candidate() -> None:
    seed_id, candidate_id = uuid.uuid4(), uuid.uuid4()
    documents = _FakeDocuments({seed_id: _document("Seed"), candidate_id: _document("Candidate")})
    progress_service = _FakeProgressService(
        {ReadingStatus.COMPLETED: [_progress_row(seed_id, status=ReadingStatus.COMPLETED)]}
    )
    memory_service = _FakeMemoryService(
        {MemoryType.PREFERENCE: [SimpleNamespace(content="mysteries")]}
    )
    # History seed matches weakly; the memory seed matches strongly — the
    # stronger reason should win the explanation.
    vector_store = _FakeChunkVectorStore(
        [[_hit(candidate_id, score=0.2)], [_hit(candidate_id, score=0.95)]]
    )
    service, _ = _service(vector_store)

    recs = await service.recommend_from_library(
        user_id=USER_ID,
        documents=documents,
        progress_repo=SimpleNamespace(),
        progress_service=progress_service,
        memories=SimpleNamespace(),
        memory_service=memory_service,
    )

    assert len(recs) == 1
    assert recs[0].score == 0.95
    assert recs[0].reason == 'You mentioned: "mysteries"'


# --- external (web) signal ----------------------------------------------------- #


class _FakeWebSearch:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        self.calls.append({"query": query, "count": count})
        return self._results


async def test_recommend_from_web_maps_search_hits() -> None:
    service, _ = _service(_FakeChunkVectorStore())
    web_search = _FakeWebSearch(
        [SearchResult(title="A Great Book", url="http://x", snippet="...", score=0.7)]
    )

    recs = await service.recommend_from_web(web_search=web_search, query="books like Dune", limit=3)

    assert recs == [
        Recommendation(
            title="A Great Book",
            reason='From a web search for "books like Dune"',
            url="http://x",
            score=0.7,
        )
    ]
    assert web_search.calls == [{"query": "books like Dune", "count": 3}]


async def test_default_web_query_uses_top_history_title() -> None:
    seed_id = uuid.uuid4()
    documents = _FakeDocuments({seed_id: _document("The Odyssey")})
    progress_service = _FakeProgressService(
        {ReadingStatus.COMPLETED: [_progress_row(seed_id, status=ReadingStatus.COMPLETED)]}
    )
    service, _ = _service(_FakeChunkVectorStore())

    query = await service.default_web_query(
        documents=documents, progress_service=progress_service, progress_repo=SimpleNamespace()
    )

    assert query == "books similar to The Odyssey"


async def test_default_web_query_is_none_without_history() -> None:
    service, _ = _service(_FakeChunkVectorStore())

    query = await service.default_web_query(
        documents=_FakeDocuments({}),
        progress_service=_FakeProgressService(),
        progress_repo=SimpleNamespace(),
    )

    assert query is None
