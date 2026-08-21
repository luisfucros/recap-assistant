"""Unit tests for the RetrievalService (read-range, hydration, dedup, citations).

The embedder, vector store, and repositories are faked at the boundary; the
read-range resolution, text hydration, near-duplicate collapse, and citation
mapping under test are real.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from api.services.retrieval_service import RetrievalService

from shared.core.enums import ReadingStatus
from shared.models.reading import ReadingProgress
from shared.vectorstore import ScoredChunk

pytestmark = pytest.mark.unit


class _FakeEmbedder:
    def __init__(self) -> None:
        self.embedded: list[str] = []

    @property
    def dim(self) -> int:
        return 3

    async def embed(self, texts, *, batch_size=None):
        self.embedded.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeVectorStore:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self._hits = hits
        self.search_kwargs: dict[str, Any] | None = None

    async def search(self, **kwargs: Any) -> list[ScoredChunk]:
        self.search_kwargs = kwargs
        return self._hits


class _FakeProgressRepo:
    def __init__(self, row: ReadingProgress | None) -> None:
        self._row = row

    async def get_by_document(self, document_id: uuid.UUID) -> ReadingProgress | None:
        return self._row


class _FakeChunkRepo:
    """Returns chunk rows (id + text) for the requested ids."""

    def __init__(self, rows: dict[uuid.UUID, str]) -> None:
        self._rows = rows
        self.requested: list[uuid.UUID] = []

    async def list_by_ids(self, ids):
        self.requested = list(ids)
        return [SimpleNamespace(id=i, text=self._rows[i]) for i in ids if i in self._rows]


def _settings(top_k: int = 8) -> SimpleNamespace:
    return SimpleNamespace(retrieval_top_k=top_k)


def _service(
    hits: list[ScoredChunk], *, top_k: int = 8
) -> tuple[RetrievalService, _FakeVectorStore, _FakeEmbedder]:
    embedder, store = _FakeEmbedder(), _FakeVectorStore(hits)
    service = RetrievalService(
        embedder=embedder,  # type: ignore[arg-type]
        vector_store=store,  # type: ignore[arg-type]
        settings=_settings(top_k),  # type: ignore[arg-type]
    )
    return service, store, embedder


def _hit(chunk_id: uuid.UUID, *, score: float, content_hash: str, **payload: Any) -> ScoredChunk:
    base = {"document_id": str(uuid.uuid4()), "content_hash": content_hash}
    base.update(payload)
    return ScoredChunk(id=str(chunk_id), score=score, payload=base)


def _progress(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    *,
    current_page: int,
    spoiler_safe: bool | None = None,
) -> ReadingProgress:
    return ReadingProgress(
        id=uuid.uuid4(),
        user_id=user_id,
        document_id=document_id,
        current_page=current_page,
        last_summarized_page=0,
        status=ReadingStatus.READING,
        spoiler_safe=spoiler_safe,
    )


# --- read-range bounding ------------------------------------------------- #


async def test_targeted_document_bounds_to_current_page() -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    cid = uuid.uuid4()
    service, store, _ = _service([_hit(cid, score=0.9, content_hash="h1")])

    await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(_progress(owner, document_id, current_page=50)),
        chunks=_FakeChunkRepo({cid: "text"}),
        document_id=document_id,
    )

    assert store.search_kwargs["max_page_end"] == 50
    assert store.search_kwargs["user_id"] == owner  # server-supplied owner
    assert store.search_kwargs["document_id"] == document_id


async def test_include_unread_drops_page_bound() -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    cid = uuid.uuid4()
    service, store, _ = _service([_hit(cid, score=0.9, content_hash="h1")])

    await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(_progress(owner, document_id, current_page=50)),
        chunks=_FakeChunkRepo({cid: "text"}),
        document_id=document_id,
        include_unread=True,
    )

    assert store.search_kwargs["max_page_end"] is None


async def test_library_wide_search_has_no_page_bound() -> None:
    owner = uuid.uuid4()
    cid = uuid.uuid4()
    service, store, _ = _service([_hit(cid, score=0.9, content_hash="h1")])

    await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(None),
        chunks=_FakeChunkRepo({cid: "text"}),
        document_id=None,
    )

    assert store.search_kwargs["max_page_end"] is None
    assert store.search_kwargs["document_id"] is None


async def test_untracked_document_bounds_to_zero() -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    service, store, _ = _service([])

    await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(None),  # no progress row yet
        chunks=_FakeChunkRepo({}),
        document_id=document_id,
    )

    # Nothing read → read-range default surfaces nothing (page_end <= 0).
    assert store.search_kwargs["max_page_end"] == 0


# --- hydration + ordering ------------------------------------------------ #


async def test_hydration_attaches_text_and_preserves_score_order() -> None:
    owner = uuid.uuid4()
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    hits = [_hit(c1, score=0.9, content_hash="h1"), _hit(c2, score=0.5, content_hash="h2")]
    service, _, _ = _service(hits)

    result = await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(None),
        chunks=_FakeChunkRepo({c1: "first", c2: "second"}),
    )

    assert [c.text for c in result.chunks] == ["first", "second"]
    assert [c.chunk_id for c in result.chunks] == [c1, c2]


async def test_missing_chunk_row_is_dropped() -> None:
    owner = uuid.uuid4()
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    hits = [_hit(c1, score=0.9, content_hash="h1"), _hit(c2, score=0.5, content_hash="h2")]
    service, _, _ = _service(hits)

    # c2 has no row (deleted between index and query) → dropped, not text-less.
    result = await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(None),
        chunks=_FakeChunkRepo({c1: "first"}),
    )

    assert [c.chunk_id for c in result.chunks] == [c1]


# --- FR-1.12 near-duplicate collapse ------------------------------------- #


async def test_duplicate_content_hash_collapsed_keeping_top_score() -> None:
    owner = uuid.uuid4()
    c1, c2, c3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    hits = [
        _hit(c1, score=0.9, content_hash="dup"),
        _hit(c2, score=0.7, content_hash="dup"),  # same passage, lower score → dropped
        _hit(c3, score=0.6, content_hash="other"),
    ]
    service, _, _ = _service(hits)
    chunk_repo = _FakeChunkRepo({c1: "a", c2: "a", c3: "b"})

    result = await service.retrieve(
        query="q", user_id=owner, progress=_FakeProgressRepo(None), chunks=chunk_repo
    )

    assert [c.chunk_id for c in result.chunks] == [c1, c3]
    # The dropped duplicate is never hydrated from the DB.
    assert c2 not in chunk_repo.requested


# --- citation mapping ---------------------------------------------------- #


async def test_citation_maps_document_labels_and_page_span() -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    cid = uuid.uuid4()
    hit = _hit(
        cid,
        score=0.9,
        content_hash="h1",
        document_id=str(document_id),
        title="The Book",
        author="An Author",
        page_start=10,
        page_end=12,
        chapter="1",
        section="A",
    )
    service, _, _ = _service([hit])

    result = await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(None),
        chunks=_FakeChunkRepo({cid: "passage"}),
    )

    (chunk,) = result.chunks
    assert chunk.citation.document_id == document_id
    assert chunk.citation.title == "The Book"
    assert chunk.citation.author == "An Author"
    assert (chunk.citation.page_start, chunk.citation.page_end) == (10, 12)
    assert result.citations == [chunk.citation]
    assert (chunk.chapter, chunk.section) == ("1", "A")


async def test_limit_defaults_to_retrieval_top_k() -> None:
    owner = uuid.uuid4()
    service, store, _ = _service([], top_k=5)

    await service.retrieve(
        query="q", user_id=owner, progress=_FakeProgressRepo(None), chunks=_FakeChunkRepo({})
    )

    assert store.search_kwargs["limit"] == 5


# --- spoiler-safe (FR-18) hard filter ------------------------------------ #


async def test_spoiler_safe_on_forces_bound_even_with_include_unread() -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    cid = uuid.uuid4()
    service, store, _ = _service([_hit(cid, score=0.9, content_hash="h1")])

    await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(_progress(owner, document_id, current_page=30)),
        chunks=_FakeChunkRepo({cid: "text"}),
        document_id=document_id,
        include_unread=True,  # ignored under spoiler-safe
        user_spoiler_safe=True,
    )

    assert store.search_kwargs["max_page_end"] == 30


async def test_per_query_override_can_disable_spoiler_safe() -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    cid = uuid.uuid4()
    service, store, _ = _service([_hit(cid, score=0.9, content_hash="h1")])

    await service.retrieve(
        query="q",
        user_id=owner,
        progress=_FakeProgressRepo(_progress(owner, document_id, current_page=30)),
        chunks=_FakeChunkRepo({cid: "text"}),
        document_id=document_id,
        include_unread=True,
        user_spoiler_safe=True,
        spoiler_safe_override=False,  # per-query wins → include_unread lifts bound
    )

    assert store.search_kwargs["max_page_end"] is None


async def test_per_document_override_disables_spoiler_safe() -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    cid = uuid.uuid4()
    service, store, _ = _service([_hit(cid, score=0.9, content_hash="h1")])

    await service.retrieve(
        query="q",
        user_id=owner,
        # per-doc override False beats the user's global True.
        progress=_FakeProgressRepo(
            _progress(owner, document_id, current_page=30, spoiler_safe=False)
        ),
        chunks=_FakeChunkRepo({cid: "text"}),
        document_id=document_id,
        include_unread=True,
        user_spoiler_safe=True,
    )

    assert store.search_kwargs["max_page_end"] is None
