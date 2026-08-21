"""Unit tests for MemoryService (write/retrieve/view/delete, spoiler-safe bound).

The embedder, vector store, and repository are faked at the boundary; the
type-vs-summary write rules, page-range validation, server-side ``user_id``
injection, hydration/ordering, and delete-ordering under test are real.
"""

import uuid
from typing import Any

import pytest
from api.services.memory_service import MemoryService

from shared.core.enums import MemoryType
from shared.core.errors import InvalidInputError, NotFoundError
from shared.models.memory import LongTermMemory
from shared.vectorstore import ScoredMemory

pytestmark = pytest.mark.unit

OWNER = uuid.uuid4()


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
    def __init__(self, hits: list[ScoredMemory] | None = None) -> None:
        self._hits = hits or []
        self.ensure_collection_calls = 0
        self.upserted: list[dict[str, Any]] = []
        self.search_kwargs: dict[str, Any] | None = None
        self.deleted: list[dict[str, Any]] = []

    async def ensure_collection(self) -> None:
        self.ensure_collection_calls += 1

    async def upsert(self, *, ids, vectors, payloads) -> None:
        self.upserted.append({"ids": ids, "vectors": vectors, "payloads": payloads})

    async def search(self, **kwargs: Any) -> list[ScoredMemory]:
        self.search_kwargs = kwargs
        return self._hits

    async def delete(self, *, user_id, memory_id) -> None:
        self.deleted.append({"user_id": user_id, "memory_id": memory_id})


class _FakeMemoryRepo:
    """Owner-scoped fake mirroring LongTermMemoryRepository's method surface."""

    def __init__(self, rows: dict[uuid.UUID, LongTermMemory] | None = None) -> None:
        self.user_id = OWNER
        self._rows = rows or {}
        self.added: list[LongTermMemory] = []
        self.deleted: list[LongTermMemory] = []
        self.requested_ids: list[uuid.UUID] = []

    async def add(self, memory: LongTermMemory) -> LongTermMemory:
        self.added.append(memory)
        self._rows[memory.id] = memory
        return memory

    async def list_by_ids(self, ids):
        self.requested_ids = list(ids)
        return [self._rows[i] for i in ids if i in self._rows]

    async def get_or_404(self, memory_id: uuid.UUID) -> LongTermMemory:
        row = self._rows.get(memory_id)
        if row is None:
            raise NotFoundError()
        return row

    async def delete(self, memory: LongTermMemory) -> None:
        self.deleted.append(memory)
        self._rows.pop(memory.id, None)

    async def list_by_type(self, memory_type, *, limit=100, offset=0):
        return [r for r in self._rows.values() if r.type == memory_type]

    async def list_recent(self, *, limit=100, offset=0):
        return list(self._rows.values())


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


def _service(
    hits: list[ScoredMemory] | None = None,
) -> tuple[MemoryService, _FakeVectorStore, _FakeEmbedder]:
    embedder, store = _FakeEmbedder(), _FakeVectorStore(hits)
    return (
        MemoryService(embedder=embedder, vector_store=store),  # type: ignore[arg-type]
        store,
        embedder,
    )


def _hit(memory_id: uuid.UUID, *, score: float) -> ScoredMemory:
    return ScoredMemory(id=str(memory_id), score=score, payload={})


# --- write_memory -------------------------------------------------------- #


async def test_write_memory_persists_embeds_and_commits() -> None:
    service, store, embedder = _service()
    memories, session = _FakeMemoryRepo(), _FakeSession()

    memory = await service.write_memory(
        memories=memories,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        type=MemoryType.PREFERENCE,
        content="likes sci-fi",
    )

    assert memory.user_id == OWNER
    assert memory.document_id is None
    assert embedder.embedded == ["likes sci-fi"]
    assert store.ensure_collection_calls == 1
    assert store.upserted[0]["ids"] == [str(memory.id)]
    assert memory.embedding_id == str(memory.id)
    assert session.commits == 1


async def test_write_memory_rejects_summary_type() -> None:
    service, _, _ = _service()

    with pytest.raises(InvalidInputError):
        await service.write_memory(
            memories=_FakeMemoryRepo(),  # type: ignore[arg-type]
            session=_FakeSession(),  # type: ignore[arg-type]
            type=MemoryType.SUMMARY,
            content="x",
        )


async def test_write_memory_restatement_updates_existing_instead_of_duplicating() -> None:
    """A near-duplicate (score above the dedup threshold) refreshes in place."""
    existing_id = uuid.uuid4()
    existing = LongTermMemory(
        id=existing_id, user_id=OWNER, type=MemoryType.PREFERENCE, content="likes sci-fi"
    )
    memories = _FakeMemoryRepo({existing_id: existing})
    session = _FakeSession()
    service, store, _ = _service([_hit(existing_id, score=0.97)])

    result = await service.write_memory(
        memories=memories,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        type=MemoryType.PREFERENCE,
        content="really likes sci-fi novels",
    )

    assert result is existing
    assert result.content == "really likes sci-fi novels"
    assert memories.added == []  # no new row inserted
    assert store.search_kwargs["type"] is MemoryType.PREFERENCE
    assert store.upserted[0]["ids"] == [str(existing_id)]  # re-indexed under the same point id
    assert session.commits == 1


async def test_write_memory_below_threshold_inserts_new_row() -> None:
    """A related-but-distinct memory (score below the threshold) is not a duplicate."""
    other_id = uuid.uuid4()
    other = LongTermMemory(
        id=other_id, user_id=OWNER, type=MemoryType.PREFERENCE, content="likes sci-fi"
    )
    memories = _FakeMemoryRepo({other_id: other})
    service, store, _ = _service([_hit(other_id, score=0.5)])

    result = await service.write_memory(
        memories=memories,  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        type=MemoryType.PREFERENCE,
        content="likes fantasy",
    )

    assert result is not other
    assert result.content == "likes fantasy"
    assert memories.added == [result]
    assert len(store.upserted) == 1


async def test_write_memory_no_hits_inserts_new_row() -> None:
    """No existing memories of this type at all — nothing to dedup against."""
    memories = _FakeMemoryRepo()
    service, _, _ = _service([])

    result = await service.write_memory(
        memories=memories,  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        type=MemoryType.FACT,
        content="is 34 years old",
    )

    assert memories.added == [result]


async def test_write_memory_orphaned_hit_falls_back_to_insert() -> None:
    """A high-scoring hit whose Postgres row is gone (e.g. deleted) isn't reused."""
    gone_id = uuid.uuid4()
    memories = _FakeMemoryRepo()  # no rows — the hit's row doesn't exist
    service, _, _ = _service([_hit(gone_id, score=0.99)])

    result = await service.write_memory(
        memories=memories,  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        type=MemoryType.HABIT,
        content="reads before bed",
    )

    assert memories.added == [result]
    assert result.id != gone_id


# --- write_summary -------------------------------------------------------- #


async def test_write_summary_persists_with_page_range() -> None:
    service, store, _ = _service()
    memories, session = _FakeMemoryRepo(), _FakeSession()
    document_id = uuid.uuid4()

    memory = await service.write_summary(
        memories=memories,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        document_id=document_id,
        page_start=1,
        page_end=20,
        content="Odysseus leaves Troy.",
    )

    assert memory.type is MemoryType.SUMMARY
    assert memory.document_id == document_id
    assert (memory.page_start, memory.page_end) == (1, 20)
    assert store.upserted
    assert session.commits == 1


@pytest.mark.parametrize(("page_start", "page_end"), [(0, 10), (20, 10)])
async def test_write_summary_rejects_invalid_page_range(page_start: int, page_end: int) -> None:
    service, _, _ = _service()

    with pytest.raises(InvalidInputError):
        await service.write_summary(
            memories=_FakeMemoryRepo(),  # type: ignore[arg-type]
            session=_FakeSession(),  # type: ignore[arg-type]
            document_id=uuid.uuid4(),
            page_start=page_start,
            page_end=page_end,
            content="x",
        )


# --- retrieve (semantic + typed + spoiler-safe bound) --------------------- #


async def test_retrieve_passes_owner_id_from_repository_not_a_parameter() -> None:
    memory_id = uuid.uuid4()
    memories = _FakeMemoryRepo(
        {memory_id: LongTermMemory(id=memory_id, user_id=OWNER, type=MemoryType.FACT, content="x")}
    )
    service, store, embedder = _service([_hit(memory_id, score=0.9)])

    await service.retrieve(memories=memories, query="what do I like?")  # type: ignore[arg-type]

    assert embedder.embedded == ["what do I like?"]
    assert store.search_kwargs["user_id"] == OWNER


async def test_retrieve_forwards_type_document_and_spoiler_safe_bound() -> None:
    document_id = uuid.uuid4()
    service, store, _ = _service([])

    await service.retrieve(
        memories=_FakeMemoryRepo(),  # type: ignore[arg-type]
        query="recap",
        type=MemoryType.SUMMARY,
        document_id=document_id,
        max_page_end=40,
        limit=3,
    )

    assert store.search_kwargs["type"] is MemoryType.SUMMARY
    assert store.search_kwargs["document_id"] == document_id
    assert store.search_kwargs["max_page_end"] == 40
    assert store.search_kwargs["limit"] == 3


async def test_retrieve_hydrates_and_preserves_score_order() -> None:
    m1, m2 = uuid.uuid4(), uuid.uuid4()
    memories = _FakeMemoryRepo(
        {
            m1: LongTermMemory(id=m1, user_id=OWNER, type=MemoryType.FACT, content="first"),
            m2: LongTermMemory(id=m2, user_id=OWNER, type=MemoryType.FACT, content="second"),
        }
    )
    service, _, _ = _service([_hit(m1, score=0.9), _hit(m2, score=0.5)])

    result = await service.retrieve(memories=memories, query="q")  # type: ignore[arg-type]

    assert [r.content for r in result] == ["first", "second"]


async def test_retrieve_drops_missing_rows() -> None:
    m1 = uuid.uuid4()
    m2_gone = uuid.uuid4()
    memories = _FakeMemoryRepo(
        {m1: LongTermMemory(id=m1, user_id=OWNER, type=MemoryType.FACT, content="here")}
    )
    service, _, _ = _service([_hit(m1, score=0.9), _hit(m2_gone, score=0.5)])

    result = await service.retrieve(memories=memories, query="q")  # type: ignore[arg-type]

    assert [r.id for r in result] == [m1]


# --- list_memories --------------------------------------------------------- #


async def test_list_memories_filters_by_type_when_given() -> None:
    service, _, _ = _service()
    fact = LongTermMemory(id=uuid.uuid4(), user_id=OWNER, type=MemoryType.FACT, content="a")
    pref = LongTermMemory(id=uuid.uuid4(), user_id=OWNER, type=MemoryType.PREFERENCE, content="b")
    memories = _FakeMemoryRepo({fact.id: fact, pref.id: pref})

    result = await service.list_memories(memories=memories, type=MemoryType.FACT)  # type: ignore[arg-type]

    assert list(result) == [fact]


async def test_list_memories_without_type_returns_all() -> None:
    service, _, _ = _service()
    fact = LongTermMemory(id=uuid.uuid4(), user_id=OWNER, type=MemoryType.FACT, content="a")
    memories = _FakeMemoryRepo({fact.id: fact})

    result = await service.list_memories(memories=memories)  # type: ignore[arg-type]

    assert list(result) == [fact]


# --- delete_memory ---------------------------------------------------------- #


async def test_delete_memory_removes_vector_then_row_then_commits() -> None:
    memory_id = uuid.uuid4()
    memory = LongTermMemory(id=memory_id, user_id=OWNER, type=MemoryType.FACT, content="x")
    memories = _FakeMemoryRepo({memory_id: memory})
    session = _FakeSession()
    service, store, _ = _service()

    await service.delete_memory(memories=memories, session=session, memory_id=memory_id)  # type: ignore[arg-type]

    assert store.deleted == [{"user_id": OWNER, "memory_id": memory_id}]
    assert memories.deleted == [memory]
    assert session.commits == 1


async def test_delete_memory_missing_raises_not_found() -> None:
    service, _, _ = _service()

    with pytest.raises(NotFoundError):
        await service.delete_memory(
            memories=_FakeMemoryRepo(),  # type: ignore[arg-type]
            session=_FakeSession(),  # type: ignore[arg-type]
            memory_id=uuid.uuid4(),
        )
