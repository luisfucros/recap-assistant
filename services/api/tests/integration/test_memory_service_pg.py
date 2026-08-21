"""Integration tests for MemoryService against real Postgres + Qdrant.

Seeds real users (and, for summaries, a real document) then drives
:class:`MemoryService` to verify the guarantees a unit test (repository/store
faked) can only approximate:

* **Isolation** — user A's retrieve/list/delete never touches user B's rows or
  vector points, even when both hold memories about the same content.
* **Round-trip** — a written memory's content is retrievable by semantic search
  and by listing, and a delete removes both the Postgres row and the Qdrant point.
* **Spoiler-safe bound** — retrieval's ``max_page_end`` excludes summaries that
  reach past it, enforced by the vector store's own filter.
"""

import contextlib
import uuid

import pytest
from api.services.memory_service import MemoryService

from shared.core.enums import DocumentFormat, DocumentStatus, MemoryType
from shared.core.errors import NotFoundError
from shared.models.document import Document
from shared.models.user import User
from shared.repositories import DocumentRepository, LongTermMemoryRepository, UserRepository
from shared.vectorstore import MemoryVectorStore

pytestmark = pytest.mark.integration

_DIM = 8


class _FakeEmbedder:
    """Deterministic embedder: same vector for every text — these tests assert
    the *filter* (isolation, type, page bound), not similarity ranking."""

    @property
    def dim(self) -> int:
        return _DIM

    async def embed(self, texts, *, batch_size=None) -> list[list[float]]:
        return [[0.5] * _DIM for _ in texts]


@pytest.fixture
async def memory_vector_store(qdrant_client):
    """A throwaway ``long_term_memory``-shaped collection, cleaned up after."""
    collection = f"test_long_term_memory_{uuid.uuid4().hex}"
    store = MemoryVectorStore(qdrant_client, collection=collection, dim=_DIM)
    try:
        yield store
    finally:
        with contextlib.suppress(Exception):
            await qdrant_client.delete_collection(collection)


def _service(store: MemoryVectorStore) -> MemoryService:
    return MemoryService(embedder=_FakeEmbedder(), vector_store=store)  # type: ignore[arg-type]


async def _make_user(db_sessionmaker, email: str) -> User:
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email=email))
        await session.commit()
        return user


async def _make_document(db_sessionmaker, user_id: uuid.UUID) -> Document:
    doc = Document(
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/sha256/{uuid.uuid4().hex}.pdf",
        content_sha256=uuid.uuid4().hex,
        format=DocumentFormat.PDF,
        status=DocumentStatus.PENDING,
        embed_model="test-model",
    )
    async with db_sessionmaker() as session:
        await DocumentRepository(session, user_id).add(doc)
        await session.commit()
    return doc


# --- write + retrieve round-trip ------------------------------------------- #


async def test_write_memory_is_retrievable_and_listable(
    db_sessionmaker, memory_vector_store
) -> None:
    user = await _make_user(db_sessionmaker, "reader@example.com")
    service = _service(memory_vector_store)

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        written = await service.write_memory(
            memories=memories, session=session, type=MemoryType.PREFERENCE, content="likes sci-fi"
        )
        assert written.embedding_id is not None

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        found = await service.retrieve(memories=memories, query="reading tastes")
        assert [m.content for m in found] == ["likes sci-fi"]

        listed = await service.list_memories(memories=memories)
        assert [m.content for m in listed] == ["likes sci-fi"]


async def test_write_summary_keys_to_document_and_page_range(
    db_sessionmaker, memory_vector_store
) -> None:
    user = await _make_user(db_sessionmaker, "reader@example.com")
    doc = await _make_document(db_sessionmaker, user.id)
    service = _service(memory_vector_store)

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        summary = await service.write_summary(
            memories=memories,
            session=session,
            document_id=doc.id,
            page_start=1,
            page_end=20,
            content="Odysseus leaves Troy.",
        )

    assert summary.type is MemoryType.SUMMARY
    assert (summary.page_start, summary.page_end) == (1, 20)

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        found = await service.retrieve(
            memories=memories, query="recap", type=MemoryType.SUMMARY, document_id=doc.id
        )
        assert [m.content for m in found] == ["Odysseus leaves Troy."]


async def test_write_memory_restatement_merges_into_existing_row(
    db_sessionmaker, memory_vector_store
) -> None:
    """Restating an already-saved preference refreshes the same row, not a new one.

    The fake embedder here is deterministic (same vector for any text), which
    is exactly the case that matters: a real embedder would place a genuine
    paraphrase this close together too, so the dedup path (MemoryService's
    same-type nearest-neighbor check) must collapse the second write into the
    first rather than leaving two near-identical rows for the same fact.
    """
    user = await _make_user(db_sessionmaker, "reader@example.com")
    service = _service(memory_vector_store)

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        first = await service.write_memory(
            memories=memories, session=session, type=MemoryType.PREFERENCE, content="likes sci-fi"
        )

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        second = await service.write_memory(
            memories=memories,
            session=session,
            type=MemoryType.PREFERENCE,
            content="really likes sci-fi novels",
        )

    assert second.id == first.id

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        listed = await service.list_memories(memories=memories, type=MemoryType.PREFERENCE)
    assert [m.content for m in listed] == ["really likes sci-fi novels"]


# --- isolation -------------------------------------------------------------- #


async def test_retrieve_never_returns_another_users_memory(
    db_sessionmaker, memory_vector_store
) -> None:
    alice = await _make_user(db_sessionmaker, "alice@example.com")
    bob = await _make_user(db_sessionmaker, "bob@example.com")
    service = _service(memory_vector_store)

    async with db_sessionmaker() as session:
        await service.write_memory(
            memories=LongTermMemoryRepository(session, alice.id),
            session=session,
            type=MemoryType.PREFERENCE,
            content="alice likes sci-fi",
        )
    async with db_sessionmaker() as session:
        await service.write_memory(
            memories=LongTermMemoryRepository(session, bob.id),
            session=session,
            type=MemoryType.PREFERENCE,
            content="bob likes romance",
        )

    # Same query, identical (deterministic) embeddings — only the server-side
    # user_id filter (taken from the repository's own scope) can separate them.
    async with db_sessionmaker() as session:
        alice_hits = await service.retrieve(
            memories=LongTermMemoryRepository(session, alice.id), query="preferences"
        )
    assert [m.content for m in alice_hits] == ["alice likes sci-fi"]

    async with db_sessionmaker() as session:
        alice_list = await service.list_memories(
            memories=LongTermMemoryRepository(session, alice.id)
        )
    assert [m.content for m in alice_list] == ["alice likes sci-fi"]


async def test_delete_removes_only_the_owners_row_and_point(
    db_sessionmaker, memory_vector_store
) -> None:
    alice = await _make_user(db_sessionmaker, "alice@example.com")
    bob = await _make_user(db_sessionmaker, "bob@example.com")
    service = _service(memory_vector_store)

    async with db_sessionmaker() as session:
        alice_memory = await service.write_memory(
            memories=LongTermMemoryRepository(session, alice.id),
            session=session,
            type=MemoryType.FACT,
            content="alice fact",
        )

    # Bob can't delete Alice's memory — it isn't in his owner-scoped repository.
    async with db_sessionmaker() as session:
        with pytest.raises(NotFoundError):
            await service.delete_memory(
                memories=LongTermMemoryRepository(session, bob.id),
                session=session,
                memory_id=alice_memory.id,
            )

    async with db_sessionmaker() as session:
        await service.delete_memory(
            memories=LongTermMemoryRepository(session, alice.id),
            session=session,
            memory_id=alice_memory.id,
        )

    async with db_sessionmaker() as session:
        assert await LongTermMemoryRepository(session, alice.id).get(alice_memory.id) is None
        remaining = await service.retrieve(
            memories=LongTermMemoryRepository(session, alice.id), query="fact"
        )
    assert remaining == []


# --- spoiler-safe bound ------------------------------------------------------ #


async def test_retrieve_bounds_summaries_by_max_page_end(
    db_sessionmaker, memory_vector_store
) -> None:
    user = await _make_user(db_sessionmaker, "reader@example.com")
    doc = await _make_document(db_sessionmaker, user.id)
    service = _service(memory_vector_store)

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        await service.write_summary(
            memories=memories,
            session=session,
            document_id=doc.id,
            page_start=1,
            page_end=20,
            content="early",
        )
        await service.write_summary(
            memories=memories,
            session=session,
            document_id=doc.id,
            page_start=41,
            page_end=60,
            content="later",
        )

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        bounded = await service.retrieve(
            memories=memories,
            query="recap",
            type=MemoryType.SUMMARY,
            document_id=doc.id,
            max_page_end=30,
        )

    assert [m.content for m in bounded] == ["early"]
