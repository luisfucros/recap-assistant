"""Integration tests for long-term memory: the repository (Postgres) and the
vector store (Qdrant).

Exercises the per-user isolation invariant and FK-cascade behavior a unit test
can only approximate with compiled SQL, plus the real Qdrant collection
lifecycle (bootstrap, upsert, isolated search, delete) — the two halves
:class:`~api.services.memory_service.MemoryService` will compose.
"""

import uuid

import pytest
from sqlalchemy import text

from shared.core.enums import DocumentFormat, DocumentStatus, MemoryType
from shared.models.document import Document
from shared.models.memory import LongTermMemory
from shared.models.user import User
from shared.repositories import DocumentRepository, LongTermMemoryRepository, UserRepository
from shared.vectorstore import MemoryVectorStore, build_memory_payload, memory_point_id

pytestmark = pytest.mark.integration


async def _make_user(session, email: str = "reader@example.com") -> User:
    user = await UserRepository(session).add(User(email=email))
    await session.commit()
    return user


def _document(user_id: uuid.UUID, sha: str) -> Document:
    return Document(
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/sha256/{sha}.pdf",
        content_sha256=sha,
        format=DocumentFormat.PDF,
        status=DocumentStatus.PENDING,
        embed_model="test-model",
    )


# --- LongTermMemoryRepository (Postgres) -------------------------------------- #


async def test_memory_scoped_reads_isolate_users(db_session) -> None:
    alice = await _make_user(db_session, "alice@example.com")
    bob = await _make_user(db_session, "bob@example.com")

    alice_repo = LongTermMemoryRepository(db_session, alice.id)
    memory = await alice_repo.add(
        LongTermMemory(user_id=alice.id, type=MemoryType.PREFERENCE, content="likes sci-fi")
    )
    await db_session.commit()

    assert await LongTermMemoryRepository(db_session, bob.id).get(memory.id) is None
    assert await alice_repo.get(memory.id) is not None


async def test_deleting_document_cascades_to_its_summary_memories(db_session) -> None:
    user = await _make_user(db_session)
    doc = await DocumentRepository(db_session, user.id).add(_document(user.id, "a" * 64))
    await db_session.commit()

    memories = LongTermMemoryRepository(db_session, user.id)
    await memories.add(
        LongTermMemory(
            user_id=user.id,
            document_id=doc.id,
            type=MemoryType.SUMMARY,
            content="Odysseus leaves Troy.",
            page_start=1,
            page_end=10,
        )
    )
    await db_session.commit()
    assert len(await memories.list_by_document(doc.id)) == 1

    # FK ondelete=CASCADE removes the memory with the document.
    await db_session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc.id})
    await db_session.commit()
    assert await memories.list_by_document(doc.id) == []


async def test_list_summaries_covering_orders_by_page_start_and_bounds_spoiler_safe(
    db_session,
) -> None:
    user = await _make_user(db_session)
    doc = await DocumentRepository(db_session, user.id).add(_document(user.id, "b" * 64))
    await db_session.commit()

    memories = LongTermMemoryRepository(db_session, user.id)
    for start, end, content in ((41, 60, "later"), (1, 20, "first"), (21, 40, "middle")):
        await memories.add(
            LongTermMemory(
                user_id=user.id,
                document_id=doc.id,
                type=MemoryType.SUMMARY,
                content=content,
                page_start=start,
                page_end=end,
            )
        )
    await db_session.commit()

    found = await memories.list_summaries_covering(doc.id)
    assert [m.content for m in found] == ["first", "middle", "later"]

    # Spoiler-safe bound: only summaries that don't reach past page 40.
    bounded = await memories.list_summaries_covering(doc.id, max_page_end=40)
    assert [m.content for m in bounded] == ["first", "middle"]


async def test_delete_removes_the_row(db_session) -> None:
    user = await _make_user(db_session)
    memories = LongTermMemoryRepository(db_session, user.id)
    memory = await memories.add(
        LongTermMemory(user_id=user.id, type=MemoryType.FACT, content="reads on the train")
    )
    await db_session.commit()

    await memories.delete(memory)
    await db_session.commit()

    assert await memories.get(memory.id) is None


# --- MemoryVectorStore (Qdrant) ------------------------------------------------ #


@pytest.fixture
async def memory_collection(qdrant_client):
    """A throwaway ``long_term_memory``-shaped collection, cleaned up after."""
    collection = f"test_long_term_memory_{uuid.uuid4().hex}"
    store = MemoryVectorStore(qdrant_client, collection=collection, dim=4)
    await store.ensure_collection()
    try:
        yield store
    finally:
        await qdrant_client.delete_collection(collection)


async def test_search_is_isolated_by_user_id(memory_collection) -> None:
    alice, bob = uuid.uuid4(), uuid.uuid4()
    alice_memory = LongTermMemory(
        id=uuid.uuid4(), user_id=alice, type=MemoryType.PREFERENCE, content="likes sci-fi"
    )
    bob_memory = LongTermMemory(
        id=uuid.uuid4(), user_id=bob, type=MemoryType.PREFERENCE, content="likes romance"
    )
    await memory_collection.upsert(
        ids=[memory_point_id(alice_memory.id), memory_point_id(bob_memory.id)],
        vectors=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        payloads=[build_memory_payload(alice_memory), build_memory_payload(bob_memory)],
    )

    hits = await memory_collection.search(user_id=alice, query_vector=[1.0, 0.0, 0.0, 0.0])

    assert [h.id for h in hits] == [memory_point_id(alice_memory.id)]


async def test_search_bounds_summaries_by_max_page_end(memory_collection) -> None:
    owner, document_id = uuid.uuid4(), uuid.uuid4()
    early = LongTermMemory(
        id=uuid.uuid4(),
        user_id=owner,
        document_id=document_id,
        type=MemoryType.SUMMARY,
        content="early",
        page_start=1,
        page_end=20,
    )
    later = LongTermMemory(
        id=uuid.uuid4(),
        user_id=owner,
        document_id=document_id,
        type=MemoryType.SUMMARY,
        content="later",
        page_start=41,
        page_end=60,
    )
    await memory_collection.upsert(
        ids=[memory_point_id(early.id), memory_point_id(later.id)],
        vectors=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        payloads=[build_memory_payload(early), build_memory_payload(later)],
    )

    hits = await memory_collection.search(
        user_id=owner, query_vector=[1.0, 0.0, 0.0, 0.0], max_page_end=30
    )

    assert [h.id for h in hits] == [memory_point_id(early.id)]


async def test_delete_removes_only_the_owners_point(memory_collection) -> None:
    owner, other = uuid.uuid4(), uuid.uuid4()
    memory = LongTermMemory(id=uuid.uuid4(), user_id=owner, type=MemoryType.FACT, content="x")
    await memory_collection.upsert(
        ids=[memory_point_id(memory.id)],
        vectors=[[1.0, 0.0, 0.0, 0.0]],
        payloads=[build_memory_payload(memory)],
    )

    # A different user's delete call never removes the point.
    await memory_collection.delete(user_id=other, memory_id=memory.id)
    assert await memory_collection.search(user_id=owner, query_vector=[1.0, 0.0, 0.0, 0.0])

    await memory_collection.delete(user_id=owner, memory_id=memory.id)
    assert await memory_collection.search(user_id=owner, query_vector=[1.0, 0.0, 0.0, 0.0]) == []
