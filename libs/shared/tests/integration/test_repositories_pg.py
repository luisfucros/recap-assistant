"""Integration tests for the repositories against a real Postgres.

Exercises the per-user isolation invariant and the schema constraints (unique
``(user_id, content_sha256)``, FK cascades) that unit tests can only approximate
with compiled SQL. External APIs aren't involved — this is pure DB behavior.
"""

import uuid

import pytest
from sqlalchemy import text

from shared.core.enums import DocumentFormat, DocumentStatus, MessageRole
from shared.models.conversation import Conversation, Message
from shared.models.document import Chunk, Document
from shared.models.user import User
from shared.repositories import (
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    MessageRepository,
    OutboxRepository,
    UserRepository,
)

pytestmark = pytest.mark.integration


async def _make_user(session, email: str = "reader@example.com") -> User:
    user = await UserRepository(session).add(User(email=email))
    await session.commit()
    return user


def _document(user_id: uuid.UUID, sha: str, *, filename: str = "book.pdf") -> Document:
    return Document(
        user_id=user_id,
        filename=filename,
        object_key=f"{user_id}/sha256/{sha}.pdf",
        content_sha256=sha,
        format=DocumentFormat.PDF,
        status=DocumentStatus.PENDING,
        embed_model="test-model",
    )


async def test_document_scoped_reads_isolate_users(db_session) -> None:
    alice = await _make_user(db_session, "alice@example.com")
    bob = await _make_user(db_session, "bob@example.com")

    alice_repo = DocumentRepository(db_session, alice.id)
    doc = await alice_repo.add(_document(alice.id, "a" * 64))
    await db_session.commit()

    # Bob's scoped repository cannot see Alice's document by id.
    bob_repo = DocumentRepository(db_session, bob.id)
    assert await bob_repo.get(doc.id) is None
    assert await alice_repo.get(doc.id) is not None


async def test_unique_content_per_user_enforced_by_db(db_session) -> None:
    from sqlalchemy.exc import IntegrityError

    user = await _make_user(db_session)
    repo = DocumentRepository(db_session, user.id)
    await repo.add(_document(user.id, "d" * 64))
    await db_session.commit()

    # Same (user_id, content_sha256) violates the unique constraint.
    with pytest.raises(IntegrityError):
        await repo.add(_document(user.id, "d" * 64, filename="again.pdf"))
        await db_session.commit()


async def test_same_content_different_users_coexist(db_session) -> None:
    alice = await _make_user(db_session, "alice@example.com")
    bob = await _make_user(db_session, "bob@example.com")
    sha = "e" * 64

    await DocumentRepository(db_session, alice.id).add(_document(alice.id, sha))
    await DocumentRepository(db_session, bob.id).add(_document(bob.id, sha))
    await db_session.commit()

    # Identical content, two independent documents — isolation wins over dedup.
    assert await DocumentRepository(db_session, alice.id).count() == 1
    assert await DocumentRepository(db_session, bob.id).count() == 1


async def test_get_by_content_sha256_is_user_scoped(db_session) -> None:
    alice = await _make_user(db_session, "alice@example.com")
    bob = await _make_user(db_session, "bob@example.com")
    sha = "f" * 64
    await DocumentRepository(db_session, alice.id).add(_document(alice.id, sha))
    await db_session.commit()

    assert await DocumentRepository(db_session, alice.id).get_by_content_sha256(sha) is not None
    assert await DocumentRepository(db_session, bob.id).get_by_content_sha256(sha) is None


async def test_deleting_document_cascades_to_chunks(db_session) -> None:
    user = await _make_user(db_session)
    doc = await DocumentRepository(db_session, user.id).add(_document(user.id, "c" * 64))
    await db_session.commit()

    chunks = ChunkRepository(db_session, user.id)
    await chunks.add_many(
        [
            Chunk(
                document_id=doc.id,
                user_id=user.id,
                ordinal=i,
                page_start=i + 1,
                page_end=i + 1,
                text=f"chunk {i}",
                content_hash=f"h{i}",
            )
            for i in range(3)
        ]
    )
    await db_session.commit()
    assert len(await chunks.list_by_document(doc.id)) == 3

    # FK ondelete=CASCADE removes the chunks with the document.
    await db_session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc.id})
    await db_session.commit()
    assert await chunks.list_by_document(doc.id) == []


async def test_list_by_document_page_range_returns_overlapping_chunks_in_order(db_session) -> None:
    user = await _make_user(db_session)
    doc = await DocumentRepository(db_session, user.id).add(_document(user.id, "d" * 64))
    await db_session.commit()

    chunks = ChunkRepository(db_session, user.id)
    await chunks.add_many(
        [
            Chunk(  # pages 1-10 — before the range
                document_id=doc.id,
                user_id=user.id,
                ordinal=0,
                page_start=1,
                page_end=10,
                text="early",
                content_hash="h0",
            ),
            Chunk(  # pages 18-25 — overlaps [20, 40] at its tail
                document_id=doc.id,
                user_id=user.id,
                ordinal=1,
                page_start=18,
                page_end=25,
                text="overlap-left",
                content_hash="h1",
            ),
            Chunk(  # pages 30-35 — fully inside
                document_id=doc.id,
                user_id=user.id,
                ordinal=2,
                page_start=30,
                page_end=35,
                text="inside",
                content_hash="h2",
            ),
            Chunk(  # pages 45-50 — after the range
                document_id=doc.id,
                user_id=user.id,
                ordinal=3,
                page_start=45,
                page_end=50,
                text="late",
                content_hash="h3",
            ),
            Chunk(  # untagged — excluded (can't be placed in the span)
                document_id=doc.id,
                user_id=user.id,
                ordinal=4,
                page_start=None,
                page_end=None,
                text="untagged",
                content_hash="h4",
            ),
        ]
    )
    await db_session.commit()

    found = await chunks.list_by_document_page_range(doc.id, page_start=20, page_end=40)
    assert [c.text for c in found] == ["overlap-left", "inside"]  # overlap set, ordinal order


async def test_list_by_document_page_range_is_user_scoped(db_session) -> None:
    alice = await _make_user(db_session, "alice2@example.com")
    bob = await _make_user(db_session, "bob2@example.com")
    doc = await DocumentRepository(db_session, alice.id).add(_document(alice.id, "e" * 64))
    await db_session.commit()
    await ChunkRepository(db_session, alice.id).add_many(
        [
            Chunk(
                document_id=doc.id,
                user_id=alice.id,
                ordinal=0,
                page_start=5,
                page_end=6,
                text="alice",
                content_hash="ha",
            )
        ]
    )
    await db_session.commit()

    # Bob's repository never sees Alice's chunks, even by her document id.
    assert (
        await ChunkRepository(db_session, bob.id).list_by_document_page_range(
            doc.id, page_start=1, page_end=10
        )
        == []
    )


async def test_chunk_repo_add_many_rejects_foreign_owner(db_session) -> None:
    user = await _make_user(db_session)
    doc = await DocumentRepository(db_session, user.id).add(_document(user.id, "b" * 64))
    await db_session.commit()

    chunks = ChunkRepository(db_session, user.id)
    stranger_chunk = Chunk(
        document_id=doc.id,
        user_id=uuid.uuid4(),  # not this repo's owner
        ordinal=0,
        text="x",
        content_hash="h",
    )
    with pytest.raises(ValueError, match="does not match"):
        await chunks.add_many([stranger_chunk])


async def test_conversation_scoped_reads_isolate_users(db_session) -> None:
    alice = await _make_user(db_session, "alice3@example.com")
    bob = await _make_user(db_session, "bob3@example.com")

    alice_repo = ConversationRepository(db_session, alice.id)
    conv = await alice_repo.add(Conversation(user_id=alice.id, title="Alice's chat"))
    await db_session.commit()

    # Bob's scoped repository cannot see Alice's conversation by id.
    assert await ConversationRepository(db_session, bob.id).get(conv.id) is None
    assert await alice_repo.get(conv.id) is not None


async def test_messages_scoped_by_user_and_conversation(db_session) -> None:
    alice = await _make_user(db_session, "alice4@example.com")
    bob = await _make_user(db_session, "bob4@example.com")
    conv = await ConversationRepository(db_session, alice.id).add(
        Conversation(user_id=alice.id, title="Alice's chat")
    )
    await db_session.commit()
    await MessageRepository(db_session, alice.id).add(
        Message(
            user_id=alice.id,
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="hello",
        )
    )
    await db_session.commit()

    # Bob's repository never sees Alice's messages, even given her conversation id.
    assert await MessageRepository(db_session, bob.id).list_by_conversation(conv.id) == []
    assert len(await MessageRepository(db_session, alice.id).list_by_conversation(conv.id)) == 1


async def test_deleting_conversation_cascades_to_messages(db_session) -> None:
    user = await _make_user(db_session, "cascade@example.com")
    conv = await ConversationRepository(db_session, user.id).add(
        Conversation(user_id=user.id, title="to delete")
    )
    await db_session.commit()

    messages = MessageRepository(db_session, user.id)
    for role, content in ((MessageRole.USER, "q"), (MessageRole.ASSISTANT, "a")):
        await messages.add(
            Message(user_id=user.id, conversation_id=conv.id, role=role, content=content)
        )
    await db_session.commit()
    assert len(await messages.list_by_conversation(conv.id)) == 2

    # FK ondelete=CASCADE removes the messages with the conversation.
    await db_session.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conv.id})
    await db_session.commit()
    assert await messages.list_by_conversation(conv.id) == []


async def test_outbox_roundtrip(db_session) -> None:
    user = await _make_user(db_session)
    outbox = OutboxRepository(db_session)
    await outbox.add(
        event_type="document.uploaded",
        aggregate_id=uuid.uuid4(),
        payload={"user_id": str(user.id)},
    )
    await db_session.commit()

    pending = await outbox.fetch_unprocessed(limit=10)
    assert len(pending) == 1
    assert await outbox.count_unprocessed() == 1
    await outbox.mark_processed(pending[0].id)
    await db_session.commit()

    assert await outbox.fetch_unprocessed(limit=10) == []
    assert await outbox.count_unprocessed() == 0
