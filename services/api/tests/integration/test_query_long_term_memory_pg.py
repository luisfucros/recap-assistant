"""Integration tests: the ``query_long_term_memory`` tool against real infra.

Drives the actual tool (built via :func:`build_agent_tools`) with a real
:class:`ToolContext` — real repositories, a real :class:`MemoryService` over a
real (throwaway) Qdrant collection — to verify what a unit test (everything
faked) can only approximate:

* **Isolation** — user A's query never returns user B's memories, even with
  identical (deterministic) embeddings, and even when A supplies B's own
  document id — there is no ``user_id`` tool argument to spoof; the owner
  comes only from A's server-side context.
* **Reading position drives memory (FR-18.3)** — the deterministic page-range
  lookup (no ``query``, a ``document_id``) hard-excludes a saved summary that
  reaches past the reader's real, Postgres-held ``current_page``.
"""

import contextlib
import uuid
from types import SimpleNamespace

import pytest
from api.agent.context import ToolContext
from api.agent.tools import build_agent_tools
from api.services.memory_service import MemoryService
from api.services.progress_service import ProgressService

from shared.core.enums import DocumentFormat, DocumentStatus, MemoryType, ReadingStatus
from shared.models.document import Document
from shared.models.reading import ReadingProgress
from shared.models.user import User
from shared.prompt import get_prompt_registry
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    LongTermMemoryRepository,
    ReadingProgressRepository,
    UserRepository,
)
from shared.vectorstore import MemoryVectorStore

pytestmark = pytest.mark.integration

_DIM = 8


class _FakeEmbedder:
    """Deterministic embedder: same vector for every text — these tests assert
    the *filter* (isolation, spoiler bound), not similarity ranking."""

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


async def _make_user(db_sessionmaker, email: str) -> User:
    async with db_sessionmaker() as session:
        user = await UserRepository(session).add(User(email=email))
        await session.commit()
        return user


async def _make_document(
    db_sessionmaker, user_id: uuid.UUID, *, title: str = "The Odyssey"
) -> Document:
    doc = Document(
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/sha256/{uuid.uuid4().hex}.pdf",
        content_sha256=uuid.uuid4().hex,
        format=DocumentFormat.PDF,
        status=DocumentStatus.INDEXED,
        embed_model="test-model",
        title=title,
    )
    async with db_sessionmaker() as session:
        await DocumentRepository(session, user_id).add(doc)
        await session.commit()
    return doc


def _context(session, user_id: uuid.UUID, memory_service: MemoryService) -> ToolContext:
    return ToolContext(
        user_id=user_id,
        documents=DocumentRepository(session, user_id),
        chunks=ChunkRepository(session, user_id),
        progress_repo=ReadingProgressRepository(session, user_id),
        progress_service=ProgressService(),
        retrieval_service=SimpleNamespace(),
        summarizer=SimpleNamespace(),
        prompts=get_prompt_registry(),
        user_spoiler_safe=False,
        session=session,
        events=SimpleNamespace(),
        memories=LongTermMemoryRepository(session, user_id),
        memory_service=memory_service,
        recommendation_service=SimpleNamespace(),
        web_search=lambda: SimpleNamespace(),
        usage=SimpleNamespace(),
        usage_service=SimpleNamespace(),
    )


def _tool(context: ToolContext):
    return next(t for t in build_agent_tools(context) if t.name == "query_long_term_memory")


# --- isolation ---------------------------------------------------------------- #


async def test_semantic_query_never_returns_another_users_memory(
    db_sessionmaker, memory_vector_store
) -> None:
    alice = await _make_user(db_sessionmaker, "alice@example.com")
    bob = await _make_user(db_sessionmaker, "bob@example.com")
    memory_service = MemoryService(embedder=_FakeEmbedder(), vector_store=memory_vector_store)

    async with db_sessionmaker() as session:
        await memory_service.write_memory(
            memories=LongTermMemoryRepository(session, alice.id),
            session=session,
            type=MemoryType.PREFERENCE,
            content="alice likes sci-fi",
        )
    async with db_sessionmaker() as session:
        await memory_service.write_memory(
            memories=LongTermMemoryRepository(session, bob.id),
            session=session,
            type=MemoryType.PREFERENCE,
            content="bob likes romance",
        )

    # Same query, identical (deterministic) embeddings — only the server-side
    # user_id filter (from Alice's own scoped context) can separate them.
    async with db_sessionmaker() as session:
        tool = _tool(_context(session, alice.id, memory_service))
        out = await tool.ainvoke({"query": "preferences"})
    assert "alice likes sci-fi" in out
    assert "bob" not in out


async def test_page_range_lookup_with_another_users_document_id_is_empty(
    db_sessionmaker, memory_vector_store
) -> None:
    alice = await _make_user(db_sessionmaker, "alice@example.com")
    bob = await _make_user(db_sessionmaker, "bob@example.com")
    bob_doc = await _make_document(db_sessionmaker, bob.id, title="Bob's Book")
    memory_service = MemoryService(embedder=_FakeEmbedder(), vector_store=memory_vector_store)

    async with db_sessionmaker() as session:
        await memory_service.write_summary(
            memories=LongTermMemoryRepository(session, bob.id),
            session=session,
            document_id=bob_doc.id,
            page_start=1,
            page_end=10,
            content="bob's recap",
        )

    # Alice supplies Bob's own document id — there is no user_id tool argument
    # to spoof; the owner comes only from Alice's server-side context, so her
    # scoped repository structurally can't surface Bob's summary.
    async with db_sessionmaker() as session:
        tool = _tool(_context(session, alice.id, memory_service))
        out = await tool.ainvoke({"document_id": str(bob_doc.id)})
    assert out == "No matching memories were found."


# --- reading position drives memory (FR-18.3) --------------------------------- #


async def test_page_range_lookup_excludes_summary_past_current_page(
    db_sessionmaker, memory_vector_store
) -> None:
    user = await _make_user(db_sessionmaker, "reader@example.com")
    doc = await _make_document(db_sessionmaker, user.id)
    memory_service = MemoryService(embedder=_FakeEmbedder(), vector_store=memory_vector_store)

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        await memory_service.write_summary(
            memories=memories,
            session=session,
            document_id=doc.id,
            page_start=1,
            page_end=20,
            content="early",
        )
        await memory_service.write_summary(
            memories=memories,
            session=session,
            document_id=doc.id,
            page_start=41,
            page_end=60,
            content="later",
        )
        session.add(
            ReadingProgress(
                user_id=user.id,
                document_id=doc.id,
                current_page=50,
                last_summarized_page=20,
                status=ReadingStatus.READING,
            )
        )
        await session.commit()

    async with db_sessionmaker() as session:
        tool = _tool(_context(session, user.id, memory_service))
        out = await tool.ainvoke({"document_id": str(doc.id)})

    # The later summary (page_end=60) reaches past current_page=50 — hard
    # excluded even with no query at all (no include_unread escape hatch here).
    assert "early" in out
    assert "later" not in out


async def test_semantic_query_bounds_summaries_by_current_page_too(
    db_sessionmaker, memory_vector_store
) -> None:
    user = await _make_user(db_sessionmaker, "reader@example.com")
    doc = await _make_document(db_sessionmaker, user.id)
    memory_service = MemoryService(embedder=_FakeEmbedder(), vector_store=memory_vector_store)

    async with db_sessionmaker() as session:
        memories = LongTermMemoryRepository(session, user.id)
        await memory_service.write_summary(
            memories=memories,
            session=session,
            document_id=doc.id,
            page_start=41,
            page_end=60,
            content="the ending",
        )
        session.add(
            ReadingProgress(
                user_id=user.id,
                document_id=doc.id,
                current_page=30,
                last_summarized_page=0,
                status=ReadingStatus.READING,
            )
        )
        await session.commit()

    async with db_sessionmaker() as session:
        tool = _tool(_context(session, user.id, memory_service))
        out = await tool.ainvoke({"query": "recap", "document_id": str(doc.id)})

    assert out == "No matching memories were found."
