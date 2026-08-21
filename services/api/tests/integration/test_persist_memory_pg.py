"""Integration test: persist_memory's page-range confirmation (FR-4.6) end-to-end.

Drives a full turn that touches a document with unsaved read pages through a
real Postgres-backed LangGraph checkpointer, approves the resulting
page-range-confirmation interrupt, and verifies against real infra that both
writes actually landed: a summary `LongTermMemory` row in Postgres (via the
real `MemoryService`/`MemoryVectorStore`, a fake embedder over a throwaway
Qdrant collection) and the advanced `last_summarized_page` on the real
`ReadingProgress` row. The LLM (tool-calling + recap) is faked; the two write
paths under test are real.
"""

import uuid
from types import SimpleNamespace

import pytest
from api.agent.context import ToolContext
from api.agent.graph import AgentModels
from api.agent.schemas import (
    Complexity,
    GuardrailDecision,
    MemoryClassification,
    PlannerDecision,
    SpoilerCheckDecision,
)
from api.checkpointer import build_pool, setup_checkpointer
from api.services.agent_service import AgentService
from api.services.memory_service import MemoryService
from api.services.progress_service import ProgressService
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import Field

from shared.core.enums import DocumentFormat, MemoryType, ReadingStatus
from shared.models.document import Chunk, Document
from shared.models.reading import ReadingProgress
from shared.models.user import User
from shared.prompt import get_prompt_registry
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    LongTermMemoryRepository,
    ReadingEventRepository,
    ReadingProgressRepository,
    UserRepository,
)
from shared.vectorstore import MemoryVectorStore

pytestmark = pytest.mark.integration

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_DIM = 8
_RECAP_TEXT = "Odysseus reaches the island of the Phaeacians."


class _FakeEmbedder:
    @property
    def dim(self) -> int:
        return _DIM

    async def embed(self, texts, *, batch_size=None) -> list[list[float]]:
        return [[0.5] * _DIM for _ in texts]


class _FakeSummarizerModel:
    """Stands in for ToolContext.summarizer: a canned recap, no network."""

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=_RECAP_TEXT)


class _ScriptedToolModel(BaseChatModel):
    """Calls get_reading_progress naming the document, then answers directly."""

    responses: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-persist-memory"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


def _models(document_id: uuid.UUID) -> AgentModels:
    model = _ScriptedToolModel()
    model.responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_reading_progress",
                    "args": {"document_id": str(document_id)},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="Here's your progress."),
    ]
    return AgentModels(
        guardrail_judge=RunnableLambda(
            lambda _p: GuardrailDecision(on_topic=True, safe=True, reason="")
        ),
        spoiler_judge=RunnableLambda(
            lambda _p: SpoilerCheckDecision(spoiler_risk=False, reason="")
        ),
        memory_classifier=RunnableLambda(
            lambda _p: MemoryClassification(type=MemoryType.FACT, salient=False)
        ),
        planner=RunnableLambda(
            lambda _p: PlannerDecision(
                complexity=Complexity.SIMPLE, needs_tools=True, tool_plan=["get_reading_progress"]
            )
        ),
        answer_model=model,
    )


@pytest.fixture
async def memory_vector_store(qdrant_client):
    collection = f"test_long_term_memory_{uuid.uuid4().hex}"
    store = MemoryVectorStore(qdrant_client, collection=collection, dim=_DIM)
    try:
        yield store
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            await qdrant_client.delete_collection(collection)


async def _seed(db_sessionmaker) -> tuple[uuid.UUID, uuid.UUID]:
    """A user with a document, 50 pages of chunks, and progress at page 50/10."""
    async with db_sessionmaker() as session:
        await UserRepository(session).add(User(id=USER_ID, email="reader@example.com"))
        await session.commit()

    doc = Document(
        id=uuid.uuid4(),
        user_id=USER_ID,
        filename="book.pdf",
        object_key=f"{USER_ID}/sha256/book.pdf",
        content_sha256=uuid.uuid4().hex,
        format=DocumentFormat.PDF,
        embed_model="test-model",
        title="The Odyssey",
        page_count=50,
    )
    async with db_sessionmaker() as session:
        await DocumentRepository(session, USER_ID).add(doc)
        await session.commit()

    chunks = [
        Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            user_id=USER_ID,
            ordinal=page,
            page_start=page,
            page_end=page,
            text=f"Page {page} of the Odyssey.",
            content_hash=uuid.uuid4().hex,
            vector_id=None,
        )
        for page in range(1, 51)
    ]
    async with db_sessionmaker() as session:
        await ChunkRepository(session, USER_ID).add_many(chunks)
        session.add(
            ReadingProgress(
                user_id=USER_ID,
                document_id=doc.id,
                current_page=50,
                last_summarized_page=10,
                status=ReadingStatus.READING,
            )
        )
        await session.commit()

    return doc.id, USER_ID


async def test_persist_memory_confirms_and_saves_against_real_postgres_and_qdrant(
    test_settings, db_sessionmaker, memory_vector_store
) -> None:
    document_id, user_id = await _seed(db_sessionmaker)

    await setup_checkpointer(test_settings)
    pool = build_pool(test_settings)
    await pool.open()
    try:
        agent_service = AgentService(_models(document_id), checkpointer=AsyncPostgresSaver(pool))
        memory_service = MemoryService(embedder=_FakeEmbedder(), vector_store=memory_vector_store)
        progress_service = ProgressService()
        conversation_id = str(uuid.uuid4())

        async def _context(session) -> ToolContext:
            return ToolContext(
                user_id=user_id,
                documents=DocumentRepository(session, user_id),
                chunks=ChunkRepository(session, user_id),
                progress_repo=ReadingProgressRepository(session, user_id),
                progress_service=progress_service,
                retrieval_service=SimpleNamespace(),
                summarizer=_FakeSummarizerModel(),
                prompts=get_prompt_registry(),
                user_spoiler_safe=False,
                session=session,
                events=ReadingEventRepository(session, user_id),
                memories=LongTermMemoryRepository(session, user_id),
                memory_service=memory_service,
                recommendation_service=SimpleNamespace(),
                web_search=lambda: SimpleNamespace(),
                usage=SimpleNamespace(),
                usage_service=SimpleNamespace(),
            )

        async with db_sessionmaker() as session:
            paused = await agent_service.run(
                tool_context=await _context(session),
                display_name="Ada",
                message="what's my progress?",
                conversation_id=conversation_id,
            )
        assert paused.interrupted is True
        assert paused.interrupt["kind"] == "page_range_confirm"
        assert paused.interrupt["proposal"] == {
            "page_start": 11,
            "page_end": 50,
            "proposal_reason": "pages read since the last saved summary",
        }

        async with db_sessionmaker() as session:
            turn = await agent_service.resume(
                tool_context=await _context(session),
                display_name="Ada",
                conversation_id=conversation_id,
                decision={"decision": "approve"},
            )
        assert turn.interrupted is False
        assert turn.answer == "Here's your progress."

        # The summary memory actually landed in Postgres, correctly page-keyed.
        async with db_sessionmaker() as session:
            memories = LongTermMemoryRepository(session, user_id)
            saved = await memories.list_by_document(document_id)
        assert len(saved) == 1
        assert saved[0].type is MemoryType.SUMMARY
        assert (saved[0].page_start, saved[0].page_end) == (11, 50)
        assert saved[0].content == _RECAP_TEXT
        assert saved[0].embedding_id is not None

        # The recap memory is retrievable via the real vector store too.
        async with db_sessionmaker() as session:
            found = await memory_service.retrieve(
                memories=LongTermMemoryRepository(session, user_id), query="Phaeacians"
            )
        assert [m.content for m in found] == [_RECAP_TEXT]

        # last_summarized_page actually advanced in Postgres.
        async with db_sessionmaker() as session:
            row = await ReadingProgressRepository(session, user_id).get_by_document(document_id)
        assert row.last_summarized_page == 50
    finally:
        await pool.close()
