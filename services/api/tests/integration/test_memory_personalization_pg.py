"""Integration test: personal-fact memory survives across sessions (FR-7.9).

Drives two turns through a real Postgres-backed LangGraph checkpointer and a
real `MemoryService`/`MemoryVectorStore` (a fake embedder over a throwaway
Qdrant collection): the first turn's `extract_memory` node saves a salient
personal fact the reader shared, and a *second*, freshly-built conversation
(same user, different `conversation_id` — simulating a brand new session) has
its `load_memories` node surface that saved fact in context, unprompted. The
LLM is faked throughout; the two long-term-memory write/read paths under test
are real.
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
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import Field

from shared.core.enums import MemoryType
from shared.models.user import User
from shared.prompt import get_prompt_registry
from shared.repositories import LongTermMemoryRepository, UserRepository
from shared.vectorstore import MemoryVectorStore

pytestmark = pytest.mark.integration

USER_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
_DIM = 8
_SAVED_FACT = "Prefers fast-paced sci-fi and fantasy novels."


class _FakeEmbedder:
    @property
    def dim(self) -> int:
        return _DIM

    async def embed(self, texts, *, batch_size=None) -> list[list[float]]:
        return [[0.5] * _DIM for _ in texts]


class _CapturingAnswerModel(BaseChatModel):
    """Records every message's text per call (to inspect load_memories' context)."""

    reply: str = "ok"
    seen: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "capturing-personalization"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append([m.text if hasattr(m, "text") else str(m.content) for m in messages])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def bind_tools(self, tools, **kwargs):
        return self


def _models(*, answer_model: BaseChatModel, classification: MemoryClassification) -> AgentModels:
    return AgentModels(
        guardrail_judge=RunnableLambda(
            lambda _p: GuardrailDecision(on_topic=True, safe=True, reason="")
        ),
        spoiler_judge=RunnableLambda(
            lambda _p: SpoilerCheckDecision(spoiler_risk=False, reason="")
        ),
        memory_classifier=RunnableLambda(lambda _p: classification),
        planner=RunnableLambda(
            lambda _p: PlannerDecision(
                complexity=Complexity.SIMPLE, needs_tools=False, tool_plan=[]
            )
        ),
        answer_model=answer_model,
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


class _FakeProgressService:
    """Stands in for ProgressService: load_progress calls reading_list every turn."""

    async def reading_list(self, **kwargs: object) -> dict:
        return {}


def _tool_context(*, session, memory_service: MemoryService) -> ToolContext:
    return ToolContext(
        user_id=USER_ID,
        documents=SimpleNamespace(),
        chunks=SimpleNamespace(),
        progress_repo=SimpleNamespace(),
        progress_service=_FakeProgressService(),
        retrieval_service=SimpleNamespace(),
        summarizer=SimpleNamespace(),
        prompts=get_prompt_registry(),
        user_spoiler_safe=False,
        session=session,
        events=SimpleNamespace(),
        memories=LongTermMemoryRepository(session, USER_ID),
        memory_service=memory_service,
        recommendation_service=SimpleNamespace(),
        web_search=lambda: SimpleNamespace(),
        usage=SimpleNamespace(),
        usage_service=SimpleNamespace(),
    )


async def test_a_personal_fact_shared_in_one_conversation_is_recalled_in_a_new_one(
    test_settings, db_sessionmaker, memory_vector_store
) -> None:
    async with db_sessionmaker() as session:
        await UserRepository(session).add(User(id=USER_ID, email="reader@example.com"))
        await session.commit()

    await setup_checkpointer(test_settings)
    pool = build_pool(test_settings)
    await pool.open()
    try:
        memory_service = MemoryService(embedder=_FakeEmbedder(), vector_store=memory_vector_store)

        # Turn 1, conversation A: the reader shares a durable preference; the
        # classifier flags it salient, so extract_memory saves it for real.
        turn_one_model = _CapturingAnswerModel(reply="Nice to meet you, Ada!")
        turn_one_service = AgentService(
            _models(
                answer_model=turn_one_model,
                classification=MemoryClassification(
                    type=MemoryType.PREFERENCE, salient=True, content=_SAVED_FACT
                ),
            ),
            checkpointer=AsyncPostgresSaver(pool),
        )
        async with db_sessionmaker() as session:
            turn = await turn_one_service.run(
                tool_context=_tool_context(session=session, memory_service=memory_service),
                display_name="Ada",
                message="hi, I'm Ada and I love fast-paced sci-fi and fantasy novels",
                conversation_id=str(uuid.uuid4()),
            )
        assert turn.blocked is False

        # The preference actually landed in Postgres, embedded and indexed.
        async with db_sessionmaker() as session:
            saved = await LongTermMemoryRepository(session, USER_ID).list_recent()
        assert len(saved) == 1
        assert saved[0].type is MemoryType.PREFERENCE
        assert saved[0].content == _SAVED_FACT
        assert saved[0].embedding_id is not None

        # Turn 2: a *different* conversation_id (a brand new session) — nothing
        # new to save this time, so the classifier returns salient=False —
        # but load_memories should still surface the earlier fact unprompted.
        turn_two_model = _CapturingAnswerModel(reply="Welcome back!")
        turn_two_service = AgentService(
            _models(
                answer_model=turn_two_model,
                classification=MemoryClassification(type=MemoryType.FACT, salient=False),
            ),
            checkpointer=AsyncPostgresSaver(pool),
        )
        async with db_sessionmaker() as session:
            turn = await turn_two_service.run(
                tool_context=_tool_context(session=session, memory_service=memory_service),
                display_name="Ada",
                message="hi again",
                conversation_id=str(uuid.uuid4()),
            )
        assert turn.blocked is False

        flattened = "\n".join(turn_two_model.seen[0])
        assert _SAVED_FACT in flattened
        assert "What we remember about Ada" in flattened

        # No duplicate write from the second, non-salient turn.
        async with db_sessionmaker() as session:
            saved_after = await LongTermMemoryRepository(session, USER_ID).list_recent()
        assert len(saved_after) == 1
    finally:
        await pool.close()
