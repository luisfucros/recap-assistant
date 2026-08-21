"""Integration test: a streamed chat turn persisted end-to-end against real infra.

Closes a coverage gap noted on the M4 task list: the SSE ``_stream_turn`` ->
``record_turn`` write was previously proven only in parts (SSE frame order
functionally with the DB faked, ``record_turn`` via repository integration
tests) — never as one live-infra flow. This drives
:func:`api.routers.chat._run_turn` (the transport-agnostic generator both
``/chat/stream`` and ``/chat/ws`` wrap) against a real Postgres-backed
checkpointer *and* a real Postgres-backed transcript store, then re-reads the
persisted rows through fresh repositories to prove the write actually landed.
The LLM is faked (deterministic, no network); only the two Postgres-backed
components under test are real.
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
from api.routers.chat import _run_turn
from api.services.agent_service import AgentService
from api.services.conversation_service import ConversationService
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from shared.core.enums import MemoryType, MessageRole, ReadingStatus
from shared.models.reading import ReadingProgress
from shared.models.user import User
from shared.prompt import get_prompt_registry
from shared.repositories import ConversationRepository, MessageRepository, UserRepository

pytestmark = pytest.mark.integration

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
DOC_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
_ANSWER = "Odysseus narrates the Odyssey."


class _AnsweringModel(BaseChatModel):
    """A deterministic, tool-free answer model (no network)."""

    @property
    def _llm_type(self) -> str:
        return "answering"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_ANSWER))])

    def bind_tools(self, tools, **kwargs):
        return self


class _FakeProgressService:
    """Stands in for ProgressService: a canned reading list, nothing recorded."""

    async def reading_list(self, *, progress) -> dict:
        row = ReadingProgress(
            user_id=USER_ID,
            document_id=DOC_ID,
            current_page=10,
            last_summarized_page=0,
            status=ReadingStatus.READING,
        )
        return {ReadingStatus.READING: [row]}


class _FakeMemoryService:
    """Stands in for MemoryService: load_memories calls list_memories every turn."""

    async def list_memories(self, **kwargs: object) -> list[object]:
        return []


def _tool_context() -> ToolContext:
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
        session=SimpleNamespace(),
        events=SimpleNamespace(),
        memories=SimpleNamespace(),
        memory_service=_FakeMemoryService(),
        recommendation_service=SimpleNamespace(),
        web_search=lambda: SimpleNamespace(),
        usage=SimpleNamespace(),
        usage_service=SimpleNamespace(),
    )


def _models() -> AgentModels:
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
                complexity=Complexity.SIMPLE, needs_tools=False, tool_plan=[]
            )
        ),
        answer_model=_AnsweringModel(),
    )


async def test_streamed_turn_persists_to_real_postgres(test_settings, db_sessionmaker) -> None:
    await setup_checkpointer(test_settings)
    pool = build_pool(test_settings)
    await pool.open()
    try:
        agent_service = AgentService(_models(), checkpointer=AsyncPostgresSaver(pool))
        conversation_service = ConversationService()

        async with db_sessionmaker() as session:
            # `conversations`/`messages` FK to `users` — seed the owner first.
            await UserRepository(session).add(User(id=USER_ID, email="reader@example.com"))
            await session.commit()

            conversations = ConversationRepository(session, USER_ID)
            messages = MessageRepository(session, USER_ID)
            conversation = await conversation_service.create(
                conversations=conversations, session=session
            )

            frames = [
                frame
                async for frame in _run_turn(
                    agent_service=agent_service,
                    conversation_service=conversation_service,
                    conversations=conversations,
                    messages=messages,
                    session=session,
                    tool_context=_tool_context(),
                    display_name="Ada",
                    answer_language="English",
                    conversation=conversation,
                    message="who narrates the story?",
                    media_parts=[],
                )
            ]

        # The stream itself: a leading conversation frame, then a terminal done
        # carrying the sanitized answer (no tools needed for this turn).
        assert frames[0]["type"] == "conversation"
        assert frames[-1] == {"type": "done", "answer": _ANSWER, "trace_id": None}

        # Re-read through *fresh* repositories bound to a new session — proves
        # the write actually committed to Postgres, not just that record_turn
        # returned without raising.
        async with db_sessionmaker() as session:
            conversations = ConversationRepository(session, USER_ID)
            messages = MessageRepository(session, USER_ID)
            stored, total = await conversation_service.list_messages(
                conversations=conversations,
                messages=messages,
                conversation_id=conversation.id,
            )
        assert total == 2
        assert [m.role for m in stored] == [MessageRole.USER, MessageRole.ASSISTANT]
        assert stored[0].content == "who narrates the story?"
        assert stored[1].content == _ANSWER
    finally:
        await pool.close()
