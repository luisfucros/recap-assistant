"""Integration tests for the durable Postgres checkpointer.

Against a real Postgres (the test stack), these prove the load-bearing property
of the durable checkpointer: a turn's state persists under its ``thread_id`` so a
follow-up turn on the same conversation **resumes prior context**, while a
different conversation starts fresh. The LLM is faked (a chat model that reports
how many user messages it was given), so "resumed" is observable without a real
model: turn two sees both messages only if the checkpointer restored turn one.
"""

import uuid
from types import SimpleNamespace
from typing import Any

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
from api.services.compaction_service import CompactionService
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import Field

from shared.core.config import Settings
from shared.core.enums import MemoryType, ReadingStatus
from shared.models.reading import ReadingProgress
from shared.prompt import get_prompt_registry

pytestmark = pytest.mark.integration

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
DOC_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class _CountingModel(BaseChatModel):
    """Answers with the number of human messages it was handed.

    That count is the whole point: it is 1 on a conversation's first turn and 2
    on the next turn only if the checkpointer restored the first turn's state.
    ``seen_messages`` additionally records each call's raw message list (so a
    compaction test can assert what history the model was actually shown);
    ``fake_token_count`` stands in for a real tokenizer.
    """

    seen_messages: list[list] = Field(default_factory=list)
    fake_token_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "counting"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen_messages.append(list(messages))
        seen = sum(1 for m in messages if isinstance(m, HumanMessage))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"seen {seen} user messages"))]
        )

    def bind_tools(self, tools, **kwargs):
        return self

    def get_num_tokens_from_messages(self, messages, tools=None, **kwargs) -> int:
        return self.fake_token_count


class _FakeProgressService:
    async def reading_list(self, *, progress: Any) -> dict:
        row = ReadingProgress(
            user_id=USER_ID,
            document_id=DOC_ID,
            current_page=10,
            last_summarized_page=0,
            status=ReadingStatus.READING,
        )
        return {ReadingStatus.READING: [row]}


class _FakeSummarizerModel:
    """Stands in for ToolContext.summarizer: a canned compaction summary, no ``.text()``."""

    def __init__(self, text: str = "The reader discussed the narrator.") -> None:
        self._text = text

    async def ainvoke(self, prompt: str) -> Any:
        return SimpleNamespace(content=self._text)


class _FakeMemoryService:
    """Stands in for MemoryService: load_memories calls list_memories every turn."""

    async def list_memories(self, **kwargs: Any) -> list[Any]:
        return []


def _context(*, summarizer: Any = None) -> ToolContext:
    return ToolContext(
        user_id=USER_ID,
        documents=SimpleNamespace(),
        chunks=SimpleNamespace(),
        progress_repo=SimpleNamespace(),
        progress_service=_FakeProgressService(),
        retrieval_service=SimpleNamespace(),
        summarizer=summarizer or SimpleNamespace(),
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


def _models(
    *, answer_model: BaseChatModel | None = None, provider: str = "anthropic"
) -> AgentModels:
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
        answer_model=answer_model or _CountingModel(),
        provider=provider,
    )


async def _run(service: AgentService, conversation_id: str, message: str) -> str:
    turn = await service.run(
        tool_context=_context(),
        display_name="Ada",
        message=message,
        conversation_id=conversation_id,
    )
    return turn.answer


async def test_setup_creates_checkpoint_tables(test_settings) -> None:
    await setup_checkpointer(test_settings)
    pool = build_pool(test_settings)
    await pool.open()
    try:
        async with pool.connection() as conn:
            # The pool uses row_factory=dict_row (the saver's requirement), so rows
            # come back keyed by column name rather than positionally.
            result = await conn.execute("SELECT to_regclass('public.checkpoints') AS name")
            assert (await result.fetchone())["name"] == "checkpoints"
    finally:
        await pool.close()


async def test_follow_up_turn_resumes_prior_context(test_settings) -> None:
    await setup_checkpointer(test_settings)
    pool = build_pool(test_settings)
    await pool.open()
    try:
        service = AgentService(_models(), checkpointer=AsyncPostgresSaver(pool))
        thread = f"conv-{uuid.uuid4()}"

        first = await _run(service, thread, "who is the narrator?")
        second = await _run(service, thread, "and where does he sail?")

        # Turn one sees only its own message; turn two sees it too — resumed state.
        assert first == "seen 1 user messages"
        assert second == "seen 2 user messages"
    finally:
        await pool.close()


async def test_distinct_conversations_do_not_share_state(test_settings) -> None:
    await setup_checkpointer(test_settings)
    pool = build_pool(test_settings)
    await pool.open()
    try:
        service = AgentService(_models(), checkpointer=AsyncPostgresSaver(pool))

        await _run(service, f"conv-{uuid.uuid4()}", "first thread question")
        other = await _run(service, f"conv-{uuid.uuid4()}", "second thread question")

        # A different thread_id starts fresh — no leakage across conversations.
        assert other == "seen 1 user messages"
    finally:
        await pool.close()


async def test_compaction_rewrites_the_real_checkpoint_and_resumes_from_the_summary(
    test_settings,
) -> None:
    """FR-4.1.3 against the durable checkpointer: a follow-up turn after
    compaction resolves from the summary seed, not the raw prior turn."""
    await setup_checkpointer(test_settings)
    pool = build_pool(test_settings)
    await pool.open()
    try:
        model = _CountingModel(fake_token_count=1000)
        compaction = CompactionService(
            settings=Settings(
                _env_file=None, llm_context_window_anthropic=1000, compaction_threshold_ratio=0.5
            )
        )
        summarizer = _FakeSummarizerModel("The reader asked about the narrator.")
        service = AgentService(
            _models(answer_model=model),
            checkpointer=AsyncPostgresSaver(pool),
            compaction=compaction,
        )
        thread = f"conv-{uuid.uuid4()}"

        first = await service.run(
            tool_context=_context(summarizer=summarizer),
            display_name="Ada",
            message="who is the narrator?",
            conversation_id=thread,
        )
        second = await service.run(
            tool_context=_context(summarizer=summarizer),
            display_name="Ada",
            message="and where does he sail?",
            conversation_id=thread,
        )

        assert first.answer == "seen 1 user messages"
        # If the real Postgres checkpoint had merely accumulated (no rewrite),
        # turn two would see both human messages; it sees only its own — the
        # checkpoint was actually cleared and reseeded from the summary.
        assert second.answer == "seen 1 user messages"
        second_call = model.seen_messages[-1]
        assert not any(
            isinstance(m, HumanMessage) and m.content == "who is the narrator?" for m in second_call
        )
        assert any(
            isinstance(m, SystemMessage) and "The reader asked about the narrator." in m.content
            for m in second_call
        )
    finally:
        await pool.close()
