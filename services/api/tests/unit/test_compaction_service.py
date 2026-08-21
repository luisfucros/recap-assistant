"""Unit tests for CompactionService (FR-4.1: token-budget auto-compaction).

No network: token counting and the summarizer are faked at the boundary. What's
under test is real: provider-aware counting (offloading only Anthropic's
network-bound tokenizer to a thread), the threshold arithmetic (ratio x the
configured per-provider context window), and the message-rewrite ops a
compaction produces (a full-history clear paired with a single seed message).
"""

from types import SimpleNamespace

import pytest
from api.services.compaction_service import CompactionService
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from shared.core.config import Settings
from shared.prompt import get_prompt_registry

pytestmark = pytest.mark.unit


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


class _FakeModel:
    """Stands in for a raw BaseChatModel's token counter."""

    def __init__(self, count: int, *, per_message_count: int = 1) -> None:
        self._count = count
        self._per_message_count = per_message_count
        self.calls: list[list] = []
        self.text_calls: list[str] = []

    def get_num_tokens_from_messages(self, messages: list) -> int:
        self.calls.append(list(messages))
        return self._count

    def get_num_tokens(self, text: str) -> int:
        self.text_calls.append(text)
        return self._per_message_count


class _FakeSummarizer:
    """Fake cheap-tier chat model: echoes a fixed summary, records the prompt."""

    def __init__(self, text: str = "A concise summary.") -> None:
        self._text = text
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self._text, text=lambda: self._text)


# --- count_tokens -------------------------------------------------------------- #


async def test_count_tokens_calls_openai_model_directly() -> None:
    service = CompactionService(settings=_settings())
    model = _FakeModel(123)
    messages = [HumanMessage(content="hi")]
    count = await service.count_tokens(messages, model=model, provider="openai")
    assert count == 123
    assert model.calls == [messages]


async def test_count_tokens_offloads_anthropic_to_a_thread() -> None:
    # Anthropic's tokenizer is a network call; it must not block the event loop,
    # but the result still comes through unchanged.
    service = CompactionService(settings=_settings())
    model = _FakeModel(456)
    count = await service.count_tokens(
        [HumanMessage(content="hi")], model=model, provider="anthropic"
    )
    assert count == 456


async def test_count_tokens_ollama_uses_a_per_message_fallback() -> None:
    # ChatOpenAI.get_num_tokens_from_messages() only implements its per-message
    # overhead formula for GPT-family model names and raises NotImplementedError
    # for anything else — including an Ollama model like "llama3.1", reached
    # through the same OpenAI-compatible client. count_tokens must therefore
    # never call it for "ollama", falling back to the generic get_num_tokens().
    service = CompactionService(settings=_settings())
    model = _FakeModel(999, per_message_count=5)
    messages = [HumanMessage(content="hi"), HumanMessage(content="there")]

    count = await service.count_tokens(messages, model=model, provider="ollama")

    assert count == 2 * (5 + 3)  # per-message count + the per-message overhead proxy
    assert model.calls == []
    assert model.text_calls == ["hi", "there"]


async def test_count_tokens_ollama_renders_tool_calls_with_no_text() -> None:
    service = CompactionService(settings=_settings())
    model = _FakeModel(999, per_message_count=1)
    messages = [
        AIMessage(content="", tool_calls=[{"name": "get_reading_progress", "args": {}, "id": "1"}])
    ]

    await service.count_tokens(messages, model=model, provider="ollama")

    assert model.text_calls == ["[called get_reading_progress]"]


# --- should_compact (FR-4.1.2) ------------------------------------------------- #


def test_should_compact_below_threshold_is_false() -> None:
    settings = _settings(llm_context_window_anthropic=1000, compaction_threshold_ratio=0.75)
    service = CompactionService(settings=settings)
    assert service.should_compact(token_count=749, provider="anthropic") is False


def test_should_compact_at_threshold_is_true() -> None:
    settings = _settings(llm_context_window_anthropic=1000, compaction_threshold_ratio=0.75)
    service = CompactionService(settings=settings)
    assert service.should_compact(token_count=750, provider="anthropic") is True


def test_should_compact_uses_the_active_providers_own_window() -> None:
    settings = _settings(
        llm_context_window_anthropic=1000,
        llm_context_window_openai=2000,
        compaction_threshold_ratio=0.5,
    )
    service = CompactionService(settings=settings)
    # 600 tokens is past Anthropic's (smaller) window's threshold...
    assert service.should_compact(token_count=600, provider="anthropic") is True
    # ...but not past OpenAI's (larger) window's threshold.
    assert service.should_compact(token_count=600, provider="openai") is False


# --- compact (FR-4.1.3) -------------------------------------------------------- #


async def test_compact_rewrites_history_to_a_single_seed_message() -> None:
    summarizer = _FakeSummarizer("The reader is partway through The Odyssey.")
    service = CompactionService(settings=_settings())
    messages = [
        HumanMessage(content="what's my progress?"),
        AIMessage(content="", tool_calls=[{"name": "get_reading_progress", "args": {}, "id": "1"}]),
        ToolMessage(content="page 50 of 100", tool_call_id="1"),
        AIMessage(content="You're on page 50."),
    ]

    result = await service.compact(
        messages=messages, summarizer=summarizer, prompts=get_prompt_registry()
    )

    assert result.summary == "The reader is partway through The Odyssey."
    assert len(result.messages) == 2
    remove, seed = result.messages
    assert isinstance(remove, RemoveMessage) and remove.id == REMOVE_ALL_MESSAGES
    assert isinstance(seed, SystemMessage)
    assert "The reader is partway through The Odyssey." in seed.content


async def test_compact_renders_a_flattened_transcript_into_the_prompt() -> None:
    summarizer = _FakeSummarizer()
    service = CompactionService(settings=_settings())
    messages = [
        HumanMessage(content="what's my progress?"),
        AIMessage(content="", tool_calls=[{"name": "get_reading_progress", "args": {}, "id": "1"}]),
        ToolMessage(content="page 50 of 100", tool_call_id="1"),
        AIMessage(content="You're on page 50."),
    ]

    await service.compact(messages=messages, summarizer=summarizer, prompts=get_prompt_registry())

    prompt = summarizer.prompts[0]
    assert "Reader: what's my progress?" in prompt
    assert "[called get_reading_progress]" in prompt
    assert "Tool: page 50 of 100" in prompt
    assert "Assistant: You're on page 50." in prompt
