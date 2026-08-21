"""Token-budget auto-compaction for the agent's short-term memory (FR-4.1).

A conversation's checkpointed message history grows without bound as turns
accumulate; left unchecked it eventually exceeds the active model's context
window. :class:`CompactionService` tracks the running token count against a
configurable fraction of that window and, once crossed, rewrites the history
into a single concise summary seed — the standard LangGraph pattern of pairing
``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` with a replacement message so the
``add_messages`` reducer clears the checkpointed history in place. This is
strictly the agent's *internal* working memory (what the LLM sees); the
user-visible transcript persists separately via ``ConversationService`` and is
never touched here.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from loguru import logger

from api.llm import context_window_for
from shared.core.config import Settings
from shared.prompt import PromptRegistry

# Anthropic's tokenizer is its official ``count_tokens`` API — a network call,
# offloaded to a worker thread to stay non-blocking. OpenAI counts locally and
# exactly via tiktoken through LangChain's own ``get_num_tokens_from_messages``
# — but that method only implements its per-message overhead formula for
# GPT-family model names, and raises ``NotImplementedError`` for anything else.
# Ollama models (e.g. ``llama3.1``, reached through the same OpenAI-compatible
# client) hit exactly that case, so they're counted via the per-message
# fallback in ``count_tokens`` instead — a coarser, non-crashing estimate, not
# an exact Llama tokenizer count.
NETWORK_TOKENIZER_PROVIDERS = frozenset({"anthropic"})

# ChatOpenAI.get_num_tokens_from_messages()'s own per-message overhead constant
# for GPT-3.5/4/5-family models; reused as a rough proxy in the non-GPT fallback.
_APPROX_MESSAGE_OVERHEAD_TOKENS = 3

_ROLE_LABELS: dict[str, str] = {
    "HumanMessage": "Reader",
    "AIMessage": "Assistant",
    "ToolMessage": "Tool",
    "SystemMessage": "System",
}


@dataclass(slots=True)
class CompactionResult:
    """The outcome of a compaction: reducer ops to rewrite the history, and the
    raw summary text (for promoting salient facts to long-term memory later)."""

    messages: list[AnyMessage]
    summary: str


def _message_text(message: AnyMessage) -> str:
    """Best-effort plain text for one message.

    A tool-calling ``AIMessage`` with no text content (the common shape while a
    turn is still gathering tool results) renders as the tools it called, so
    callers can note that a lookup happened without needing the raw tool-call
    payload.
    """
    text = str(message.text) if hasattr(message, "text") else str(getattr(message, "content", ""))
    if not text and isinstance(message, AIMessage) and message.tool_calls:
        names = ", ".join(call["name"] for call in message.tool_calls)
        text = f"[called {names}]"
    return text


def _flatten_transcript(messages: Sequence[AnyMessage]) -> str:
    """Render a message history as a plain-text transcript for the compaction prompt."""
    lines: list[str] = []
    for message in messages:
        role = _ROLE_LABELS.get(type(message).__name__, type(message).__name__)
        text = _message_text(message)
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


class CompactionService:
    """Counts a session's context tokens and compacts once past the configured threshold."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    async def count_tokens(
        self, messages: Sequence[AnyMessage], *, model: BaseChatModel, provider: str
    ) -> int:
        """Provider-aware token count over the current message history (FR-4.1.1)."""
        if provider in NETWORK_TOKENIZER_PROVIDERS:
            return await asyncio.to_thread(model.get_num_tokens_from_messages, list(messages))
        if provider == "openai":
            return model.get_num_tokens_from_messages(list(messages))
        # Non-GPT models reached via ChatOpenAI (Ollama's OpenAI-compatible
        # client): get_num_tokens_from_messages() raises NotImplementedError for
        # a model name it doesn't recognize as GPT-family. get_num_tokens(text)
        # has no such restriction — it tiktoken-encodes generically regardless
        # of model name — so sum it per message plus the same per-message
        # overhead constant OpenAI's own formula uses, as a non-crashing proxy.
        return sum(
            model.get_num_tokens(_message_text(message)) + _APPROX_MESSAGE_OVERHEAD_TOKENS
            for message in messages
        )

    def should_compact(self, *, token_count: int, provider: str) -> bool:
        """Whether the running count has crossed the provider's threshold (FR-4.1.2)."""
        window = context_window_for(self._settings, provider)
        crossed = token_count >= self._settings.compaction_threshold_ratio * window
        if crossed:
            logger.debug(
                "compaction: threshold crossed (tokens={}, window={}, provider={})",
                token_count,
                window,
                provider,
            )
        return crossed

    async def compact(
        self,
        *,
        messages: Sequence[AnyMessage],
        summarizer: BaseChatModel,
        prompts: PromptRegistry,
    ) -> CompactionResult:
        """Summarize the full history via the cheap tier into a concise seed (FR-4.1.3).

        Returns reducer ops — ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` then a
        seed ``SystemMessage`` — for the caller to apply as a graph state update,
        so the checkpointed history is replaced rather than appended to.
        """
        transcript = _flatten_transcript(messages)
        prompt_obj = prompts.get("compaction", "v1")
        prompt = prompt_obj.render(transcript=transcript)
        response = await summarizer.ainvoke(prompt)
        summary = response.text() if hasattr(response, "text") else str(response.content)
        logger.info("compaction: rewrote {} messages into a summary seed", len(messages))
        seed = SystemMessage(content=f"Summary of the conversation so far:\n{summary}")
        return CompactionResult(
            messages=[RemoveMessage(id=REMOVE_ALL_MESSAGES), seed], summary=summary
        )
