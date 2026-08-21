"""The LangGraph agent's shared state.

One ``AgentState`` instance flows through every node of a turn. It extends the
conversation transcript (``messages``, reduced by :func:`add_messages` so nodes
append while the checkpointer preserves history across turns) with the per-turn
control fields the nodes set and read: the deterministic input-screen result, the
guardrail verdict, the planner's routing decision, and the final sanitized answer.

Only ``messages`` uses a reducer; every other field is last-write-wins, so a node
returning a partial dict overwrites just the keys it names. Fields are optional
(``total=False``) because each node fills in only its own slice.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from api.services.multimodal_service import MediaPart


class AgentState(TypedDict, total=False):
    """State threaded through the agent graph for a single turn.

    Attributes:
        messages: The running transcript; appended to via the ``add_messages``
            reducer so history accumulates and same-id messages are replaced
            (``normalize_input`` overwrites the raw human message with its
            redacted form in place).
        display_name: The reader's name, injected into the ``generate`` prompt.
        spoiler_safe: The user's default spoiler-safe setting for this turn.
        injection_detected: Set by ``normalize_input`` when the deterministic
            screen flags a prompt-injection attempt (forces a guardrail block).
        on_topic: The input guardrail's topical verdict.
        safe: The input guardrail's safety verdict.
        block_reason: A user-facing reason when the turn is blocked; empty/absent
            when the turn proceeds. Its presence is what marks a turn as blocked.
        needs_tools: The planner's decision on whether tools are required, which
            gates whether ``generate`` binds the read tools.
        complexity: The planner's complexity label (``simple``/``standard``/
            ``complex``); retained for tracing and future model-tier selection.
        media_parts: The turn's non-text attachments (audio/image), consumed by
            ``normalize_input`` — transcribed/described to text and folded into the
            human message before any downstream node sees them (FR-19). Input-only.
        answer: The final, XSS-sanitized answer text set by ``guardrail_out``.
        active_document_id: The document (if any) this turn's tool calls named,
            as a string id — reset to ``None`` at the top of every turn by
            ``normalize_input`` and (re)set by ``run_tools`` from the tool calls
            it just executed, so it never leaks a stale value from an earlier
            turn. ``persist_memory`` (FR-4.6) reads it to know which document's
            reading position to check.
        token_count: The running token count over ``messages``, recomputed and
            set by ``compact`` at the end of every turn (FR-4.1.1). Informational
            (the count that actually gates compaction is recomputed fresh each
            turn, not read back from here) — kept for observability/tracing.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    display_name: str
    spoiler_safe: bool
    injection_detected: bool
    media_parts: list[MediaPart]
    on_topic: bool
    safe: bool
    block_reason: str
    needs_tools: bool
    complexity: str
    answer: str
    active_document_id: str | None
    token_count: int
