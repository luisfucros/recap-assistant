"""Ordered stream events emitted by :class:`~api.services.agent_service.AgentService`.

A turn is surfaced to the caller as a sequence of these events, in a guaranteed
order: any ``ToolCallEvent``/``ToolResultEvent`` pairs (the tool-step timeline)
come first, then the answer's ``TokenEvent``s, then a single terminal
``DoneEvent`` — unless a guardrail stops the turn, in which case the stream is a
single ``BlockedEvent`` and nothing else. ``NodeStatusEvent`` is a non-terminal
exception to all of the above: it can appear anywhere before the terminal event
(live progress, one per graph node reached), and a caller that doesn't care
about it can simply ignore the type.

These are transport-agnostic: the SSE/WebSocket routes (a later slice) serialize
them with :meth:`AgentEvent.as_dict`, and the non-streaming ``run`` path reuses
the same shapes so both surfaces speak one vocabulary.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallEvent:
    """A tool the agent decided to call, with the arguments it chose.

    Emitted the moment the model requests the call, before the tool runs, so a UI
    can show the step as *in progress*. ``args`` holds only the LLM-supplied
    semantic arguments — the owner ``user_id`` is injected server-side and never
    appears here.
    """

    name: str
    args: dict[str, Any]
    id: str
    type: str = field(default="tool_call", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for SSE/WebSocket transport."""
        return {"type": self.type, "name": self.name, "args": self.args, "id": self.id}


@dataclass(slots=True)
class ToolResultEvent:
    """The observation a tool returned, paired to its :class:`ToolCallEvent` by ``id``."""

    name: str
    content: str
    id: str
    type: str = field(default="tool_result", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for SSE/WebSocket transport."""
        return {"type": self.type, "name": self.name, "content": self.content, "id": self.id}


@dataclass(slots=True)
class TokenEvent:
    """One streamed chunk of the final answer's natural-language text."""

    text: str
    type: str = field(default="token", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for SSE/WebSocket transport."""
        return {"type": self.type, "text": self.text}


@dataclass(slots=True)
class DoneEvent:
    """Terminal event carrying the full, sanitized answer.

    The streamed ``TokenEvent`` text is raw model output; ``answer`` here is the
    XSS-sanitized final form (from the output guardrail) and is authoritative — a
    client should treat it as the canonical answer, replacing the streamed text.
    ``trace_id`` correlates the turn to its trace when tracing is enabled (``None``
    when Langfuse is not configured), so a client can deep-link to the trace.
    """

    answer: str
    trace_id: str | None = None
    type: str = field(default="done", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for SSE/WebSocket transport."""
        return {"type": self.type, "answer": self.answer, "trace_id": self.trace_id}


@dataclass(slots=True)
class BlockedEvent:
    """Terminal event when an input guardrail stops the turn (off-topic/unsafe).

    Carries a polite, user-facing reason; no tokens or tool events accompany it.
    """

    reason: str
    type: str = field(default="blocked", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for SSE/WebSocket transport."""
        return {"type": self.type, "reason": self.reason}


@dataclass(slots=True)
class NodeStatusEvent:
    """A node has started running, with a short, human-readable description.

    Purely advisory progress: emitted live (via LangGraph's custom stream mode)
    the moment a node begins, well before its own tool-call/token/done events —
    if any — would appear, so a UI has something to show during the gap before
    the first token (e.g. while the planner or a gated tool runs). Never
    terminal; safe to ignore.
    """

    node: str
    description: str
    type: str = field(default="node_status", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for SSE/WebSocket transport."""
        return {"type": self.type, "node": self.node, "description": self.description}


@dataclass(slots=True)
class InterruptEvent:
    """Terminal-for-this-turn event: the turn is paused, awaiting the user (HITL).

    Emitted instead of a ``done``/``blocked`` event whenever a node calls
    LangGraph's ``interrupt()`` — a gated tool call awaiting approval
    (``kind="tool_approval"``), a page-range confirmation before a summary
    memory is saved (``kind="page_range_confirm"``, FR-4.6), or a missing-
    position question (``kind="ask_pages_read"``, FR-4.7). ``payload`` is
    exactly the dict passed to ``interrupt()`` at the pause site — its shape
    varies by ``kind``, always including ``kind`` itself. The turn is
    *paused*, not finished — the checkpointer holds the state under this
    conversation's thread id, so ``POST /chat/{conversation_id}/resume`` with
    an answer/decision continues the same turn from exactly here.
    """

    payload: dict[str, Any]
    type: str = field(default="interrupt", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for SSE/WebSocket transport (payload keys at the top level)."""
        return {"type": self.type, **self.payload}


AgentEvent = (
    ToolCallEvent
    | ToolResultEvent
    | TokenEvent
    | DoneEvent
    | BlockedEvent
    | NodeStatusEvent
    | InterruptEvent
)
"""Any event the agent stream can yield."""
