"""The agent service: run or stream one reading-assistant turn.

This is the boundary the ``/chat`` routes (a later slice) delegate to. It owns the
compiled LangGraph agent and translates a turn into the caller's two shapes:

* :meth:`AgentService.run` — a single non-streamed :class:`AgentTurn` (answer plus
  the tool steps taken, or a guardrail block).
* :meth:`AgentService.stream` — the same turn as an ordered async stream of
  :mod:`~api.agent.events`: tool-call/tool-result pairs first, then answer tokens,
  then a terminal ``done`` — or a lone ``blocked`` when a guardrail stops the turn.

The LLM entry points (:class:`AgentModels`) are built once from settings (heavy
clients loaded at startup, per the app's one-time-load rule) and reused across
turns; the per-turn, user-scoped :class:`ToolContext` is passed in by the caller,
so the owner ``user_id`` is never taken from the model. The checkpointer keys on
``conversation_id`` so a follow-up turn resumes prior context.
"""

import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt
from loguru import logger

from api.agent.context import ToolContext
from api.agent.events import (
    AgentEvent,
    BlockedEvent,
    DoneEvent,
    InterruptEvent,
    NodeStatusEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from api.agent.graph import (
    GENERATE,
    GUARDRAIL_IN,
    GUARDRAIL_OUT,
    TOOLS,
    AgentModels,
    MultimodalNotConfiguredError,
    build_agent_graph,
)
from api.agent.schemas import (
    GuardrailDecision,
    MemoryClassification,
    PlannerDecision,
    SpoilerCheckDecision,
)
from api.llm import build_answer_models, build_resilient_structured_model, model_id_for
from api.services.compaction_service import CompactionService
from api.services.multimodal_service import MediaPart, MultimodalNormalizer
from api.services.scratchpad_service import ScratchpadService
from shared.core.config import Settings
from shared.core.errors import ConflictError
from shared.observability.tracing import NoOpTracer, Tracer


class NoPendingInterruptError(ConflictError):
    """``AgentService.resume`` was called on a conversation with nothing paused.

    Resuming a thread with no pending ``interrupt()`` would otherwise silently
    run the graph from an empty state rather than fail cleanly.
    """

    code = "NO_PENDING_INTERRUPT"
    message = "This conversation has no gated tool call awaiting approval."


@dataclass(slots=True)
class ToolStep:
    """One tool the agent called this turn and the observation it returned."""

    name: str
    args: dict
    result: str


@dataclass(slots=True)
class AgentTurn:
    """The non-streamed outcome of a turn.

    ``blocked`` distinguishes a guardrail refusal (``answer`` holds the polite
    reason, ``tool_steps`` is empty) from a normal answer. ``interrupted`` marks
    a turn *paused* (not finished) on a LangGraph ``interrupt()`` awaiting the
    user's answer/decision (HITL) — ``interrupt`` then carries the pause's exact
    payload (always including a ``kind``: ``tool_approval``,
    ``page_range_confirm``, ``ask_pages_read``, or ``spoiler_warning``); resume it
    via :meth:`AgentService.resume`. ``trace_id`` correlates the turn to its trace
    when tracing is enabled (``None`` otherwise).
    """

    answer: str
    blocked: bool = False
    tool_steps: list[ToolStep] = field(default_factory=list)
    trace_id: str | None = None
    interrupted: bool = False
    interrupt: dict[str, Any] | None = None


def build_agent_models(settings: Settings) -> AgentModels:
    """Build the agent's LLM entry points from settings (once, at startup).

    The guardrail, planner, spoiler judge, and memory classifier run on the cheap tier with
    structured output *and* retry/fallback resilience. The answer model is the
    default-tier, tool-bindable, streamable chat model; its raw per-provider
    fallbacks and retry budget are passed alongside it so the graph can compose
    ``with_retry``/``with_fallbacks`` over the *tool-bound* model (binding must
    precede resilience — see :func:`~api.llm.build_answer_models`).
    """
    answer_model, answer_fallbacks = build_answer_models(settings, tier="default")
    return AgentModels(
        guardrail_judge=build_resilient_structured_model(settings, GuardrailDecision, tier="cheap"),
        planner=build_resilient_structured_model(settings, PlannerDecision, tier="cheap"),
        spoiler_judge=build_resilient_structured_model(
            settings, SpoilerCheckDecision, tier="cheap"
        ),
        memory_classifier=build_resilient_structured_model(
            settings, MemoryClassification, tier="cheap"
        ),
        answer_model=answer_model,
        answer_fallbacks=answer_fallbacks,
        max_retries=settings.llm_max_retries,
        provider=settings.llm_provider,
        cheap_model=model_id_for(settings, "cheap"),
        default_model=model_id_for(settings, "default"),
    )


class AgentService:
    """Runs and streams reading-assistant turns over the compiled agent graph."""

    def __init__(
        self,
        models: AgentModels,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        tracer: Tracer | None = None,
        scratchpad: ScratchpadService | None = None,
        multimodal: Callable[[], MultimodalNormalizer] | None = None,
        compaction: CompactionService | None = None,
    ) -> None:
        self._models = models
        self._checkpointer = checkpointer
        self._tracer = tracer or NoOpTracer()
        self._scratchpad = scratchpad
        # A *factory*, not a built normalizer: the transcriber/vision providers need
        # API keys, so we defer building them until a turn actually carries media —
        # a text-only chat must never fail because a media provider key is unset.
        self._multimodal = multimodal
        # Optional token-budget auto-compaction (FR-4.1); None disables it (history
        # grows unbounded, the pre-M5 behavior).
        self._compaction = compaction

    def _graph(
        self,
        tool_context: ToolContext,
        display_name: str,
        answer_language: str,
        conversation_id: str,
        turn_id: str,
        normalizer: MultimodalNormalizer | None,
    ) -> CompiledStateGraph:
        """Compile a graph bound to this turn's user-scoped context."""
        return build_agent_graph(
            tool_context=tool_context,
            models=self._models,
            display_name=display_name,
            answer_language=answer_language,
            tracer=self._tracer,
            scratchpad=self._scratchpad,
            normalizer=normalizer,
            compaction=self._compaction,
            conversation_id=conversation_id,
            turn_id=turn_id,
            checkpointer=self._checkpointer,
        )

    def _normalizer_for(self, media_parts: list[MediaPart] | None) -> MultimodalNormalizer | None:
        """Build the multimodal normalizer only when the turn actually has media.

        Deferring the build keeps text-only turns independent of the transcription/
        vision provider keys (building those providers can raise if a key is unset).
        """
        if not media_parts:
            return None
        if self._multimodal is None:
            raise MultimodalNotConfiguredError(
                "chat turn carried media parts but no multimodal normalizer is configured"
            )
        return self._multimodal()

    @staticmethod
    def _inputs(
        message: str,
        tool_context: ToolContext,
        display_name: str,
        media_parts: list[MediaPart] | None,
    ) -> dict:
        """Build the graph's initial state for a turn.

        The human message carries an explicit id so ``normalize_input`` can
        overwrite it in place with its redacted form (rather than appending a
        duplicate) via the ``add_messages`` reducer. Any media parts ride alongside
        it and are folded into that message's text by ``normalize_input`` (FR-19).
        """
        return {
            "messages": [HumanMessage(content=message, id=str(uuid.uuid4()))],
            "display_name": display_name,
            "spoiler_safe": tool_context.user_spoiler_safe,
            "media_parts": media_parts or [],
        }

    @staticmethod
    def _config(conversation_id: str) -> dict:
        """The LangGraph config that scopes checkpointed state to a conversation."""
        return {"configurable": {"thread_id": conversation_id}}

    async def run(
        self,
        *,
        tool_context: ToolContext,
        display_name: str,
        message: str,
        conversation_id: str,
        answer_language: str = "English",
        media_parts: list[MediaPart] | None = None,
    ) -> AgentTurn:
        """Run a turn to completion and return its answer (or guardrail block)."""
        turn_id = str(uuid.uuid4())
        normalizer = self._normalizer_for(media_parts)
        graph = self._graph(
            tool_context, display_name, answer_language, conversation_id, turn_id, normalizer
        )
        started = time.monotonic()
        with (
            logger.contextualize(
                turn_id=turn_id, conversation_id=conversation_id, user_id=str(tool_context.user_id)
            ),
            self._tracer.span("agent.turn", conversation_id=conversation_id) as span,
        ):
            logger.info("agent.turn.start", message_length=len(message))
            # Captured inside the span so it reflects this turn's trace (None when
            # tracing is disabled); surfaced on the turn for client-side deep-links.
            trace_id = self._tracer.current_trace_id()
            state = await graph.ainvoke(
                self._inputs(message, tool_context, display_name, media_parts),
                self._config(conversation_id),
                durability="sync",
            )
            interrupt_payload = _interrupt_payload(state)
            blocked = bool(state.get("block_reason"))
            span.update(
                output={
                    "blocked": blocked,
                    "interrupted": interrupt_payload is not None,
                    "complexity": state.get("complexity"),
                }
            )
            outcome = (
                "interrupted"
                if interrupt_payload is not None
                else "blocked"
                if blocked
                else "answered"
            )
            logger.info(
                "agent.turn.finish",
                outcome=outcome,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        if interrupt_payload is not None:
            return AgentTurn(
                answer="", interrupted=True, interrupt=interrupt_payload, trace_id=trace_id
            )
        if blocked:
            return AgentTurn(answer=state["block_reason"], blocked=True, trace_id=trace_id)
        return AgentTurn(
            answer=state.get("answer", ""),
            tool_steps=_tool_steps(state["messages"]),
            trace_id=trace_id,
        )

    async def resume(
        self,
        *,
        tool_context: ToolContext,
        display_name: str,
        conversation_id: str,
        decision: dict[str, Any],
        answer_language: str = "English",
    ) -> AgentTurn:
        """Resume a turn paused on a gated tool-call approval (HITL).

        ``decision`` is matched to the paused ``interrupt()`` call —
        ``{"decision": "approve"}``, ``{"decision": "deny"}``, or
        ``{"decision": "edit", "args": {...}}`` — and continues the *same*
        checkpointed turn (keyed by ``conversation_id``) from exactly where it
        paused, rather than starting a new one. A turn with more than one gated
        call left to resolve pauses again; the returned :class:`AgentTurn` is
        ``interrupted`` once more in that case.

        Raises:
            NoPendingInterruptError: The conversation's checkpoint has nothing
                paused (already resolved, or never interrupted) — resuming it
                would otherwise silently re-run the graph from an empty state.
        """
        turn_id = str(uuid.uuid4())
        graph = self._graph(
            tool_context, display_name, answer_language, conversation_id, turn_id, None
        )
        config = self._config(conversation_id)
        snapshot = await graph.aget_state(config)
        if not any(task.interrupts for task in snapshot.tasks):
            raise NoPendingInterruptError()
        started = time.monotonic()
        with (
            logger.contextualize(
                turn_id=turn_id, conversation_id=conversation_id, user_id=str(tool_context.user_id)
            ),
            self._tracer.span("agent.turn.resume", conversation_id=conversation_id) as span,
        ):
            logger.info("agent.turn.resume.start", decision=decision.get("decision"))
            trace_id = self._tracer.current_trace_id()
            state = await graph.ainvoke(Command(resume=decision), config, durability="sync")
            interrupt_payload = _interrupt_payload(state)
            blocked = bool(state.get("block_reason"))
            span.update(output={"blocked": blocked, "interrupted": interrupt_payload is not None})
            outcome = (
                "interrupted"
                if interrupt_payload is not None
                else "blocked"
                if blocked
                else "answered"
            )
            logger.info(
                "agent.turn.resume.finish",
                outcome=outcome,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        if interrupt_payload is not None:
            return AgentTurn(
                answer="", interrupted=True, interrupt=interrupt_payload, trace_id=trace_id
            )
        if blocked:
            return AgentTurn(answer=state["block_reason"], blocked=True, trace_id=trace_id)
        return AgentTurn(
            answer=state.get("answer", ""),
            tool_steps=_tool_steps(state["messages"]),
            trace_id=trace_id,
        )

    async def stream(
        self,
        *,
        tool_context: ToolContext,
        display_name: str,
        message: str,
        conversation_id: str,
        answer_language: str = "English",
        media_parts: list[MediaPart] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a turn as ordered events: tool steps, then tokens, then ``done``.

        A guardrail block yields a single ``blocked`` event and nothing else. Token
        events carry raw model text; the terminal ``done`` carries the authoritative
        sanitized answer.
        """
        turn_id = str(uuid.uuid4())
        normalizer = self._normalizer_for(media_parts)
        graph = self._graph(
            tool_context, display_name, answer_language, conversation_id, turn_id, normalizer
        )
        started = time.monotonic()
        outcome = "answered"
        with (
            logger.contextualize(
                turn_id=turn_id, conversation_id=conversation_id, user_id=str(tool_context.user_id)
            ),
            self._tracer.span("agent.turn", conversation_id=conversation_id),
        ):
            logger.info("agent.turn.start", message_length=len(message))
            trace_id = self._tracer.current_trace_id()
            try:
                # durability="sync" (LangGraph's default is "async", writing each
                # step's checkpoint in the background): this loop `return`s the
                # instant an interrupt is seen, abandoning this generator — with the
                # default, that can race a still-in-flight background checkpoint
                # write for the very step that paused, so a resume immediately after
                # can find no pending interrupt at all (confirmed: a real, load-
                # dependent flake against a Postgres checkpointer). Every step's
                # checkpoint must be durably written before its chunk is yielded.
                async for mode, chunk in graph.astream(
                    self._inputs(message, tool_context, display_name, media_parts),
                    self._config(conversation_id),
                    stream_mode=["updates", "messages", "custom"],
                    durability="sync",
                ):
                    if mode == "messages":
                        token = _token_from_messages(chunk)
                        if token is not None:
                            yield token
                        continue
                    if mode == "custom":
                        yield NodeStatusEvent(node=chunk["node"], description=chunk["description"])
                        continue
                    # mode == "updates": one {node_name: state_delta} per completed
                    # node, or a single {"__interrupt__": (Interrupt(...),)} when a
                    # node paused the turn (HITL) — never both in the same chunk.
                    if "__interrupt__" in chunk:
                        payload = _interrupt_payload({"__interrupt__": chunk["__interrupt__"]})
                        if payload is not None:
                            outcome = "interrupted"
                            yield InterruptEvent(payload=payload)
                        return
                    for node, delta in chunk.items():
                        if node == GUARDRAIL_IN and delta.get("block_reason"):
                            outcome = "blocked"
                            yield BlockedEvent(reason=delta["block_reason"])
                            return
                        for event in _events_from_update(node, delta):
                            # Stamp the terminal event with this turn's trace id so
                            # the client can correlate the answer to its trace.
                            if isinstance(event, DoneEvent):
                                event.trace_id = trace_id
                            yield event
            finally:
                logger.info(
                    "agent.turn.finish",
                    outcome=outcome,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )


def _interrupt_payload(state: dict) -> dict[str, Any] | None:
    """Extract the paused ``interrupt()`` value from a graph output/update, if any.

    A gated tool call surfaces as ``state["__interrupt__"]`` — a tuple of
    :class:`~langgraph.types.Interrupt` (LangGraph never raises for this; the
    run/update simply carries it) — rather than a raised exception. Only the
    first is used: :func:`~api.agent.graph.build_agent_graph`'s gate resolves
    one call at a time, so at most one is ever pending.
    """
    interrupts: tuple[Interrupt, ...] = state.get("__interrupt__") or ()
    if not interrupts:
        return None
    value = interrupts[0].value
    return value if isinstance(value, dict) else {"reason": str(value)}


def _token_from_messages(chunk: tuple) -> TokenEvent | None:
    """Translate a ``messages``-mode chunk into a ``TokenEvent`` (answer text only).

    Only the ``generate`` node's content chunks are user-facing answer tokens; the
    tool-deciding pass streams tool-call chunks with empty content (skipped), and
    the cheap-tier guardrail/planner calls run in other nodes (ignored).
    """
    message_chunk, metadata = chunk
    if metadata.get("langgraph_node") != GENERATE:
        return None
    content = getattr(message_chunk, "content", None)
    if not content:
        return None
    text = content if isinstance(content, str) else message_chunk.text
    return TokenEvent(text=text) if text else None


def _events_from_update(node: str, delta: dict) -> list[AgentEvent]:
    """Translate a completed node's state delta into tool/done events."""
    messages = delta.get("messages", []) if isinstance(delta, dict) else []
    if node == GENERATE:
        return [
            ToolCallEvent(name=call["name"], args=call.get("args", {}), id=call["id"])
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
    if node == TOOLS:
        return [
            ToolResultEvent(
                name=message.name or "", content=str(message.content), id=message.tool_call_id
            )
            for message in messages
            if isinstance(message, ToolMessage)
        ]
    if node == GUARDRAIL_OUT and delta.get("answer") is not None:
        return [DoneEvent(answer=delta["answer"])]
    return []


def _tool_steps(messages: list[AnyMessage]) -> list[ToolStep]:
    """Reconstruct the turn's tool steps by pairing tool calls with their results."""
    requested: dict[str, tuple[str, dict]] = {}
    steps: list[ToolStep] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                requested[call["id"]] = (call["name"], call.get("args", {}))
        elif isinstance(message, ToolMessage):
            name, args = requested.get(message.tool_call_id, (message.name or "", {}))
            steps.append(ToolStep(name=name, args=args, result=str(message.content)))
    return steps
