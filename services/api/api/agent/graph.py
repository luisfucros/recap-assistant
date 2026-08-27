"""Assembly of the reading assistant's LangGraph state graph.

The graph runs a fixed pipeline per turn, with a tool loop in the middle::

    normalize_input → guardrail_in ─(blocked)→ END
                          │
                     (proceeds)
                          ↓
    load_progress → load_memories → plan → generate ⇄ tools
                                 │
                          (no tool calls)
                                 ↓
    extract_memory → persist_memory → guardrail_out → compact → END

Each stage has one job:

* **normalize_input** — front door for a turn's input (FR-19). First folds any
  audio/image attachments into text (transcribe/describe, archiving the originals)
  so the whole pipeline is single-modality; then runs the deterministic screen —
  flag prompt-injection and redact secrets/PII — overwriting the user message in
  place with its safe, text-only form before any hosted LLM (or guardrail) sees it.
* **guardrail_in** — the LLM topical/safety gate (a structured
  :class:`~api.agent.schemas.GuardrailDecision`); an injection flag short-circuits
  to a block without spending an LLM call. The judge sees the current message
  plus a short prior user/assistant slice (not tool payloads) so follow-ups
  stay classifiable without the full checkpoint.
* **load_progress** — inject the reader's reading-list context so the planner and
  answer are position-aware.
* **load_memories** — inject the reader's recently-saved personal memories
  (preferences/facts/habits/FAQs, not page-range summaries) so a turn is
  personalized even when it's simple enough that the planner never calls
  ``query_long_term_memory`` (a greeting, an aside about the reader). Mirrors
  ``load_progress``.
* **plan** — a cheap-tier :class:`~api.agent.schemas.PlannerDecision`: does the
  turn need tools at all? Sees the current message plus the same short prior
  user/assistant slice as ``guardrail_in``, so follow-ups can still plan
  retrieval instead of skipping tools.
* **generate** — the answer model. When tools are needed it is bound to the six
  tools and drives the tool loop; its final tool-free message is the streamed
  answer. This is the only node whose output is free-form prose.
* **tools** — executes the requested tools (owner injected server-side via the
  :class:`~api.agent.context.ToolContext` the tools close over). A call to a
  tool whose ``requires_approval`` flag is set (HITL; ``web_search`` today, of
  the six) interrupts the whole turn for the user's approve/edit/deny decision
  before running, via LangGraph's ``interrupt()``/checkpointer resume;
  ``recommend``'s external branch gates itself the same way from inside the
  tool body instead (the static flag can't express a per-argument gate).
* **extract_memory** — a cheap-tier :class:`~api.agent.schemas.MemoryClassification`
  judge over the reader's latest message plus a short prior user/assistant
  slice: a salient, non-``summary`` verdict (a *general*, lasting personal
  fact/preference/habit — not the current book or this sitting's recap range)
  is saved immediately via :class:`~api.services.memory_service.MemoryService`
  — no confirmation needed, unlike a page-range summary. Best-effort: a
  failure here never discards the turn's already-generated answer.
* **persist_memory** — after a turn that named a document
  (``state['active_document_id']``), confirms and saves a page-range summary
  when the reader has advanced past the last one (FR-4.6); a no-op otherwise.
* **guardrail_out** — XSS-sanitize the answer into ``state['answer']``, then, when
  this turn's tools named a document and spoiler-safe is on for it, run a
  cheap-tier :class:`~api.agent.schemas.SpoilerCheckDecision` over the sanitized
  text (FR-18.3) — retrieval/summaries already hard-filter to the read range, so
  this catches ahead-of-position leaks from web search or the model's own
  knowledge. A flagged answer pauses (``interrupt()``) to warn and ask for
  explicit opt-in (FR-18.4) rather than silently revealing or withholding it.
* **compact** — token-budget auto-compaction of the checkpointed history
  (FR-4.1): recomputes the running token count and, once it crosses the
  configured fraction of the active model's context window, summarizes the
  whole history (cheap tier) and rewrites it down to that summary via a
  ``RemoveMessage(id=REMOVE_ALL_MESSAGES)``. A no-op when no
  :class:`~api.services.compaction_service.CompactionService` is configured.
  Runs *after* the answer is finalized, so it only ever affects the *next*
  turn's context, never this one; the curated user-visible transcript
  (``ConversationService``) is untouched either way.

Every prompt is pulled from the registry by ``name@version`` (no inline prompts);
the LLM entry points are injected as :class:`AgentModels` so the graph is unit
-testable against fakes with no network.

Every node above is wrapped by ``_traced_node``: it logs the node's start/finish
(and any exception) at INFO under the ``turn_id``/``conversation_id``/``user_id``
context ``AgentService`` binds for the whole turn, and — for every node except
``compact`` — emits a short, human-readable status via LangGraph's custom
stream mode (``AgentService.stream`` translates it into a ``NodeStatusEvent``)
so a client can show live progress before the first answer token arrives.
"""

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, interrupt
from loguru import logger

from api.agent.context import ToolContext
from api.agent.guardrails import GuardrailService
from api.agent.schemas import (
    GuardrailDecision,
    MemoryClassification,
    PageRangeProposal,
    PlannerDecision,
    SpoilerCheckDecision,
)
from api.agent.state import AgentState
from api.agent.tools import (
    _document_title,
    _parse_document_id,
    build_agent_tools,
    requires_approval,
)
from api.services.compaction_service import CompactionService
from api.services.multimodal_service import MultimodalNormalizer, NormalizedPart
from api.services.scratchpad_service import ScratchpadNote, ScratchpadService
from shared.core.enums import MemoryType, ScratchpadKind
from shared.core.spoiler import resolve_spoiler_safe
from shared.observability.metrics import record_tokens, time_operation
from shared.observability.tracing import NoOpTracer, Tracer

# Node names, referenced by the routers and by the streaming service (which keys
# token/tool events off ``GENERATE``/``TOOLS``). Kept as constants so the two
# stay in lockstep.
NORMALIZE = "normalize_input"
GUARDRAIL_IN = "guardrail_in"
LOAD_PROGRESS = "load_progress"
LOAD_MEMORIES = "load_memories"
PLAN = "plan"
GENERATE = "generate"
TOOLS = "tools"
EXTRACT_MEMORY = "extract_memory"
PERSIST_MEMORY = "persist_memory"
GUARDRAIL_OUT = "guardrail_out"
COMPACT = "compact"

# Short, human-readable descriptions of what each node is doing, used for live
# progress streaming (AgentService.stream(), via the "custom" LangGraph stream
# mode) as well as structured logging. COMPACT is intentionally absent: it
# runs after guardrail_out has already produced the answer/DoneEvent, so a
# status for it would arrive after the turn looks finished to the user.
_NODE_DESCRIPTIONS: dict[str, str] = {
    NORMALIZE: "Reading your message...",
    GUARDRAIL_IN: "Checking your request...",
    LOAD_PROGRESS: "Checking your reading progress...",
    LOAD_MEMORIES: "Recalling what we know about you...",
    PLAN: "Planning how to respond...",
    GENERATE: "Preparing a response...",
    TOOLS: "Looking things up...",
    EXTRACT_MEMORY: "Wrapping up...",
    PERSIST_MEMORY: "Updating your reading notes...",
    GUARDRAIL_OUT: "Finalizing your response...",
}


def _traced_node(
    name: str, fn: Callable[[AgentState], Awaitable[dict]]
) -> Callable[[AgentState], Awaitable[dict]]:
    """Wrap a node with start/finish/error logging and a live progress event.

    Every node's entry/exit is logged at INFO so a turn is traceable end-to-end
    regardless of whether Langfuse tracing is configured — the ambient
    ``turn_id``/``conversation_id``/``user_id`` come from the
    ``logger.contextualize`` block ``AgentService`` binds around the whole
    turn, not threaded here. ``get_stream_writer()`` resolves to LangGraph's
    built-in no-op writer unless the caller requested ``stream_mode="custom"``
    (only ``AgentService.stream`` does), so ``run``/``resume`` pay nothing
    extra for it.
    """

    async def wrapper(state: AgentState) -> dict:
        description = _NODE_DESCRIPTIONS.get(name)
        if description:
            get_stream_writer()({"node": name, "description": description})
        logger.info("node.start", node=name, description=description)
        started = time.monotonic()
        try:
            result = await fn(state)
        except GraphInterrupt:
            # HITL pause (tool approval / page-range confirm / spoiler warning):
            # expected control flow, not a node failure — no error log.
            raise
        except Exception:
            logger.opt(exception=True).error("node.error", node=name)
            raise
        logger.info("node.finish", node=name, duration_ms=int((time.monotonic() - started) * 1000))
        return result

    return wrapper


class MultimodalNotConfiguredError(RuntimeError):
    """A turn carried audio/image parts but no normalizer was wired to handle them.

    A configuration error, not a user error: the route only accepts media when the
    providers are configured, so reaching a media turn without a normalizer means
    the wiring is wrong.
    """


# Canned refusals for paths that never reach the LLM (injection short-circuit,
# empty-reason fallback). Keyed by the same human-readable language names the
# generate prompt uses so a blocked turn is in the reader's language (FR-16.4).
_EN = "English"
_BLOCK_COPY: dict[str, dict[str, str]] = {
    "off_topic": {
        "English": (
            "I'm your reading companion, so I can only help with the books and "
            "documents you've added and your reading of them."
        ),
        "Spanish": (
            "Soy tu compañero de lectura, así que solo puedo ayudarte con los "
            "libros y documentos que hayas añadido y con tu lectura de ellos."
        ),
        "German": (
            "Ich bin dein Leseassistent und kann dir nur bei den Büchern und "
            "Dokumenten helfen, die du hinzugefügt hast, und bei deiner Lektüre."
        ),
        "French": (
            "Je suis ton compagnon de lecture : je ne peux t'aider qu'avec les "
            "livres et documents que tu as ajoutés et avec ta lecture."
        ),
        "Italian": (
            "Sono il tuo compagno di lettura, quindi posso aiutarti solo con i "
            "libri e i documenti che hai aggiunto e con la tua lettura."
        ),
    },
    "injection": {
        "English": (
            "I can't follow instructions that try to change how I work. I'm here "
            "to help with your reading — ask me about your books or documents."
        ),
        "Spanish": (
            "No puedo seguir instrucciones que intenten cambiar cómo trabajo. "
            "Estoy aquí para ayudarte con tu lectura: pregúntame por tus libros "
            "o documentos."
        ),
        "German": (
            "Ich kann keine Anweisungen befolgen, die versuchen, meine "
            "Arbeitsweise zu ändern. Ich bin da, um dir bei deiner Lektüre zu "
            "helfen — frag mich nach deinen Büchern oder Dokumenten."
        ),
        "French": (
            "Je ne peux pas suivre des instructions qui tentent de changer ma "
            "façon de travailler. Je suis là pour t'aider dans ta lecture — "
            "pose-moi des questions sur tes livres ou documents."
        ),
        "Italian": (
            "Non posso seguire istruzioni che cercano di cambiare il mio "
            "funzionamento. Sono qui per aiutarti con la lettura: chiedimi dei "
            "tuoi libri o documenti."
        ),
    },
}


def _localized_block_reason(kind: str, language: str) -> str:
    """User-facing canned refusal in the reader's answer language (FR-16.4).

    Unknown languages fall back to English rather than failing a blocked turn.
    """
    copies = _BLOCK_COPY[kind]
    return copies.get(language, copies[_EN])


@dataclass(slots=True)
class AgentModels:
    """The LLM entry points the graph nodes call, injected for testability.

    ``guardrail_judge``, ``planner``, ``spoiler_judge``, and ``memory_classifier``
    are structured-output runnables (they return a :class:`GuardrailDecision` /
    :class:`PlannerDecision` / :class:`SpoilerCheckDecision` /
    :class:`MemoryClassification`), typically
    ``build_resilient_chat_model(..., tier='cheap').with_structured_output(...)``.

    ``answer_model`` is the **raw** primary chat model and ``answer_fallbacks`` the
    raw per-provider fallbacks — both un-wrapped so the graph can ``bind_tools`` to
    each (a ``BaseChatModel`` method the resilience wrappers hide) *before*
    composing ``with_retry``/``with_fallbacks`` over the tool-bound models. Left
    empty, the answer runs on the primary alone (its previous behavior).
    ``max_retries`` is the transient-error retry budget applied per provider.

    ``provider``/``cheap_model``/``default_model`` are the settings-resolved
    labels for trace-span metadata (see ``api.llm.model_id_for``): the guardrail
    judge and planner run on wrapped resilience ``Runnable``s, which don't expose
    the underlying model for introspection, so their span labels come from
    config instead of the (unavailable) model object.
    """

    guardrail_judge: Runnable  # str prompt -> GuardrailDecision
    planner: Runnable  # str prompt -> PlannerDecision
    spoiler_judge: Runnable  # str prompt -> SpoilerCheckDecision
    memory_classifier: Runnable  # str prompt -> MemoryClassification
    answer_model: BaseChatModel  # primary, raw (tool-bindable, streamable)
    answer_fallbacks: list[BaseChatModel] = field(default_factory=list)  # raw fallbacks
    max_retries: int = 0  # per-provider transient-error retry budget
    provider: str = ""  # settings.llm_provider, for span metadata
    cheap_model: str = ""  # resolved cheap-tier model id, for span metadata
    default_model: str = ""  # resolved default-tier model id, for span metadata


def _latest_user_text(state: AgentState) -> str:
    """Return the text of the most recent human message in the transcript."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


# Bounded backdrop for ``guardrail_in``: enough turns to resolve anaphora, not
# the full checkpoint (tool payloads and per-turn injected system notes stay out).
_GUARDRAIL_HISTORY_TURNS = 4
_GUARDRAIL_SNIPPET_CHARS = 400
_INJECTED_SYSTEM_PREFIXES = (
    "Reading list for this user:",
    "What we remember about ",
    "Nothing has been saved about ",
)
_NO_PRIOR_TURNS = "(none — first turn)"


def _plain_text(message: AnyMessage) -> str:
    """Strip a message down to non-empty text; non-strings become ``str``."""
    content = message.content
    if not isinstance(content, str):
        content = str(content) if content else ""
    return content.strip()


def _clip_snippet(text: str) -> str:
    """Cap one history line so a long prior answer cannot bloat the judge."""
    if len(text) <= _GUARDRAIL_SNIPPET_CHARS:
        return text
    return text[:_GUARDRAIL_SNIPPET_CHARS].rstrip() + "…"


def _guardrail_history_line(message: AnyMessage) -> tuple[str, str] | None:
    """Map a checkpoint message to a (role, text) line, or skip it.

    Human/AI prose is kept. Tool results are dropped (they carry retrieved
    passages). Per-turn ``load_progress``/``load_memories`` system notes are
    dropped; a compaction seed (any other system message) is kept so a
    rewritten thread still has backdrop.
    """
    text = _plain_text(message)
    if not text:
        return None
    if isinstance(message, HumanMessage):
        return ("Reader", text)
    if isinstance(message, AIMessage):
        return ("Assistant", text)
    if isinstance(message, SystemMessage) and not text.startswith(_INJECTED_SYSTEM_PREFIXES):
        return ("Summary", text)
    return None


def _recent_chat_context(
    messages: list[AnyMessage], *, max_turns: int = _GUARDRAIL_HISTORY_TURNS
) -> str:
    """Format a short prior-turn slice for the cheap-tier judges.

    Drops this turn's assistant reply if it is already on the transcript
    (``extract_memory`` runs after generate) and the current human utterance
    (rendered separately as ``$message``). Keeps at most ``max_turns`` earlier
    reader turns plus the assistant/summary lines among them. Shared so
    guardrail, planner, and memory classifier resolve anaphora without the
    full checkpoint or tool payloads.
    """
    lines = [pair for message in messages if (pair := _guardrail_history_line(message))]
    if lines and lines[-1][0] == "Assistant":
        lines = lines[:-1]
    if lines and lines[-1][0] == "Reader":
        lines = lines[:-1]
    if not lines:
        return _NO_PRIOR_TURNS
    reader_idxs = [i for i, (role, _) in enumerate(lines) if role == "Reader"]
    start = reader_idxs[-max_turns] if len(reader_idxs) >= max_turns else 0
    return "\n".join(f"{role}: {_clip_snippet(text)}" for role, text in lines[start:])


def _document_id_from_tool_calls(message: AnyMessage) -> str | None:
    """Return the first valid ``document_id`` argument among a message's tool calls.

    Backs ``active_document_id`` (state.py): the answer model's tool calls are
    the only structured signal of which document a turn was about, so
    ``persist_memory``/``guardrail_out`` know what to check without guessing
    from prose. Always the *canonical* uuid string — never the raw argument —
    since ``_parse_document_id`` tolerates a tool call whose ``document_id``
    was the whole "Title [id: <uuid>]" label rather than the bare uuid; both
    downstream readers do a plain ``uuid.UUID(...)`` on this value with no
    tolerance of their own, so returning the raw label back would just move
    the same parse failure one step later instead of fixing it.
    """
    if not isinstance(message, AIMessage):
        return None
    for call in message.tool_calls:
        raw = call.get("args", {}).get("document_id")
        if not raw:
            continue
        parsed = _parse_document_id(raw)
        if parsed is not None:
            return str(parsed)
    return None


def _record_token_usage(response: object) -> dict[str, int]:
    """Record prompt/completion token metrics from a model response, if reported.

    Real chat models attach ``usage_metadata`` (``input_tokens``/``output_tokens``);
    fakes and providers that don't report usage are skipped. Returns the counts so
    the caller can also attach them to the trace span.
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    prompt_tokens = int(usage.get("input_tokens", 0))
    completion_tokens = int(usage.get("output_tokens", 0))
    if prompt_tokens:
        record_tokens("prompt", prompt_tokens)
    if completion_tokens:
        record_tokens("completion", completion_tokens)
    return {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}


def _format_notes(notes: list[ScratchpadNote]) -> str:
    """Render recalled scratchpad notes as a compact working-notes system message."""
    lines = "\n".join(f"- {note.kind.value}: {note.text}" for note in notes)
    return f"Your working notes for this turn (plan and relevant findings):\n{lines}"


def _spoiler_withheld_answer(title: str, current_page: int) -> str:
    """The safe fallback answer when the reader declines a flagged spoiler reveal."""
    return (
        f"I held back part of that answer since it goes beyond page {current_page} "
        f'of "{title}". Let me know if you\'d like me to include it anyway.'
    )


def _combine_input_text(typed: str, parts: list[NormalizedPart]) -> str:
    """Fold a turn's typed text and its normalized attachments into one message.

    The attachments are labelled (transcript / image description) and ordered as
    the user attached them, so the answer model can tell what came from where while
    still reasoning over plain text only (FR-19.4).
    """
    sections: list[str] = []
    if typed.strip():
        sections.append(typed.strip())
    for part in parts:
        label = "audio transcript" if part.kind == "audio" else "image description"
        sections.append(f"[Attached {label}]\n{part.text}")
    return "\n\n".join(sections)


def build_agent_graph(
    *,
    tool_context: ToolContext,
    models: AgentModels,
    display_name: str,
    answer_language: str,
    checkpointer: BaseCheckpointSaver | None = None,
    tracer: Tracer | None = None,
    scratchpad: ScratchpadService | None = None,
    normalizer: MultimodalNormalizer | None = None,
    compaction: CompactionService | None = None,
    conversation_id: str = "",
    turn_id: str = "",
) -> CompiledStateGraph:
    """Compile the agent graph bound to one user's per-turn context.

    Args:
        tool_context: The user-scoped handles/services the read tools and the
            ``load_progress`` node operate through; also the source of the prompt
            registry and the user's default spoiler-safe setting.
        models: The guardrail/planner/answer LLM entry points (injected so tests
            can supply fakes with no network).
        display_name: The reader's name, rendered into the ``generate`` prompt.
        answer_language: The human-readable language the answer is written in
            (the reader's ``preferred_language``), rendered into the ``generate``
            and ``guardrail_in`` prompts so both the reply and any polite
            refusal are in their language even for a foreign document.
        checkpointer: Persists per-conversation state across turns; ``None`` yields
            a stateless graph (each turn starts fresh).
        tracer: Optional Langfuse tracer; each LLM/tool node opens a child span
            under the turn. Defaults to the no-op tracer (tracing is best-effort).
        scratchpad: Optional turn working memory (FR-7.8); ``plan`` writes the
            plan, tool steps append findings, and ``generate`` recalls the
            relevant slices. ``None`` disables it (the agent works without it).
        normalizer: Turns a turn's audio/image attachments into text before the
            screen (FR-19). Required only when a turn actually carries media; a
            media turn built without it raises :class:`MultimodalNotConfiguredError`.
        compaction: Optional token-budget auto-compaction (FR-4.1); ``compact``
            recomputes the running token count every turn and rewrites the
            checkpointed history once it crosses the configured threshold.
            ``None`` disables it (history grows unbounded, today's behavior).
        conversation_id: The thread id, part of the scratchpad key.
        turn_id: This turn's id, part of the scratchpad key (fresh per turn).

    Returns:
        A compiled graph invoked/streamed per turn via ``AgentService``.
    """
    tracer = tracer or NoOpTracer()
    tools = build_agent_tools(tool_context, tracer)

    async def _gate_tool_call(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Interrupt for human approval before running a gated tool call (HITL).

        Checked per call via the tool's declared ``requires_approval`` flag
        (:func:`~api.agent.tools.requires_approval`) — never a hard-coded tool
        name — so read-only tools run straight through and any future
        consequential tool (external egress/cost/side effect) is gated simply
        by setting the flag. ``interrupt()`` pauses the whole turn (persisted by
        the checkpointer under this conversation's thread id) until
        ``AgentService.resume`` supplies a decision:

        * ``{"decision": "approve"}`` — run the call unchanged.
        * ``{"decision": "edit", "args": {...}}`` — run it with the edited args.
        * ``{"decision": "deny"}`` — skip it; the model sees a denial message
          instead of a result (every ``tool_call`` must get a response).
        """
        if request.tool is None or not requires_approval(request.tool):
            return await execute(request)
        call = request.tool_call
        logger.info("tool.approval_required", tool=call["name"], id=call["id"])
        decision = interrupt(
            {
                "kind": "tool_approval",
                "tool_call": {"name": call["name"], "args": call["args"], "id": call["id"]},
                "reason": f"'{call['name']}' reaches beyond your stored data and needs "
                "your approval before it runs.",
            }
        )
        action = decision.get("decision") if isinstance(decision, dict) else None
        if action == "deny":
            return ToolMessage(
                content=f"The user denied this call; '{call['name']}' was not run.",
                name=call["name"],
                tool_call_id=call["id"],
            )
        if action == "edit" and isinstance(decision.get("args"), dict):
            request = request.override(tool_call={**call, "args": decision["args"]})
        return await execute(request)

    tool_node = ToolNode(tools, awrap_tool_call=_gate_tool_call)
    guardrails = GuardrailService()
    prompts = tool_context.prompts
    user_id = tool_context.user_id
    answer_attempts = 1 + max(models.max_retries, 0)

    def _answer_runnable(needs_tools: bool) -> Runnable:
        """Compose the answer model for this turn: bind tools, then add resilience.

        Tool binding must come *before* retry/fallbacks — ``bind_tools`` is a
        ``BaseChatModel`` method the wrappers don't expose — so each provider's
        model is bound (when the turn needs tools) and given its retry budget
        first, then cross-provider fallbacks are composed over the bound models so
        a provider outage falls through to the next without failing the turn. With
        no fallbacks and no retries this returns the primary unchanged (streaming
        and tool-calling behave exactly as before).
        """

        def prepared(model: BaseChatModel) -> Runnable:
            bound = model.bind_tools(tools) if needs_tools else model
            return (
                bound.with_retry(stop_after_attempt=answer_attempts)
                if answer_attempts > 1
                else bound
            )

        primary = prepared(models.answer_model)
        fallbacks = [prepared(m) for m in models.answer_fallbacks]
        return primary.with_fallbacks(fallbacks) if fallbacks else primary

    async def _remember(note: ScratchpadNote) -> None:
        """Append a note to the turn's scratchpad, best-effort (never breaks a turn)."""
        if scratchpad is None:
            return
        try:
            await scratchpad.append(
                user_id=user_id, conversation_id=conversation_id, turn_id=turn_id, note=note
            )
        except Exception:
            logger.warning("scratchpad.append: failed; continuing without it")

    async def _recall(query: str) -> list[ScratchpadNote]:
        """Recall the turn's relevant scratchpad slices, best-effort (else empty)."""
        if scratchpad is None:
            return []
        try:
            return await scratchpad.recall(
                user_id=user_id, conversation_id=conversation_id, turn_id=turn_id, query=query
            )
        except Exception:
            logger.warning("scratchpad.recall: failed; continuing without it")
            return []

    async def _track_tokens(prompt_tokens: int, completion_tokens: int) -> None:
        """Persist the answer model's per-user token spend, best-effort (NFR-13).

        The durable counterpart to the low-cardinality ``recap_llm_tokens_total``
        Prometheus counter, which can't carry a ``user_id`` label. A tracking
        failure must never break the turn, so it's swallowed like the
        scratchpad's own best-effort writes above. Skips the call entirely when
        the model reported no usage (a fake, or a provider that doesn't send
        ``usage_metadata``) rather than writing a zero-valued event.
        """
        if not prompt_tokens and not completion_tokens:
            return
        try:
            await tool_context.usage_service.record_token_usage(
                session=tool_context.session,
                usage=tool_context.usage,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception:
            logger.warning("usage.record_tokens: failed; continuing without it")

    async def _track_tool_call(tool_name: str) -> None:
        """Persist one per-user tool-call count, best-effort (NFR-13)."""
        try:
            await tool_context.usage_service.record_tool_call(
                session=tool_context.session, usage=tool_context.usage, tool_name=tool_name
            )
        except Exception:
            logger.warning("usage.record_tool_call: failed; continuing without it")

    async def normalize_input(state: AgentState) -> dict:
        """Fold attachments to text, then screen: flag injection, redact in place.

        Media parts (FR-19) are transcribed/described and merged into the human
        message *first*, so the deterministic screen here — and every downstream
        node, guardrails included — only ever sees text (FR-19.5).
        """
        # Reset every turn (last-write-wins fields persist across turns via the
        # checkpointer otherwise) — run_tools re-sets it if this turn's tool
        # calls name a document, so persist_memory never acts on a stale one.
        target = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if target is None:
            return {"injection_detected": False, "active_document_id": None}
        text = target.content if isinstance(target.content, str) else str(target.content)
        media_parts = state.get("media_parts") or []
        if media_parts:
            if normalizer is None:
                raise MultimodalNotConfiguredError(
                    "received media parts but no multimodal normalizer is configured"
                )
            with tracer.span("normalize_input") as span:
                normalized = await normalizer.normalize(media_parts, user_id=user_id)
                span.update(output={"parts": len(normalized)})
            text = _combine_input_text(text, normalized)
        screen = guardrails.screen_input(text)
        # Overwrite the message in place (same id) so no raw secret persists in the
        # transcript the checkpointer keeps or the model later sees.
        redacted = HumanMessage(content=screen.redacted_text, id=target.id)
        return {
            "messages": [redacted],
            "injection_detected": screen.injection_detected,
            "active_document_id": None,
        }

    async def guardrail_in(state: AgentState) -> dict:
        """Topical/safety gate. An injection flag blocks without an LLM call."""
        if state.get("injection_detected"):
            return {
                "on_topic": False,
                "safe": False,
                "block_reason": _localized_block_reason("injection", answer_language),
            }
        prompt_obj = prompts.get("guardrail_in", "v4")
        prompt = prompt_obj.render(
            message=_latest_user_text(state),
            answer_language=answer_language,
            conversation_context=_recent_chat_context(state["messages"]),
        )
        with (
            tracer.span(
                "guardrail_in",
                prompt=prompt_obj.ref,
                provider=models.provider,
                model=models.cheap_model,
            ) as span,
            time_operation("llm"),
        ):
            decision: GuardrailDecision = await models.guardrail_judge.ainvoke(prompt)
            span.update(output={"on_topic": decision.on_topic, "safe": decision.safe})
        if decision.on_topic and decision.safe:
            logger.info("guardrail_in.decision", on_topic=True, safe=True, blocked=False)
            return {"on_topic": True, "safe": True, "block_reason": ""}
        logger.info(
            "guardrail_in.decision", on_topic=decision.on_topic, safe=decision.safe, blocked=True
        )
        return {
            "on_topic": decision.on_topic,
            "safe": decision.safe,
            "block_reason": decision.reason
            or _localized_block_reason("off_topic", answer_language),
        }

    async def load_progress(state: AgentState) -> dict:
        """Attach the reader's reading-list context so the turn is position-aware."""
        grouped = await tool_context.progress_service.reading_list(
            progress=tool_context.progress_repo
        )
        lines = [
            f"- {status.value}: {len(rows)} document(s), latest on page {rows[0].current_page}"
            for status, rows in grouped.items()
            if rows
        ]
        summary = "\n".join(lines) if lines else "The reader hasn't started any documents yet."
        context = SystemMessage(content=f"Reading list for this user:\n{summary}")
        return {"messages": [context]}

    async def load_memories(state: AgentState) -> dict:
        """Inject the reader's recently-saved personal memories for personalization.

        Runs every turn, unconditionally — the same reasoning as
        ``load_progress``: a turn simple enough that the planner never calls
        ``query_long_term_memory`` (a greeting, an aside about the reader)
        should still be personalized. Page-range summaries are excluded; those
        are retrieved on demand, scoped to a document, not blanket personal
        context.
        """
        recent = await tool_context.memory_service.list_memories(
            memories=tool_context.memories, limit=20
        )
        personal = [m for m in recent if m.type is not MemoryType.SUMMARY][:8]
        if personal:
            lines = "\n".join(f"- {m.type.value}: {m.content}" for m in personal)
            summary = f"What we remember about {display_name} from earlier conversations:\n{lines}"
        else:
            summary = f"Nothing has been saved about {display_name} yet."
        return {"messages": [SystemMessage(content=summary)]}

    async def plan(state: AgentState) -> dict:
        """Classify complexity and whether the turn needs tools (cheap tier)."""
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        prompt_obj = prompts.get("planner", "v2")
        prompt = prompt_obj.render(
            message=_latest_user_text(state),
            tools=tool_lines,
            conversation_context=_recent_chat_context(state["messages"]),
        )
        with (
            tracer.span(
                "plan", prompt=prompt_obj.ref, provider=models.provider, model=models.cheap_model
            ) as span,
            time_operation("llm"),
        ):
            decision: PlannerDecision = await models.planner.ainvoke(prompt)
            span.update(
                output={
                    "needs_tools": decision.needs_tools,
                    "complexity": decision.complexity.value,
                }
            )
        plan_text = (
            f"Plan ({decision.complexity.value}): call " + ", ".join(decision.tool_plan)
            if decision.tool_plan
            else f"Plan ({decision.complexity.value}): answer directly"
        )
        await _remember(ScratchpadNote(kind=ScratchpadKind.PLAN, text=plan_text))
        logger.info(
            "plan.decision",
            complexity=decision.complexity.value,
            needs_tools=decision.needs_tools,
            tool_plan=decision.tool_plan,
        )
        return {"needs_tools": decision.needs_tools, "complexity": decision.complexity.value}

    async def generate(state: AgentState) -> dict:
        """Produce the answer, driving the tool loop when tools are needed."""
        prompt_obj = prompts.get("generate", "v4")
        system = SystemMessage(
            content=prompt_obj.render(display_name=display_name, answer_language=answer_language)
        )
        model = _answer_runnable(bool(state.get("needs_tools")))
        # Pull only the relevant scratchpad slices back into context (kept out of
        # the prompt otherwise, so a long turn's notes don't bloat every step).
        recalled = await _recall(_latest_user_text(state))
        notes = [SystemMessage(content=_format_notes(recalled))] if recalled else []
        with (
            tracer.span(
                "generate",
                prompt=prompt_obj.ref,
                provider=models.provider,
                model=models.default_model,
            ) as span,
            time_operation("llm"),
        ):
            response = await model.ainvoke([system, *notes, *state["messages"]])
            usage = _record_token_usage(response)
            span.update(
                output={"tool_calls": len(getattr(response, "tool_calls", []) or [])} | usage
            )
        await _track_tokens(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        return {"messages": [response]}

    async def run_tools(state: AgentState) -> dict:
        """Execute the requested tools, timed and traced as one tool step."""
        last_ai_message = state["messages"][-1]
        if isinstance(last_ai_message, AIMessage):
            for call in last_ai_message.tool_calls:
                logger.info("tool.call", tool=call["name"], id=call["id"])
        with tracer.span("tools") as span, time_operation("tool"):
            result = await tool_node.ainvoke(state)
            span.update(output={"messages": len(result.get("messages", []))})
        if isinstance(last_ai_message, AIMessage):
            for call in last_ai_message.tool_calls:
                await _track_tool_call(call["name"])
        # Record each observation as a finding so a later step can recall it
        # without the full tool output riding in the prompt every turn. Content
        # itself is never logged (may carry retrieved document text/PII) — only
        # the tool identity and id are.
        for message in result.get("messages", []):
            if isinstance(message, ToolMessage):
                logger.info("tool.result", tool=message.name, id=message.tool_call_id)
            await _remember(ScratchpadNote(kind=ScratchpadKind.FINDING, text=str(message.content)))
        # Always (re)set — a turn whose calls named no document must clear any
        # value left by an earlier turn, not silently inherit it.
        result["active_document_id"] = _document_id_from_tool_calls(state["messages"][-1])
        return result

    async def extract_memory(state: AgentState) -> dict:
        """Save a durable personal fact the reader just shared, if any (FR-7.9).

        Runs the cheap-tier :class:`MemoryClassification` judge over the
        reader's latest message (plus a short prior-turn slice for follow-ups);
        a salient, non-``summary`` verdict is saved immediately via
        :class:`~api.services.memory_service.MemoryService` — no confirmation
        needed, unlike a page-range summary (FR-4.6). Only general, lasting
        traits belong here (genre tastes, identity, how they usually read);
        the current title or a recap range for this sitting must not be
        persisted, or later sessions replay it as a standing instruction.
        Best-effort: a classification or write failure never breaks the turn.
        """
        try:
            prompt_obj = prompts.get("memory_classify", "v2")
            prompt = prompt_obj.render(
                message=_latest_user_text(state),
                conversation_context=_recent_chat_context(state["messages"]),
            )
            with (
                tracer.span(
                    "extract_memory",
                    prompt=prompt_obj.ref,
                    provider=models.provider,
                    model=models.cheap_model,
                ) as span,
                time_operation("llm"),
            ):
                decision: MemoryClassification = await models.memory_classifier.ainvoke(prompt)
                span.update(output={"salient": decision.salient, "type": decision.type.value})
            saved = (
                decision.salient and decision.type is not MemoryType.SUMMARY and decision.content
            )
            logger.info(
                "extract_memory.decision",
                salient=decision.salient,
                type=decision.type.value,
                saved=bool(saved),
            )
            if saved:
                await tool_context.memory_service.write_memory(
                    memories=tool_context.memories,
                    session=tool_context.session,
                    type=decision.type,
                    content=decision.content,
                )
        except Exception:
            logger.warning("extract_memory: failed; continuing without it")
        return {}

    async def persist_memory(state: AgentState) -> dict:
        """Confirm and save a summary for newly-read pages, if any (FR-4.6).

        Fires only when this turn's tools named a specific document
        (``state["active_document_id"]``, set by ``run_tools``) *and* that
        document has pages read past its last saved summary
        (``current_page > last_summarized_page``). Proposes the unsummarized
        gap as the new summary's range and pauses (``interrupt()``) for the
        reader to confirm or edit it before anything is written — a deny, no
        active document, or nothing new to recap are all no-ops, never a
        fabricated summary.
        """
        raw_document_id = state.get("active_document_id")
        if raw_document_id is None:
            return {}
        document_id = uuid.UUID(raw_document_id)
        row = await tool_context.progress_repo.get_by_document(document_id)
        if row is None or row.current_page <= row.last_summarized_page:
            return {}
        proposal = PageRangeProposal(
            page_start=row.last_summarized_page + 1,
            page_end=row.current_page,
            proposal_reason="pages read since the last saved summary",
        )
        document_title = await _document_title(tool_context, document_id)
        with tracer.span("persist_memory", document_id=str(document_id)) as span:
            decision = interrupt(
                {
                    "kind": "page_range_confirm",
                    "document_id": str(document_id),
                    "document_title": document_title,
                    "proposal": proposal.model_dump(),
                }
            )
            action = decision.get("decision") if isinstance(decision, dict) else None
            if action == "deny":
                span.update(output={"saved": False, "reason": "denied"})
                return {}
            page_start = (
                decision.get("page_start", proposal.page_start)
                if isinstance(decision, dict)
                else proposal.page_start
            )
            page_end = (
                decision.get("page_end", proposal.page_end)
                if isinstance(decision, dict)
                else proposal.page_end
            )
            chunks = await tool_context.chunks.list_by_document_page_range(
                document_id, page_start=page_start, page_end=page_end
            )
            if not chunks:
                span.update(output={"saved": False, "reason": "no indexed text"})
                return {}
            passages = "\n\n".join(f"[pp. {c.page_start}-{c.page_end}] {c.text}" for c in chunks)
            prompt_obj = prompts.get("summarize", "v1")
            prompt = prompt_obj.render(
                title=document_title, page_start=page_start, page_end=page_end, passages=passages
            )
            message = await tool_context.summarizer.ainvoke(prompt)
            content = message.text() if hasattr(message, "text") else str(message.content)
            await tool_context.memory_service.write_summary(
                memories=tool_context.memories,
                session=tool_context.session,
                document_id=document_id,
                page_start=page_start,
                page_end=page_end,
                content=content,
            )
            await tool_context.progress_service.advance_summarized_page(
                session=tool_context.session,
                progress=tool_context.progress_repo,
                document_id=document_id,
                page=page_end,
            )
            span.update(output={"saved": True, "page_start": page_start, "page_end": page_end})
        return {}

    async def guardrail_out(state: AgentState) -> dict:
        """Sanitize the final answer and, when applicable, screen it for spoilers.

        XSS sanitization (defense-in-depth) always runs. The spoiler check
        (FR-18.3) only runs when this turn's tools named a document
        (``active_document_id``, set the same way ``persist_memory`` reads it) and
        spoiler-safe resolves on for it — retrieval and summaries already
        hard-filter to the read range, so this is the backstop for content that
        could otherwise leak from web search or the model's own knowledge. A
        flagged answer pauses (``interrupt()``) to warn and ask for explicit
        opt-in rather than silently revealing or silently withholding it
        (FR-18.4); declining substitutes a safe, explanatory answer for the
        flagged text.
        """
        last = state["messages"][-1]
        text = last.text if isinstance(last, AIMessage) else str(getattr(last, "content", ""))
        sanitized = guardrails.sanitize_output(text)
        raw_document_id = state.get("active_document_id")
        if raw_document_id is None:
            return {"answer": sanitized}
        document_id = uuid.UUID(raw_document_id)
        row = await tool_context.progress_repo.get_by_document(document_id)
        spoiler_on = resolve_spoiler_safe(
            per_query=None,
            per_document=row.spoiler_safe if row is not None else None,
            user_default=tool_context.user_spoiler_safe,
        )
        if row is None or not spoiler_on:
            return {"answer": sanitized}
        document_title = await _document_title(tool_context, document_id)
        prompt_obj = prompts.get("spoiler_check", "v1")
        prompt = prompt_obj.render(
            title=document_title, current_page=row.current_page, answer=sanitized
        )
        with (
            tracer.span(
                "guardrail_out",
                prompt=prompt_obj.ref,
                provider=models.provider,
                model=models.cheap_model,
            ) as span,
            time_operation("llm"),
        ):
            decision: SpoilerCheckDecision = await models.spoiler_judge.ainvoke(prompt)
            span.update(output={"spoiler_risk": decision.spoiler_risk})
        logger.info("guardrail_out.spoiler_decision", spoiler_risk=decision.spoiler_risk)
        if not decision.spoiler_risk:
            return {"answer": sanitized}
        resolution = interrupt(
            {
                "kind": "spoiler_warning",
                "document_id": str(document_id),
                "document_title": document_title,
                "current_page": row.current_page,
                "reason": decision.reason
                or "This answer includes content past where you are in the book.",
            }
        )
        action = resolution.get("decision") if isinstance(resolution, dict) else None
        if action == "approve":
            return {"answer": sanitized}
        return {"answer": _spoiler_withheld_answer(document_title, row.current_page)}

    async def compact(state: AgentState) -> dict:
        """Auto-compact the checkpointed history once it's past budget (FR-4.1).

        Runs last, after the answer is finalized, so a rewrite here only ever
        affects the *next* turn's context — this turn's answer and the
        user-visible transcript (persisted separately) are unaffected either
        way. A no-op with no ``compaction`` service configured.
        """
        if compaction is None:
            return {}
        messages = state["messages"]
        token_count = await compaction.count_tokens(
            messages, model=models.answer_model, provider=models.provider
        )
        if not compaction.should_compact(token_count=token_count, provider=models.provider):
            return {"token_count": token_count}
        with tracer.span("compact", message_count=len(messages), token_count=token_count) as span:
            result = await compaction.compact(
                messages=messages, summarizer=tool_context.summarizer, prompts=prompts
            )
            span.update(output={"compacted": True, "summary_length": len(result.summary)})
        return {"messages": result.messages, "token_count": 0}

    def _after_guardrail_in(state: AgentState) -> str:
        return END if state.get("block_reason") else LOAD_PROGRESS

    def _after_generate(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return TOOLS
        return EXTRACT_MEMORY

    graph = StateGraph(AgentState)
    graph.add_node(NORMALIZE, _traced_node(NORMALIZE, normalize_input))
    graph.add_node(GUARDRAIL_IN, _traced_node(GUARDRAIL_IN, guardrail_in))
    graph.add_node(LOAD_PROGRESS, _traced_node(LOAD_PROGRESS, load_progress))
    graph.add_node(LOAD_MEMORIES, _traced_node(LOAD_MEMORIES, load_memories))
    graph.add_node(PLAN, _traced_node(PLAN, plan))
    graph.add_node(GENERATE, _traced_node(GENERATE, generate))
    graph.add_node(TOOLS, _traced_node(TOOLS, run_tools))
    graph.add_node(EXTRACT_MEMORY, _traced_node(EXTRACT_MEMORY, extract_memory))
    graph.add_node(PERSIST_MEMORY, _traced_node(PERSIST_MEMORY, persist_memory))
    graph.add_node(GUARDRAIL_OUT, _traced_node(GUARDRAIL_OUT, guardrail_out))
    graph.add_node(COMPACT, _traced_node(COMPACT, compact))

    graph.add_edge(START, NORMALIZE)
    graph.add_edge(NORMALIZE, GUARDRAIL_IN)
    graph.add_conditional_edges(GUARDRAIL_IN, _after_guardrail_in, [LOAD_PROGRESS, END])
    graph.add_edge(LOAD_PROGRESS, LOAD_MEMORIES)
    graph.add_edge(LOAD_MEMORIES, PLAN)
    graph.add_edge(PLAN, GENERATE)
    graph.add_conditional_edges(GENERATE, _after_generate, [TOOLS, EXTRACT_MEMORY])
    graph.add_edge(TOOLS, GENERATE)
    graph.add_edge(EXTRACT_MEMORY, PERSIST_MEMORY)
    graph.add_edge(PERSIST_MEMORY, GUARDRAIL_OUT)
    graph.add_edge(GUARDRAIL_OUT, COMPACT)
    graph.add_edge(COMPACT, END)

    return graph.compile(checkpointer=checkpointer)
