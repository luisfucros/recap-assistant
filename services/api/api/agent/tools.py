"""The agent's tools: four user-scoped read tools plus two consequential ones.

The four read tools give the agent its knowledge sources, each wrapping a
service that already enforces isolation and reading-position rules:

* ``get_reading_progress`` — the reading list, statuses, and current/last-recapped
  page (Source 1, Postgres) via :class:`ProgressService`.
* ``retrieve_chunks`` — position-aware, spoiler-safe semantic search over the
  reader's documents (Source 3, vectors) via :class:`RetrievalService`.
* ``summarize`` — a grounded recap of a page span via a cheap-tier model over the
  chunks that cover it. With no tracked position and no explicit range, it
  interrupts to ask which pages were read (FR-4.7) rather than fabricating a
  recap or silently returning nothing, then records the answered position.
* ``query_long_term_memory`` — recall from Source 2 (``long_term_memory``): a
  semantic search over saved summaries/preferences/facts/habits/FAQs via
  :class:`MemoryService`, or — with no ``query`` and a ``document_id`` — a
  deterministic, page-ordered lookup of a document's saved summaries via
  :class:`LongTermMemoryRepository.list_summaries_covering`, so a later recap
  can reuse a saved summary instead of re-reading and re-summarizing the chunks.

Two more reach beyond the reader's own stored data (FR-13.1) and are gated
accordingly:

* ``web_search`` — a general web search via the configured
  :class:`~shared.providers.base.WebSearchProvider` (Brave/Tavily). Declared
  ``requires_approval=True``, so the tool node (``api.agent.graph``) interrupts
  before it ever runs.
* ``recommend`` — explainable reading recommendations via
  :class:`~api.services.recommendation_service.RecommendationService`. The
  internal, library-similarity path runs ungated; only when the model asks to
  also search the web (``include_web=True``) does the tool itself interrupt for
  approval before that branch runs, mirroring ``summarize``'s ask-when-missing
  pause — the declarative per-tool flag can't express "gated on this call's
  arguments", so the gate lives in the tool body instead of the static flag.

Every tool is built by :func:`build_agent_tools`, which closes each one over a
per-turn :class:`~api.agent.context.ToolContext`. The LLM supplies only semantic
arguments (a query, a document id, a page range); the **owner** (``user_id``) and
all data handles come from the context, server-side — so a tool call can never
widen its scope beyond the authenticated user. This is where the isolation
invariant is upheld on the tool boundary.
"""

import re
import uuid

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from api.agent.context import ToolContext
from api.llm import model_label
from api.services.recommendation_service import Recommendation
from api.services.retrieval_service import RetrievedChunk
from shared.core.enums import MemoryType
from shared.core.spoiler import resolve_spoiler_safe
from shared.models.memory import LongTermMemory
from shared.models.reading import ReadingProgress
from shared.observability.metrics import time_operation
from shared.observability.tracing import NoOpTracer, Tracer
from shared.providers.base import SearchResult

# Every document_id field below shares this warning: a document's title (shown
# to the model in get_reading_progress's own output) is untrustworthy as an id —
# it comes from the file's own metadata/text and can be anything (an accession
# number, a filename, a subtitle). The literal id is always shown alongside it
# as "[id: <uuid>]"; that substring, verbatim, is the only valid document_id.
_DOCUMENT_ID_HINT = (
    "Must be the literal id from a prior get_reading_progress call's output "
    "(the uuid shown as '[id: <uuid>]' there) — never a title, filename, or "
    "any other identifier that appears in the document's own text or metadata."
)


class GetReadingProgressArgs(BaseModel):
    """Arguments for the ``get_reading_progress`` tool (no owner — injected)."""

    document_id: str | None = Field(
        default=None,
        description="A specific document's id to report on; omit to get the whole "
        "reading list (all tracked books grouped by status, plus recent activity). "
        f"{_DOCUMENT_ID_HINT}",
    )


class RetrieveChunksArgs(BaseModel):
    """Arguments for the ``retrieve_chunks`` tool (no owner — injected)."""

    query: str = Field(description="The natural-language search text to find passages for.")
    document_id: str | None = Field(
        default=None,
        description="Restrict the search to one document (recommended when the "
        "question is about a specific book); omit to search the reader's library. "
        f"{_DOCUMENT_ID_HINT}",
    )
    chapter: str | None = Field(default=None, description="Restrict to a chapter label.")
    section: str | None = Field(default=None, description="Restrict to a section label.")
    include_unread: bool = Field(
        default=False,
        description="Also search pages the reader hasn't reached yet. Ignored when "
        "spoiler-safe mode is on — unread pages are never returned then.",
    )
    limit: int | None = Field(
        default=None, description="Maximum passages to return (defaults to the system top-k)."
    )


class SummarizeArgs(BaseModel):
    """Arguments for the ``summarize`` tool (no owner — injected)."""

    document_id: str = Field(description=f"The document to recap. {_DOCUMENT_ID_HINT}")
    page_start: int | None = Field(
        default=None,
        description="First page of the span to recap (1-based). Omit to start just "
        "after the last page already summarized.",
    )
    page_end: int | None = Field(
        default=None,
        description="Last page of the span to recap (1-based). Omit to end at the "
        "reader's current page. Never exceeds the current page under spoiler-safe mode.",
    )


class QueryLongTermMemoryArgs(BaseModel):
    """Arguments for the ``query_long_term_memory`` tool (no owner — injected)."""

    query: str | None = Field(
        default=None,
        description="Semantic search text (e.g. 'what did I say about the ending'). "
        "Omit it with a document_id to look up that document's saved summaries "
        "directly by page range instead of searching — check this before "
        "re-summarizing a range that may already be saved.",
    )
    type: MemoryType | None = Field(
        default=None,
        description="Restrict to one memory kind (preference, summary, concept, "
        "fact, habit, or faq); omit to search across all kinds.",
    )
    document_id: str | None = Field(
        default=None,
        description="Restrict to one document's memories; required for a "
        f"page-range lookup (when query is omitted). {_DOCUMENT_ID_HINT}",
    )
    page_start: int | None = Field(
        default=None, description="Only memories overlapping on or after this page."
    )
    page_end: int | None = Field(
        default=None, description="Only memories overlapping on or before this page."
    )
    limit: int | None = Field(
        default=None, description="Maximum memories to return (defaults to the system top-k)."
    )


class WebSearchArgs(BaseModel):
    """Arguments for the ``web_search`` tool (no owner — injected)."""

    query: str = Field(description="The web search query.")
    count: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return.")


class RecommendArgs(BaseModel):
    """Arguments for the ``recommend`` tool (no owner — injected)."""

    include_web: bool = Field(
        default=False,
        description="Also search the web for further reading suggestions beyond "
        "the reader's own library. This reaches a third party, so it needs the "
        "reader's approval before it runs — expect the call to pause.",
    )
    query: str | None = Field(
        default=None,
        description="An explicit web-search query to use when include_web is "
        "true (e.g. a genre, author, or theme). Omit to build one from the "
        "reader's reading history.",
    )
    limit: int = Field(default=5, ge=1, le=10, description="Maximum recommendations to return.")


# Matches the "[id: <uuid>]" suffix _document_label renders. Despite the prompt's
# explicit instruction to extract just the uuid, the model sometimes echoes the
# whole label it was shown back as the document_id argument instead (a real,
# observed failure — see _parse_document_id) — tolerating that shape here is
# strictly more robust than relying on prompting alone to get it right every time.
_LABEL_ID_PATTERN = re.compile(
    r"\[id:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\]"
)


def _parse_document_id(value: str) -> uuid.UUID | None:
    """Parse a tool-supplied document id, returning ``None`` if none can be found.

    Accepts a bare uuid, or the full "Title [id: <uuid>]" label verbatim — the
    latter recovers from the model passing back what it was shown instead of
    extracting the id from it, rather than failing the call outright.
    """
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        pass
    match = _LABEL_ID_PATTERN.search(value)
    if match is None:
        return None
    try:
        return uuid.UUID(match.group(1))
    except ValueError:  # pragma: no cover — the regex only captures valid hex groups
        return None


def requires_approval(tool: BaseTool) -> bool:
    """Whether a tool call must be approved by the user before it runs (HITL).

    Read from the tool's ``extras`` bag (a plain ``dict`` every ``BaseTool``
    carries) rather than a hard-coded tool-name list, so the gate is a
    **declared per-tool property**: any future consequential tool (external
    egress, external cost/rate-limited API, or a side effect) inherits the gate
    simply by setting ``extras={"requires_approval": True}`` at construction —
    the tool-execution node (``api.agent.graph``) checks this for every call.
    Defaults to ``False``, so the four read-only tools run unblocked; ``web_search``
    sets it. ``recommend`` deliberately does *not* — its external branch gates
    itself with a direct ``interrupt()`` call instead, since the static flag
    can't express "only when this call's ``include_web`` argument is set".
    """
    return bool((tool.extras or {}).get("requires_approval", False))


def build_agent_tools(context: ToolContext, tracer: Tracer | None = None) -> list[BaseTool]:
    """Build the agent's six tools bound to one turn's user-scoped context.

    Each returned tool references only ``context`` for its owner and data handles,
    so the LLM cannot supply a ``user_id`` and cannot reach another user's data.
    Returns the tools in a stable order (progress, retrieve, summarize, memory,
    web_search, recommend). ``tracer`` is optional (defaults to a no-op) —
    ``retrieve_chunks`` and ``query_long_term_memory`` each open a child span
    with the returned hits' ids/scores, ``summarize`` one with its prompt ref
    and model, and ``web_search``/``recommend`` ones with their result counts,
    for detail the outer per-turn "tools" span doesn't carry.
    """
    tracer = tracer or NoOpTracer()

    async def get_reading_progress(document_id: str | None = None) -> str:
        """Report the reader's progress: one document, or the whole reading list."""
        if document_id is None:
            return await _reading_list(context)
        parsed = _parse_document_id(document_id)
        if parsed is None:
            return f"'{document_id}' is not a valid document id."
        row = await context.progress_service.get_progress(
            progress=context.progress_repo, document_id=parsed
        )
        if row is None:
            return "That document isn't being tracked yet — no reading progress recorded."
        label = await _document_label(context, parsed)
        return _format_progress(row, label)

    async def retrieve_chunks(
        query: str,
        document_id: str | None = None,
        chapter: str | None = None,
        section: str | None = None,
        include_unread: bool = False,
        limit: int | None = None,
    ) -> str:
        """Search the reader's documents for passages relevant to a query."""
        parsed: uuid.UUID | None = None
        if document_id is not None:
            parsed = _parse_document_id(document_id)
            if parsed is None:
                return f"'{document_id}' is not a valid document id."
        with tracer.span("retrieve_chunks", query=query, document_id=document_id) as span:
            result = await context.retrieval_service.retrieve(
                query=query,
                user_id=context.user_id,
                progress=context.progress_repo,
                chunks=context.chunks,
                document_id=parsed,
                include_unread=include_unread,
                chapter=chapter,
                section=section,
                limit=limit,
                user_spoiler_safe=context.user_spoiler_safe,
            )
            span.update(
                output={
                    "chunks": [
                        {
                            "chunk_id": str(c.chunk_id),
                            "score": c.score,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                        }
                        for c in result.chunks
                    ]
                }
            )
        if not result.chunks:
            return (
                "No relevant passages were found within the pages the reader has "
                "reached. If they've read further, ask them to update their position."
            )
        return "\n\n".join(_format_chunk(i, c) for i, c in enumerate(result.chunks, start=1))

    async def summarize(
        document_id: str, page_start: int | None = None, page_end: int | None = None
    ) -> str:
        """Recap a page span of a document, grounded in the pages it covers."""
        parsed = _parse_document_id(document_id)
        if parsed is None:
            return f"'{document_id}' is not a valid document id."
        document = await context.documents.get(parsed)
        if document is None:
            return "That document isn't in the reader's library."
        row = await context.progress_service.get_progress(
            progress=context.progress_repo, document_id=parsed
        )
        if row is None and page_start is None and page_end is None:
            # FR-4.7: no tracked position and no explicit range either — ask
            # instead of silently returning nothing (never fabricate a recap).
            title = document.title or "this document"
            answer = interrupt(
                {
                    "kind": "ask_pages_read",
                    "document_id": str(parsed),
                    "document_title": title,
                    "reason": f"No reading position is tracked for {title} yet, so "
                    "there's nothing to recap without knowing what's been read.",
                }
            )
            page_start, page_end = _pages_from_answer(answer)
            if page_start is None or page_end is None:
                return "No pages were given, so nothing was recapped."
            row = await context.progress_service.record_position(
                session=context.session,
                documents=context.documents,
                progress=context.progress_repo,
                events=context.events,
                document_id=parsed,
                current_page=page_end,
            )
        start, end = _resolve_summary_span(row, context, page_start=page_start, page_end=page_end)
        if end < start:
            return (
                "There's nothing new to recap — the reader hasn't recorded progress "
                "past the last summarized page."
            )
        chunks = await context.chunks.list_by_document_page_range(
            parsed, page_start=start, page_end=end
        )
        if not chunks:
            return f"No text is indexed for pages {start}-{end} of this document."
        passages = "\n\n".join(f"[pp. {c.page_start}-{c.page_end}] {c.text}" for c in chunks)
        prompt_obj = context.prompts.get("summarize", "v1")
        prompt = prompt_obj.render(
            title=document.title or "this document",
            page_start=start,
            page_end=end,
            passages=passages,
        )
        with tracer.span("summarize", prompt=prompt_obj.ref, model=model_label(context.summarizer)):
            message = await context.summarizer.ainvoke(prompt)
        return message.text() if hasattr(message, "text") else str(message.content)

    async def query_long_term_memory(
        query: str | None = None,
        type: MemoryType | None = None,
        document_id: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        limit: int | None = None,
    ) -> str:
        """Recall saved summaries, preferences, facts, habits, or FAQs."""
        parsed: uuid.UUID | None = None
        if document_id is not None:
            parsed = _parse_document_id(document_id)
            if parsed is None:
                return f"'{document_id}' is not a valid document id."
        top_k = limit or 8
        page_filtered = page_start is not None or page_end is not None
        fetch_limit = max(top_k, 20) if page_filtered else top_k

        with tracer.span(
            "query_long_term_memory",
            query=query,
            document_id=document_id,
            type=type.value if type is not None else None,
        ) as span:
            if query is not None:
                max_page_end = await _memory_read_range_bound(context, parsed)
                hits = await context.memory_service.retrieve(
                    memories=context.memories,
                    query=query,
                    type=type,
                    document_id=parsed,
                    max_page_end=max_page_end,
                    limit=fetch_limit,
                )
            elif parsed is not None:
                if type is not None and type is not MemoryType.SUMMARY:
                    # Document-scoped memories are always summaries; nothing else
                    # can match a document_id filter.
                    hits = []
                else:
                    max_page_end = await _memory_read_range_bound(context, parsed)
                    hits = await context.memories.list_summaries_covering(
                        parsed, max_page_end=max_page_end
                    )
            else:
                hits = await context.memory_service.list_memories(
                    memories=context.memories, type=type, limit=fetch_limit
                )

            if page_filtered:
                hits = [
                    m
                    for m in hits
                    if _overlaps_page_range(m, page_start=page_start, page_end=page_end)
                ]
            hits = hits[:top_k]
            span.update(
                output={
                    "memories": [
                        {
                            "id": str(m.id),
                            "type": m.type.value,
                            "document_id": str(m.document_id) if m.document_id else None,
                            "page_start": m.page_start,
                            "page_end": m.page_end,
                        }
                        for m in hits
                    ]
                }
            )

        if not hits:
            return "No matching memories were found."
        return "\n\n".join([await _format_memory(context, memory) for memory in hits])

    async def web_search(query: str, count: int = 5) -> str:
        """Search the web for information beyond the reader's own documents."""
        with tracer.span("web_search", query=query) as span, time_operation("web_search"):
            results = await context.web_search().search(query, count=count)
            span.update(output={"results": [{"title": r.title, "url": r.url} for r in results]})
        if not results:
            return "No web results were found."
        return "\n\n".join(_format_search_result(i, r) for i, r in enumerate(results, start=1))

    async def recommend(include_web: bool = False, query: str | None = None, limit: int = 5) -> str:
        """Recommend documents to read next, optionally enriched with web results."""
        with tracer.span("recommend", include_web=include_web) as span:
            internal = await context.recommendation_service.recommend_from_library(
                user_id=context.user_id,
                documents=context.documents,
                progress_repo=context.progress_repo,
                progress_service=context.progress_service,
                memories=context.memories,
                memory_service=context.memory_service,
                limit=limit,
            )
            if not include_web:
                span.update(output={"internal": len(internal), "external": 0})
                return _format_recommendations(internal)

            proposed_query = query or await context.recommendation_service.default_web_query(
                documents=context.documents,
                progress_service=context.progress_service,
                progress_repo=context.progress_repo,
            )
            if proposed_query is None:
                span.update(
                    output={"internal": len(internal), "external": 0, "skipped": "no_history"}
                )
                return _format_recommendations(internal) + (
                    "\n\n(There's no reading history yet to search the web from — "
                    "ask again with a specific genre, author, or theme.)"
                )
            # The web search itself is the consequential part of this call — the
            # declarative per-tool flag can't express "only when this argument is
            # set", so the gate is a direct interrupt() here (mirrors summarize's
            # ask-when-missing pause), reusing the tool_approval payload shape so
            # the existing approval UI renders it with no changes.
            decision = interrupt(
                {
                    "kind": "tool_approval",
                    "tool_call": {
                        "name": "recommend",
                        "args": {"include_web": True, "query": proposed_query},
                        "id": "recommend-web",
                    },
                    "reason": "Searching the web for further reading suggestions reaches "
                    "beyond your stored data and needs your approval.",
                }
            )
            action = decision.get("decision") if isinstance(decision, dict) else None
            if action == "deny":
                span.update(output={"internal": len(internal), "external": 0, "denied": True})
                return (
                    _format_recommendations(internal)
                    + "\n\n(Skipped the web search — not approved.)"
                )
            if action == "edit" and isinstance(decision.get("args"), dict):
                proposed_query = decision["args"].get("query", proposed_query)
            with time_operation("web_search"):
                external = await context.recommendation_service.recommend_from_web(
                    web_search=context.web_search(), query=proposed_query, limit=limit
                )
            span.update(output={"internal": len(internal), "external": len(external)})
        return _format_recommendations(internal + external)

    return [
        StructuredTool.from_function(
            coroutine=get_reading_progress,
            name="get_reading_progress",
            description="Get the reader's reading list and per-document progress "
            "(status, current page, last summarized page). Call with no document to "
            "see everything, or with a document id for one book.",
            args_schema=GetReadingProgressArgs,
        ),
        StructuredTool.from_function(
            coroutine=retrieve_chunks,
            name="retrieve_chunks",
            description="Semantic search over the reader's documents for passages "
            "relevant to a query. Results stay within the pages the reader has "
            "reached unless include_unread is set (and never past them in "
            "spoiler-safe mode). Use this to ground answers about the content.",
            args_schema=RetrieveChunksArgs,
        ),
        StructuredTool.from_function(
            coroutine=summarize,
            name="summarize",
            description="Produce a grounded recap of a page span of a document, "
            "defaulting to the pages read since the last summary. Use this for "
            "'catch me up' / 'what did I just read' requests.",
            args_schema=SummarizeArgs,
        ),
        StructuredTool.from_function(
            coroutine=query_long_term_memory,
            name="query_long_term_memory",
            description="Recall saved long-term memories: preferences, facts, "
            "habits, FAQs, and page-range summaries. Omit query with a "
            "document_id to look up that document's saved summaries directly by "
            "page range — check this before calling summarize on a range that "
            "may already be saved, so it isn't re-read and re-summarized.",
            args_schema=QueryLongTermMemoryArgs,
        ),
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description="Search the web for information beyond the reader's own "
            "documents. Reaches a third party, so it always pauses for the "
            "reader's approval before it runs.",
            args_schema=WebSearchArgs,
            extras={"requires_approval": True},
        ),
        StructuredTool.from_function(
            coroutine=recommend,
            name="recommend",
            description="Recommend documents to read next, from the reader's own "
            "library (similarity to what they've read/completed, plus stated "
            "preferences) — explained as 'because you read X'. Set include_web "
            "to also search the web for further suggestions; that part pauses "
            "for the reader's approval before it runs.",
            args_schema=RecommendArgs,
        ),
    ]


# --- formatting & span helpers ------------------------------------------------ #


async def _document_title(context: ToolContext, document_id: uuid.UUID) -> str:
    """Return the owned document's title (or a neutral fallback)."""
    document = await context.documents.get(document_id)
    if document is None or document.title is None:
        return "Untitled document"
    return document.title


async def _document_label(context: ToolContext, document_id: uuid.UUID) -> str:
    """Render a document as "Title [id: <uuid>]" — the model's only source of a valid id.

    A title alone is not a trustworthy identifier: it is populated from the
    document's own PDF metadata/text at ingestion time and can be anything (an
    accession number, a filename, a subtitle) — never assume it looks like a
    title. Every tool output that names a document must go through this (not
    ``_document_title`` alone), or the model has no way to construct a valid
    ``document_id`` for a later tool call.
    """
    title = await _document_title(context, document_id)
    return f"{title} [id: {document_id}]"


async def _reading_list(context: ToolContext) -> str:
    """Render the reader's tracked documents grouped by status, newest first."""
    grouped = await context.progress_service.reading_list(progress=context.progress_repo)
    sections: list[str] = []
    for status, rows in grouped.items():
        if not rows:
            continue
        lines = [
            f"  - {await _document_label(context, row.document_id)} "
            f"(page {row.current_page}, last recapped {row.last_summarized_page})"
            for row in rows
        ]
        sections.append(f"{status.value}:\n" + "\n".join(lines))
    if not sections:
        return "The reader hasn't started tracking any documents yet."
    return "\n".join(sections)


def _format_progress(row: ReadingProgress, label: str) -> str:
    """Render one document's progress row as a compact observation."""
    return (
        f"{label}: {row.status.value}, on page {row.current_page}; "
        f"pages 1-{row.last_summarized_page} have been summarized."
    )


def _format_chunk(index: int, chunk: RetrievedChunk) -> str:
    """Render a retrieved passage with a citation header for the model to quote."""
    cite = chunk.citation
    where = f"pp. {chunk.page_start}-{chunk.page_end}"
    title = cite.title or "Untitled document"
    return f"[{index}] {title} ({where}):\n{chunk.text}"


async def _format_memory(context: ToolContext, memory: LongTermMemory) -> str:
    """Render one recalled memory as a compact, typed observation."""
    if memory.type is MemoryType.SUMMARY and memory.document_id is not None:
        label = await _document_label(context, memory.document_id)
        return f"[summary] {label} (pp. {memory.page_start}-{memory.page_end}): {memory.content}"
    return f"[{memory.type.value}] {memory.content}"


def _format_search_result(index: int, result: SearchResult) -> str:
    """Render one web-search hit for the model to cite."""
    return f"[{index}] {result.title} ({result.url}):\n{result.snippet}"


def _format_recommendations(items: list[Recommendation]) -> str:
    """Render a list of recommendations, each with its explanation and citation."""
    if not items:
        return (
            "No recommendations yet — read or complete a document, or share a "
            "reading preference, to get started."
        )
    lines = []
    for i, rec in enumerate(items, start=1):
        by = f" by {rec.author}" if rec.author else ""
        where = f" ({rec.url})" if rec.url else ""
        lines.append(f"[{i}] {rec.title}{by} — {rec.reason}{where}")
    return "\n".join(lines)


async def _memory_read_range_bound(
    context: ToolContext, document_id: uuid.UUID | None
) -> int | None:
    """Resolve the spoiler-safe ``page_end`` bound for a memory query.

    Mirrors ``RetrievalService``'s own read-range bound: a library-wide query
    (no ``document_id``) is never page-bounded here — a summary's page range
    only means something relative to one document's current page, so the
    agent must scope to a document for the bound to apply (the output-side
    spoiler check is the backstop otherwise). For a targeted document there is
    no ``include_unread`` escape hatch, unlike ``retrieve_chunks`` — FR-18.3
    makes the bound on saved summaries a hard constraint, not a default. With
    no progress row yet, the bound is 0: nothing is recallable until a
    position is recorded.
    """
    if document_id is None:
        return None
    row = await context.progress_repo.get_by_document(document_id)
    return row.current_page if row is not None else 0


def _overlaps_page_range(
    memory: LongTermMemory, *, page_start: int | None, page_end: int | None
) -> bool:
    """Whether a memory's page range overlaps the requested ``[page_start, page_end]``.

    A memory with no page range (a user-level preference/fact/habit/FAQ, not a
    summary) never matches a page-range filter — there's nothing to overlap.
    """
    if page_start is None and page_end is None:
        return True
    if memory.page_start is None or memory.page_end is None:
        return False
    if page_end is not None and memory.page_start > page_end:
        return False
    return not (page_start is not None and memory.page_end < page_start)


def _pages_from_answer(answer: object) -> tuple[int | None, int | None]:
    """Parse a resume answer to an ``ask_pages_read`` interrupt (FR-4.7).

    Expects ``{"page_start": int?, "page_end": int}`` — ``page_end`` is
    required (it's the reader's reported position); ``page_start`` defaults to
    1 (recap from the beginning) when omitted. Anything else (a non-dict, or a
    missing/non-int ``page_end``) means no usable answer.
    """
    if not isinstance(answer, dict):
        return None, None
    end = answer.get("page_end")
    if not isinstance(end, int):
        return None, None
    start = answer.get("page_start")
    return (start if isinstance(start, int) else 1), end


def _resolve_summary_span(
    row: ReadingProgress | None,
    context: ToolContext,
    *,
    page_start: int | None,
    page_end: int | None,
) -> tuple[int, int]:
    """Resolve the (start, end) page span to recap, honoring spoiler-safe mode.

    Defaults span the pages read since the last summary: ``start`` is just past
    ``last_summarized_page`` and ``end`` is the reader's ``current_page``. When
    spoiler-safe resolves on (per-document override, else the user default), the
    end is clamped to ``current_page`` so an explicit ``page_end`` can't be used to
    recap — and thereby reveal — pages the reader hasn't reached. With no progress
    row yet, the read position is 0, so nothing is summarizable until pages are
    recorded.
    """
    current_page = row.current_page if row is not None else 0
    last_summarized = row.last_summarized_page if row is not None else 0
    start = page_start if page_start is not None else last_summarized + 1
    end = page_end if page_end is not None else current_page
    spoiler_on = resolve_spoiler_safe(
        per_query=None,
        per_document=row.spoiler_safe if row is not None else None,
        user_default=context.user_spoiler_safe,
    )
    if spoiler_on:
        end = min(end, current_page)
    return start, end
