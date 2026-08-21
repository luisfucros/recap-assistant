"""Unit tests for the agent's user-scoped read tools.

The services, repositories, and cheap-tier model are faked at the boundary; the
tools' argument handling, server-side scope injection, spoiler-safe span
clamping, and observation formatting under test are real. The load-bearing
assertion across these tests: the owner (``user_id``) and data handles come from
the injected :class:`ToolContext`, never from tool arguments the LLM controls.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from api.agent.context import ToolContext
from api.agent.tools import _parse_document_id, build_agent_tools, requires_approval
from api.services.recommendation_service import Recommendation
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from shared.core.enums import MemoryType, ReadingStatus
from shared.models.reading import ReadingProgress
from shared.prompt import get_prompt_registry
from shared.providers.base import SearchResult

pytestmark = pytest.mark.unit

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
DOC_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class _NoopArgs(BaseModel):
    """Empty args schema for the HITL gate tests' throwaway tools."""


async def _noop() -> str:
    return "ok"


# --- boundary fakes ---------------------------------------------------------- #


class _FakeProgressService:
    """Stands in for ProgressService: returns canned progress; records position calls."""

    def __init__(self, row: ReadingProgress | None, grouped: dict | None = None) -> None:
        self._row = row
        self._grouped = grouped or {}
        self.record_position_calls: list[dict[str, Any]] = []

    async def get_progress(
        self, *, progress: Any, document_id: uuid.UUID
    ) -> ReadingProgress | None:
        return self._row

    async def reading_list(self, *, progress: Any) -> dict:
        return self._grouped

    async def record_position(self, **kwargs: Any) -> ReadingProgress:
        self.record_position_calls.append(kwargs)
        self._row = ReadingProgress(
            id=uuid.uuid4(),
            user_id=USER_ID,
            document_id=kwargs["document_id"],
            current_page=kwargs["current_page"],
            last_summarized_page=0,
            status=ReadingStatus.READING,
        )
        return self._row


class _FakeProgressRepo:
    """Stands in for ReadingProgressRepository: returns one canned row."""

    def __init__(self, row: ReadingProgress | None = None) -> None:
        self.row = row

    async def get_by_document(self, document_id: uuid.UUID) -> ReadingProgress | None:
        return self.row


class _FakeMemories:
    """Fake LongTermMemoryRepository exposing only the deterministic lookup."""

    def __init__(self, covering: list[Any] | None = None) -> None:
        self._covering = covering or []
        self.covering_calls: list[dict[str, Any]] = []

    async def list_summaries_covering(
        self, document_id: uuid.UUID, *, max_page_end: int | None = None
    ) -> list[Any]:
        self.covering_calls.append({"document_id": document_id, "max_page_end": max_page_end})
        return list(self._covering)


class _FakeMemoryService:
    """Captures retrieve/list_memories calls; returns one canned result list."""

    def __init__(self, result: list[Any] | None = None) -> None:
        self._result = result or []
        self.retrieve_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> list[Any]:
        self.retrieve_calls.append(kwargs)
        return list(self._result)

    async def list_memories(self, **kwargs: Any) -> list[Any]:
        self.list_calls.append(kwargs)
        return list(self._result)


class _FakeRetrievalService:
    """Captures the kwargs a tool passes so injected scope can be asserted."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._result


class _FakeChunks:
    """Fake ChunkRepository exposing only the coverage fetch summarize uses."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.range_calls: list[tuple] = []

    async def list_by_document_page_range(
        self, document_id: uuid.UUID, *, page_start: int, page_end: int
    ) -> list[Any]:
        self.range_calls.append((document_id, page_start, page_end))
        return self._chunks


class _FakeDocuments:
    def __init__(self, document: Any) -> None:
        self._document = document

    async def get(self, entity_id: uuid.UUID) -> Any:
        return self._document


class _FakeSummarizer:
    """Fake cheap-tier chat model: echoes a fixed answer, records the prompt."""

    def __init__(self, text: str = "A concise recap.") -> None:
        self._text = text
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self._text, text=lambda: self._text)


def _progress_row(
    *, current_page: int = 50, last_summarized: int = 20, spoiler_safe: bool | None = None
) -> ReadingProgress:
    return ReadingProgress(
        user_id=USER_ID,
        document_id=DOC_ID,
        current_page=current_page,
        last_summarized_page=last_summarized,
        status=ReadingStatus.READING,
        spoiler_safe=spoiler_safe,
    )


def _retrieved(text: str, *, page_start: int, page_end: int, title: str = "The Odyssey") -> Any:
    citation = SimpleNamespace(
        document_id=DOC_ID, title=title, author=None, page_start=page_start, page_end=page_end
    )
    return SimpleNamespace(
        chunk_id=uuid.uuid4(),
        document_id=DOC_ID,
        text=text,
        score=0.9,
        page_start=page_start,
        page_end=page_end,
        chapter=None,
        section=None,
        citation=citation,
    )


def _memory(
    content: str,
    *,
    memory_id: uuid.UUID | None = None,
    type: MemoryType = MemoryType.PREFERENCE,
    document_id: uuid.UUID | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Any:
    return SimpleNamespace(
        id=memory_id or uuid.uuid4(),
        content=content,
        type=type,
        document_id=document_id,
        page_start=page_start,
        page_end=page_end,
    )


class _FakeWebSearchProvider:
    """Stands in for WebSearchProvider: returns one canned result list."""

    def __init__(self, results: list[Any] | None = None) -> None:
        self._results = results or []
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, *, count: int = 5) -> list[Any]:
        self.calls.append({"query": query, "count": count})
        return list(self._results)


class _FakeRecommendationService:
    """Stands in for RecommendationService: returns one canned list per method."""

    def __init__(
        self,
        internal: list[Any] | None = None,
        external: list[Any] | None = None,
        default_query: str | None = None,
    ) -> None:
        self._internal = internal or []
        self._external = external or []
        self._default_query = default_query
        self.recommend_from_library_calls: list[dict[str, Any]] = []
        self.recommend_from_web_calls: list[dict[str, Any]] = []
        self.default_web_query_calls: list[dict[str, Any]] = []

    async def recommend_from_library(self, **kwargs: Any) -> list[Any]:
        self.recommend_from_library_calls.append(kwargs)
        return list(self._internal)

    async def recommend_from_web(self, **kwargs: Any) -> list[Any]:
        self.recommend_from_web_calls.append(kwargs)
        return list(self._external)

    async def default_web_query(self, **kwargs: Any) -> str | None:
        self.default_web_query_calls.append(kwargs)
        return self._default_query


def _make_context(
    *,
    progress_service: Any = None,
    progress_repo: Any = None,
    retrieval_service: Any = None,
    chunks: Any = None,
    documents: Any = None,
    summarizer: Any = None,
    user_spoiler_safe: bool = False,
    session: Any = None,
    memories: Any = None,
    memory_service: Any = None,
    web_search_provider: Any = None,
    recommendation_service: Any = None,
) -> ToolContext:
    return ToolContext(
        user_id=USER_ID,
        documents=documents or _FakeDocuments(SimpleNamespace(title="The Odyssey")),
        chunks=chunks or _FakeChunks([]),
        progress_repo=progress_repo or _FakeProgressRepo(None),
        progress_service=progress_service or _FakeProgressService(None),
        retrieval_service=retrieval_service or _FakeRetrievalService(SimpleNamespace(chunks=[])),
        summarizer=summarizer or _FakeSummarizer(),
        prompts=get_prompt_registry(),
        user_spoiler_safe=user_spoiler_safe,
        session=session or SimpleNamespace(),
        events=SimpleNamespace(),
        memories=memories or _FakeMemories(),
        memory_service=memory_service or _FakeMemoryService(),
        recommendation_service=recommendation_service or _FakeRecommendationService(),
        web_search=lambda: web_search_provider or _FakeWebSearchProvider(),
        usage=SimpleNamespace(),
        usage_service=SimpleNamespace(),
    )


class _SpySpan:
    def __init__(self, name: str, updates: list) -> None:
        self._name = name
        self._updates = updates

    def update(self, **fields: Any) -> None:
        self._updates.append((self._name, fields))


class _SpyTracer:
    """Records the span names/attributes opened and their updates."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, dict]] = []
        self.updates: list[tuple[str, dict]] = []

    def span(self, name: str, **attributes: Any) -> Any:
        self.opened.append((name, attributes))
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield _SpySpan(name, self.updates)

        return _cm()

    def current_trace_id(self) -> str | None:
        return None

    def flush(self) -> None:
        pass


def _tool(context: ToolContext, name: str, tracer: Any = None) -> Any:
    return next(t for t in build_agent_tools(context, tracer) if t.name == name)


# --- tool wiring ------------------------------------------------------------- #


def test_build_agent_tools_returns_six_named_tools() -> None:
    tools = build_agent_tools(_make_context())
    assert [t.name for t in tools] == [
        "get_reading_progress",
        "retrieve_chunks",
        "summarize",
        "query_long_term_memory",
        "web_search",
        "recommend",
    ]


def test_tool_schemas_do_not_expose_user_id() -> None:
    # The owner is injected from context; no tool may accept it as an argument.
    for tool in build_agent_tools(_make_context()):
        assert "user_id" not in tool.args


# --- requires_approval (HITL gate, FR/M5) ------------------------------------- #


_READ_TOOL_NAMES = {
    "get_reading_progress",
    "retrieve_chunks",
    "summarize",
    "query_long_term_memory",
}


def test_read_tools_do_not_require_approval() -> None:
    # None of the four read-only tools reach beyond the user's own stored
    # data, so none are gated — the flag defaults to False, not opt-out.
    for tool in build_agent_tools(_make_context()):
        if tool.name in _READ_TOOL_NAMES:
            assert requires_approval(tool) is False


def test_web_search_requires_approval() -> None:
    # web_search reaches a third party, so the declarative flag gates every call.
    tool = _tool(_make_context(), "web_search")
    assert requires_approval(tool) is True


def test_recommend_does_not_require_approval_at_the_tool_level() -> None:
    # recommend's internal path is ungated; its external branch self-gates via
    # an in-body interrupt() instead of the static flag (tested in
    # test_agent_service.py, where a graph context can actually pause).
    tool = _tool(_make_context(), "recommend")
    assert requires_approval(tool) is False


def test_requires_approval_reads_the_tool_extras_flag() -> None:
    gated = StructuredTool.from_function(
        coroutine=_noop,
        name="web_search",
        description="d",
        args_schema=_NoopArgs,
        extras={"requires_approval": True},
    )
    ungated = StructuredTool.from_function(
        coroutine=_noop, name="other", description="d", args_schema=_NoopArgs
    )
    assert requires_approval(gated) is True
    assert requires_approval(ungated) is False


def test_requires_approval_defaults_false_with_no_extras() -> None:
    tool = StructuredTool.from_function(
        coroutine=_noop, name="other", description="d", args_schema=_NoopArgs
    )
    assert tool.extras is None
    assert requires_approval(tool) is False


# --- document id parsing ------------------------------------------------------ #


def test_parse_document_id_accepts_a_bare_uuid() -> None:
    assert _parse_document_id(str(DOC_ID)) == DOC_ID


def test_parse_document_id_recovers_from_the_full_label() -> None:
    # A real, observed failure: despite the prompt's explicit instruction to
    # pass only the uuid, the model echoed the whole "Title [id: <uuid>]" label
    # it was shown as the document_id argument — this must still resolve rather
    # than reject the call outright.
    label = f"2001. Una odisea espacial [id: {DOC_ID}]"
    assert _parse_document_id(label) == DOC_ID


def test_parse_document_id_rejects_a_title_with_no_id() -> None:
    assert _parse_document_id("2001. Una odisea espacial") is None


def test_parse_document_id_rejects_garbage() -> None:
    assert _parse_document_id("not-a-uuid") is None


# --- get_reading_progress ---------------------------------------------------- #


async def test_get_reading_progress_single_document() -> None:
    context = _make_context(progress_service=_FakeProgressService(_progress_row()))
    tool = _tool(context, "get_reading_progress")
    out = await tool.ainvoke({"document_id": str(DOC_ID)})
    assert "The Odyssey" in out and "page 50" in out
    # The real document id must be surfaced verbatim — it's the model's only
    # legitimate source of a document_id for a later tool call (a title alone,
    # e.g. from a scanned filing's own metadata, is not a reliable identifier).
    assert f"[id: {DOC_ID}]" in out


async def test_get_reading_progress_rejects_bad_document_id() -> None:
    tool = _tool(_make_context(), "get_reading_progress")
    out = await tool.ainvoke({"document_id": "not-a-uuid"})
    assert "not a valid document id" in out


async def test_get_reading_progress_untracked_document() -> None:
    context = _make_context(progress_service=_FakeProgressService(None))
    tool = _tool(context, "get_reading_progress")
    out = await tool.ainvoke({"document_id": str(DOC_ID)})
    assert "isn't being tracked" in out


async def test_get_reading_progress_reading_list() -> None:
    grouped = {ReadingStatus.READING: [_progress_row(current_page=42, last_summarized=10)]}
    context = _make_context(progress_service=_FakeProgressService(None, grouped))
    tool = _tool(context, "get_reading_progress")
    out = await tool.ainvoke({})
    assert "reading" in out and "page 42" in out and "The Odyssey" in out
    assert f"[id: {DOC_ID}]" in out


# --- retrieve_chunks --------------------------------------------------------- #


async def test_retrieve_chunks_injects_owner_and_flags_not_from_args() -> None:
    retrieval = _FakeRetrievalService(
        SimpleNamespace(chunks=[_retrieved("Odysseus sails.", page_start=1, page_end=2)])
    )
    context = _make_context(retrieval_service=retrieval, user_spoiler_safe=True)
    tool = _tool(context, "retrieve_chunks")
    out = await tool.ainvoke({"query": "who is the narrator", "document_id": str(DOC_ID)})

    assert "Odysseus sails." in out and "The Odyssey" in out
    call = retrieval.calls[0]
    # Owner + spoiler default come from context, never from the tool arguments.
    assert call["user_id"] == USER_ID
    assert call["user_spoiler_safe"] is True
    assert call["document_id"] == DOC_ID


async def test_retrieve_chunks_bad_document_id_short_circuits() -> None:
    retrieval = _FakeRetrievalService(SimpleNamespace(chunks=[]))
    context = _make_context(retrieval_service=retrieval)
    tool = _tool(context, "retrieve_chunks")
    out = await tool.ainvoke({"query": "x", "document_id": "nope"})
    assert "not a valid document id" in out
    assert retrieval.calls == []  # never reached the service


async def test_retrieve_chunks_empty_result_message() -> None:
    context = _make_context(retrieval_service=_FakeRetrievalService(SimpleNamespace(chunks=[])))
    tool = _tool(context, "retrieve_chunks")
    out = await tool.ainvoke({"query": "anything"})
    assert "No relevant passages" in out


async def test_retrieve_chunks_traces_chunk_ids_and_scores() -> None:
    retrieved = _retrieved("Odysseus sails.", page_start=1, page_end=2)
    retrieval = _FakeRetrievalService(SimpleNamespace(chunks=[retrieved]))
    context = _make_context(retrieval_service=retrieval)
    tracer = _SpyTracer()
    tool = _tool(context, "retrieve_chunks", tracer=tracer)

    await tool.ainvoke({"query": "who is the narrator"})

    assert tracer.opened[0][0] == "retrieve_chunks"
    name, fields = tracer.updates[0]
    assert name == "retrieve_chunks"
    chunk = fields["output"]["chunks"][0]
    assert chunk["chunk_id"] == str(retrieved.chunk_id)
    assert chunk["score"] == retrieved.score


# --- summarize --------------------------------------------------------------- #


async def test_summarize_defaults_span_to_unsummarized_read_pages() -> None:
    chunks = _FakeChunks([SimpleNamespace(page_start=21, page_end=50, text="Trials at sea.")])
    summarizer = _FakeSummarizer("They faced trials at sea.")
    context = _make_context(
        progress_service=_FakeProgressService(_progress_row(current_page=50, last_summarized=20)),
        chunks=chunks,
        summarizer=summarizer,
    )
    tool = _tool(context, "summarize")
    out = await tool.ainvoke({"document_id": str(DOC_ID)})

    assert out == "They faced trials at sea."
    # Default span is (last_summarized + 1 .. current_page) == (21 .. 50).
    assert chunks.range_calls[0] == (DOC_ID, 21, 50)
    assert "pages 21 to 50" in summarizer.prompts[0]


async def test_summarize_clamps_page_end_to_current_page_when_spoiler_safe() -> None:
    chunks = _FakeChunks([SimpleNamespace(page_start=21, page_end=50, text="…")])
    context = _make_context(
        progress_service=_FakeProgressService(_progress_row(current_page=50, last_summarized=20)),
        chunks=chunks,
        user_spoiler_safe=True,
    )
    tool = _tool(context, "summarize")
    # LLM asks for pages up to 200; spoiler-safe forces the end back to page 50.
    await tool.ainvoke({"document_id": str(DOC_ID), "page_end": 200})
    assert chunks.range_calls[0] == (DOC_ID, 21, 50)


async def test_summarize_traces_prompt_ref_and_model() -> None:
    chunks = _FakeChunks([SimpleNamespace(page_start=21, page_end=50, text="Trials at sea.")])
    context = _make_context(
        progress_service=_FakeProgressService(_progress_row(current_page=50, last_summarized=20)),
        chunks=chunks,
    )
    tracer = _SpyTracer()
    tool = _tool(context, "summarize", tracer=tracer)

    await tool.ainvoke({"document_id": str(DOC_ID)})

    name, attributes = tracer.opened[0]
    assert name == "summarize"
    assert attributes["prompt"] == "summarize@v1"
    assert "model" in attributes


async def test_summarize_unknown_document() -> None:
    context = _make_context(documents=_FakeDocuments(None))
    tool = _tool(context, "summarize")
    out = await tool.ainvoke({"document_id": str(DOC_ID)})
    assert "isn't in the reader's library" in out


async def test_summarize_accepts_the_full_document_label_as_document_id() -> None:
    # End-to-end regression for the reported bug: the model passed
    # "Title [id: <uuid>]" (get_reading_progress's own label) whole, instead of
    # extracting the uuid — summarize must still recap rather than error out.
    chunks = _FakeChunks([SimpleNamespace(page_start=21, page_end=50, text="Trials at sea.")])
    context = _make_context(
        progress_service=_FakeProgressService(_progress_row(current_page=50, last_summarized=20)),
        chunks=chunks,
    )
    tool = _tool(context, "summarize")
    label = f"2001. Una odisea espacial [id: {DOC_ID}]"
    out = await tool.ainvoke({"document_id": label})
    assert out != f"'{label}' is not a valid document id."
    assert chunks.range_calls[0] == (DOC_ID, 21, 50)


async def test_summarize_nothing_new_to_recap() -> None:
    context = _make_context(
        progress_service=_FakeProgressService(_progress_row(current_page=20, last_summarized=20)),
    )
    tool = _tool(context, "summarize")
    out = await tool.ainvoke({"document_id": str(DOC_ID)})
    assert "nothing new to recap" in out


# --- query_long_term_memory --------------------------------------------------- #


async def test_query_long_term_memory_semantic_search_injects_owner_and_bounds() -> None:
    memory_service = _FakeMemoryService([_memory("likes sci-fi")])
    memories = _FakeMemories()
    context = _make_context(
        memory_service=memory_service,
        memories=memories,
        progress_repo=_FakeProgressRepo(_progress_row(current_page=50)),
    )
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke(
        {"query": "reading tastes", "type": "preference", "document_id": str(DOC_ID)}
    )

    assert "likes sci-fi" in out
    call = memory_service.retrieve_calls[0]
    assert call["memories"] is memories
    assert call["query"] == "reading tastes"
    assert call["type"] is MemoryType.PREFERENCE
    assert call["document_id"] == DOC_ID
    # Targeting a document always bounds the query to the read range (FR-18.3).
    assert call["max_page_end"] == 50


async def test_query_long_term_memory_no_document_is_never_page_bounded() -> None:
    memory_service = _FakeMemoryService([_memory("likes sci-fi")])
    context = _make_context(memory_service=memory_service)
    tool = _tool(context, "query_long_term_memory")
    await tool.ainvoke({"query": "reading tastes"})
    assert memory_service.retrieve_calls[0]["max_page_end"] is None


async def test_query_long_term_memory_no_progress_row_bounds_to_zero() -> None:
    memory_service = _FakeMemoryService([])
    context = _make_context(memory_service=memory_service, progress_repo=_FakeProgressRepo(None))
    tool = _tool(context, "query_long_term_memory")
    await tool.ainvoke({"query": "anything", "document_id": str(DOC_ID)})
    assert memory_service.retrieve_calls[0]["max_page_end"] == 0


async def test_query_long_term_memory_bad_document_id_short_circuits() -> None:
    memory_service = _FakeMemoryService([])
    context = _make_context(memory_service=memory_service)
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke({"query": "x", "document_id": "nope"})
    assert "not a valid document id" in out
    assert memory_service.retrieve_calls == []


async def test_query_long_term_memory_page_range_lookup_uses_covering_when_no_query() -> None:
    covering = [
        _memory("early", type=MemoryType.SUMMARY, document_id=DOC_ID, page_start=1, page_end=20),
        _memory("later", type=MemoryType.SUMMARY, document_id=DOC_ID, page_start=41, page_end=60),
    ]
    memories = _FakeMemories(covering)
    context = _make_context(
        memories=memories, progress_repo=_FakeProgressRepo(_progress_row(current_page=60))
    )
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke({"document_id": str(DOC_ID), "page_start": 30, "page_end": 45})

    assert "later" in out and "early" not in out
    assert memories.covering_calls[0] == {"document_id": DOC_ID, "max_page_end": 60}


async def test_query_long_term_memory_page_lookup_type_mismatch_is_empty() -> None:
    # Document-scoped memories are always summaries; a non-summary type filter
    # can never match, so the deterministic lookup is skipped entirely.
    memories = _FakeMemories([_memory("later", type=MemoryType.SUMMARY, document_id=DOC_ID)])
    context = _make_context(memories=memories)
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke({"document_id": str(DOC_ID), "type": "preference"})

    assert out == "No matching memories were found."
    assert memories.covering_calls == []


async def test_query_long_term_memory_lists_recent_when_no_query_or_document() -> None:
    memory_service = _FakeMemoryService([_memory("likes sci-fi")])
    context = _make_context(memory_service=memory_service)
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke({"type": "habit"})

    assert "likes sci-fi" in out
    call = memory_service.list_calls[0]
    assert call["type"] is MemoryType.HABIT


async def test_query_long_term_memory_formats_summary_with_title_and_pages() -> None:
    memory_service = _FakeMemoryService(
        [
            _memory(
                "They reach Ithaca.",
                type=MemoryType.SUMMARY,
                document_id=DOC_ID,
                page_start=41,
                page_end=60,
            )
        ]
    )
    context = _make_context(memory_service=memory_service)
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke({"query": "recap"})
    assert out == f"[summary] The Odyssey [id: {DOC_ID}] (pp. 41-60): They reach Ithaca."


async def test_query_long_term_memory_formats_non_summary_without_pages() -> None:
    memory_service = _FakeMemoryService([_memory("likes sci-fi", type=MemoryType.PREFERENCE)])
    context = _make_context(memory_service=memory_service)
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke({"query": "preferences"})
    assert out == "[preference] likes sci-fi"


async def test_query_long_term_memory_no_results_message() -> None:
    context = _make_context(memory_service=_FakeMemoryService([]))
    tool = _tool(context, "query_long_term_memory")
    out = await tool.ainvoke({"query": "anything"})
    assert out == "No matching memories were found."


async def test_query_long_term_memory_traces_hits() -> None:
    hit = _memory("likes sci-fi", type=MemoryType.PREFERENCE)
    memory_service = _FakeMemoryService([hit])
    context = _make_context(memory_service=memory_service)
    tracer = _SpyTracer()
    tool = _tool(context, "query_long_term_memory", tracer=tracer)

    await tool.ainvoke({"query": "preferences"})

    assert tracer.opened[0][0] == "query_long_term_memory"
    name, fields = tracer.updates[0]
    assert name == "query_long_term_memory"
    assert fields["output"]["memories"][0]["id"] == str(hit.id)


# --- web_search ---------------------------------------------------------------- #


async def test_web_search_formats_results_for_citation() -> None:
    provider = _FakeWebSearchProvider(
        [SearchResult(title="A Great Book", url="http://x", snippet="A snippet.")]
    )
    context = _make_context(web_search_provider=provider)
    tool = _tool(context, "web_search")

    out = await tool.ainvoke({"query": "books like Dune"})

    assert out == "[1] A Great Book (http://x):\nA snippet."
    assert provider.calls == [{"query": "books like Dune", "count": 5}]


async def test_web_search_respects_count_argument() -> None:
    provider = _FakeWebSearchProvider([])
    context = _make_context(web_search_provider=provider)
    tool = _tool(context, "web_search")

    await tool.ainvoke({"query": "x", "count": 3})

    assert provider.calls == [{"query": "x", "count": 3}]


async def test_web_search_no_results_message() -> None:
    context = _make_context(web_search_provider=_FakeWebSearchProvider([]))
    tool = _tool(context, "web_search")
    out = await tool.ainvoke({"query": "nothing findable"})
    assert out == "No web results were found."


async def test_web_search_traces_query_and_results() -> None:
    provider = _FakeWebSearchProvider([SearchResult(title="T", url="http://x", snippet="s")])
    context = _make_context(web_search_provider=provider)
    tracer = _SpyTracer()
    tool = _tool(context, "web_search", tracer=tracer)

    await tool.ainvoke({"query": "q"})

    name, attributes = tracer.opened[0]
    assert name == "web_search"
    assert attributes["query"] == "q"
    _, fields = tracer.updates[0]
    assert fields["output"]["results"] == [{"title": "T", "url": "http://x"}]


# --- recommend: internal path (ungated) ---------------------------------------- #


async def test_recommend_internal_only_formats_library_recommendations() -> None:
    recs = [
        Recommendation(
            document_id=DOC_ID,
            title="The Iliad",
            author="Homer",
            reason="Because you read The Odyssey",
            score=0.8,
        )
    ]
    service = _FakeRecommendationService(internal=recs)
    context = _make_context(recommendation_service=service)
    tool = _tool(context, "recommend")

    out = await tool.ainvoke({})

    assert out == "[1] The Iliad by Homer — Because you read The Odyssey"
    call = service.recommend_from_library_calls[0]
    assert call["user_id"] == USER_ID
    assert call["limit"] == 5
    # The external path never runs when include_web is left at its default.
    assert service.recommend_from_web_calls == []
    assert service.default_web_query_calls == []


async def test_recommend_no_recommendations_message() -> None:
    context = _make_context(recommendation_service=_FakeRecommendationService(internal=[]))
    tool = _tool(context, "recommend")
    out = await tool.ainvoke({})
    assert "No recommendations yet" in out


async def test_recommend_respects_limit_argument() -> None:
    service = _FakeRecommendationService()
    context = _make_context(recommendation_service=service)
    tool = _tool(context, "recommend")

    await tool.ainvoke({"limit": 2})

    assert service.recommend_from_library_calls[0]["limit"] == 2


async def test_recommend_include_web_with_no_history_skips_without_pausing() -> None:
    # default_web_query returning None means there's nothing to search the web
    # from — this must return a message, not reach interrupt() (which would
    # raise outside a real graph context).
    service = _FakeRecommendationService(internal=[], default_query=None)
    context = _make_context(recommendation_service=service)
    tool = _tool(context, "recommend")

    out = await tool.ainvoke({"include_web": True})

    assert "no reading history yet" in out.lower()
    assert service.recommend_from_web_calls == []


async def test_recommend_traces_internal_and_external_counts() -> None:
    service = _FakeRecommendationService(internal=[Recommendation(title="A", reason="r")])
    context = _make_context(recommendation_service=service)
    tracer = _SpyTracer()
    tool = _tool(context, "recommend", tracer=tracer)

    await tool.ainvoke({})

    name, attributes = tracer.opened[0]
    assert name == "recommend"
    assert attributes["include_web"] is False
    _, fields = tracer.updates[0]
    assert fields["output"] == {"internal": 1, "external": 0}
