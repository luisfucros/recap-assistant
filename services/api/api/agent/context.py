"""Per-turn, user-scoped context the agent's tools close over.

This is the load-bearing seam for the isolation invariant on the tool side. The
agent's tools take their **semantic** arguments from the LLM (a query, a document
id, a page range) but take their **owner** — the ``user_id`` — and their data
handles (user-scoped repositories, the request's services) from *here*, a context
assembled server-side from the authenticated request. The LLM never supplies, and
cannot widen, the scope: there is no ``user_id`` tool argument to spoof.

``build_agent_tools`` (in :mod:`api.agent.tools`) binds one fresh
:class:`ToolContext` per turn and returns tools that reference only it, so a tool
call can only ever touch the caller's own reading data.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.memory_service import MemoryService
from api.services.progress_service import ProgressService
from api.services.recommendation_service import RecommendationService
from api.services.retrieval_service import RetrievalService
from api.services.usage_service import UsageService
from shared.models.user import User
from shared.prompt import PromptRegistry
from shared.providers.base import WebSearchProvider
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    LongTermMemoryRepository,
    ReadingEventRepository,
    ReadingProgressRepository,
    UsageEventRepository,
)


@dataclass(slots=True)
class ToolContext:
    """The user-scoped handles and services one turn's tools operate through.

    Every field is derived server-side from the authenticated request. The
    repositories are already bound to the owner ``user_id`` at construction (they
    are :class:`~shared.repositories.base.UserScopedRepository` instances), and
    ``user_id`` is carried explicitly for the vector-search filter the retrieval
    service injects. ``session``/``events``/``memories``/``memory_service`` are the
    write-path handles: the ask-when-missing recap flow (FR-4.7) records the
    position the reader answers with, and ``persist_memory`` (FR-4.6) saves the
    confirmed page-range summary and commits both. ``web_search`` is a zero-arg
    factory, not a built provider — constructing it can raise if the configured
    provider's API key is unset, so it's deferred until a call is actually
    approved and about to run (the ``web_search`` tool, and ``recommend``'s
    external branch), the same way the multimodal normalizer defers building
    the transcription/vision providers until a turn actually carries media.
    ``usage``/``usage_service`` record the turn's per-user token spend and
    tool-call counts (NFR-13) — the durable counterpart to the low-cardinality
    Prometheus metrics, which can't carry a ``user_id`` label.
    """

    user_id: uuid.UUID
    documents: DocumentRepository
    chunks: ChunkRepository
    progress_repo: ReadingProgressRepository
    progress_service: ProgressService
    retrieval_service: RetrievalService
    summarizer: BaseChatModel
    prompts: PromptRegistry
    user_spoiler_safe: bool
    session: AsyncSession
    events: ReadingEventRepository
    memories: LongTermMemoryRepository
    memory_service: MemoryService
    recommendation_service: RecommendationService
    web_search: Callable[[], WebSearchProvider]
    usage: UsageEventRepository
    usage_service: UsageService


def build_tool_context(
    *,
    session: AsyncSession,
    user: User,
    progress_service: ProgressService,
    retrieval_service: RetrievalService,
    summarizer: BaseChatModel,
    prompts: PromptRegistry,
    memory_service: MemoryService,
    recommendation_service: RecommendationService,
    web_search: Callable[[], WebSearchProvider],
    usage_service: UsageService,
) -> ToolContext:
    """Assemble one turn's :class:`ToolContext` by hand.

    HTTP routes get a :class:`ToolContext` from FastAPI's per-request
    ``api.deps.get_tool_context`` dependency chain; two callers can't use that
    chain and build the same user-scoped handles here instead, bound to their
    own session: the ``/chat/ws`` route (one WebSocket connection carries many
    turns, so a single request-scoped session would outlive any one of them)
    and ``EvaluationService`` (which runs turns for a fixture system user,
    never the authenticated caller). Lives here rather than in ``api.deps`` so
    ``EvaluationService`` (built inside ``Resources``) can import it without a
    ``deps`` -> ``Resources`` -> ``deps`` circular import.
    """
    return ToolContext(
        user_id=user.id,
        documents=DocumentRepository(session, user.id),
        chunks=ChunkRepository(session, user.id),
        progress_repo=ReadingProgressRepository(session, user.id),
        progress_service=progress_service,
        retrieval_service=retrieval_service,
        summarizer=summarizer,
        prompts=prompts,
        user_spoiler_safe=user.spoiler_safe,
        session=session,
        events=ReadingEventRepository(session, user.id),
        memories=LongTermMemoryRepository(session, user.id),
        memory_service=memory_service,
        recommendation_service=recommendation_service,
        web_search=web_search,
        usage=UsageEventRepository(session, user.id),
        usage_service=usage_service,
    )
