"""EvaluationService — runs a versioned dataset through retrieval + the agent,
scores it, and persists the run (FR-12).

Each case in a dataset (:mod:`api.evaluation.datasets.loader`) asks a question
about a small, self-contained fixture "document" the dataset carries as plain
text. A run seeds that fixture as real ``Document``/``Chunk`` rows and real
vectors (once — subsequent runs of the same dataset version reuse them,
looked up by deterministic id), all owned by a dedicated system user rather
than any real reader's account, so retrieval and the agent's answer are
genuine, not simulated, while never touching real user data. Each case is then
scored two ways: retrieval quality (hit rate/recall/MRR of the expected
chunks) and, unless the turn was blocked or paused on a HITL interrupt,
answer quality via an LLM-as-judge (faithfulness/relevance/citation
correctness). The run — every case's scores plus the run-level aggregate, and
the prompt/model/embedding identifiers it ran with — is persisted so two runs
are comparable across a prompt or provider change (FR-12.3).
"""

import hashlib
import uuid
from collections.abc import Callable
from typing import Any
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from api.agent.context import build_tool_context
from api.evaluation.datasets.loader import EvalCase, EvalDataset, load_dataset
from api.evaluation.scorers.judge import EvaluationJudgment
from api.evaluation.scorers.retrieval import hit_rate, mean_reciprocal_rank, recall
from api.llm import model_id_for
from api.services.agent_service import AgentService
from api.services.memory_service import MemoryService
from api.services.progress_service import ProgressService
from api.services.recommendation_service import RecommendationService
from api.services.retrieval_service import RetrievalService
from api.services.usage_service import UsageService
from shared.core.config import Settings
from shared.core.enums import DocumentFormat, DocumentStatus, EvaluationRunStatus
from shared.models.document import Chunk, Document
from shared.models.evaluation import EvaluationRun
from shared.models.user import User
from shared.observability.tracing import Tracer
from shared.prompt import PromptRegistry
from shared.providers import Embedder, WebSearchProvider
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    EvaluationRunRepository,
    ReadingEventRepository,
    ReadingProgressRepository,
    UserRepository,
)
from shared.vectorstore import ChunkVectorStore, build_chunk_payload, chunk_point_id

# Fixed namespace for every id this service derives deterministically (the eval
# system user, and each fixture document/chunk) — arbitrary but constant, so
# ids are stable across processes and re-runs without a lookup table.
_NAMESPACE = uuid.UUID("6f9c1e2a-2b3c-4d5e-8f9a-0b1c2d3e4f5a")
_EVAL_SYSTEM_EMAIL = "eval-system@recap.internal"


def _deterministic_id(*parts: str) -> UUID:
    return uuid.uuid5(_NAMESPACE, ":".join(parts))


class EvaluationService:
    """Orchestrates one dataset run: seed fixtures, run cases, score, persist."""

    def __init__(
        self,
        *,
        agent_service: AgentService,
        retrieval_service: RetrievalService,
        progress_service: ProgressService,
        memory_service: MemoryService,
        recommendation_service: RecommendationService,
        usage_service: UsageService,
        web_search: Callable[[], WebSearchProvider],
        summarizer: BaseChatModel,
        embedder: Embedder,
        vector_store: ChunkVectorStore,
        judge_model: Runnable,
        prompts: PromptRegistry,
        tracer: Tracer,
        settings: Settings,
    ) -> None:
        self._agent_service = agent_service
        self._retrieval_service = retrieval_service
        self._progress_service = progress_service
        self._memory_service = memory_service
        self._recommendation_service = recommendation_service
        self._usage_service = usage_service
        self._web_search = web_search
        self._summarizer = summarizer
        self._embedder = embedder
        self._vector_store = vector_store
        self._judge_model = judge_model
        self._prompts = prompts
        self._tracer = tracer
        self._settings = settings

    async def run_evaluation(
        self,
        *,
        dataset_name: str,
        session: AsyncSession,
        users: UserRepository,
        runs: EvaluationRunRepository,
        triggered_by: UUID | None = None,
    ) -> EvaluationRun:
        """Load a dataset by name, run every case, score it, and persist the run.

        A failure partway through (an LLM/embedding call exhausting its
        fallbacks, a dataset fixture that doesn't seed cleanly, ...) still
        yields a persisted, fetchable :class:`EvaluationRun` — ``status``
        ``FAILED`` and ``error`` set — rather than losing the attempt, mirroring
        the ingestion pipeline's terminal-failure recording.

        Raises:
            NotFoundError: ``dataset_name`` doesn't match a shipped dataset (no
                run is persisted for this — there is nothing to have attempted).
        """
        dataset = load_dataset(dataset_name)
        logger.info("evaluation.run: started dataset {}@{}", dataset.name, dataset.version)
        prompt_ref = self._prompts.get("generate").ref
        llm_model = model_id_for(self._settings, "default")

        # Tagged by prompt/model/embedding version (FR-12.3) so two runs — e.g.
        # before/after a prompt or provider change — are comparable in Langfuse
        # (a no-op when tracing is disabled, per the observability split).
        with self._tracer.span(
            "evaluation_run",
            dataset=dataset.name,
            version=dataset.version,
            prompt_version=prompt_ref,
            llm_provider=self._settings.llm_provider,
            llm_model=llm_model,
            embedding_model=self._settings.embedding_model,
        ) as span:
            try:
                case_results, summary = await self._run_dataset(
                    dataset, users=users, session=session
                )
                status, error = EvaluationRunStatus.COMPLETED, None
            except Exception as exc:
                logger.opt(exception=exc).error(
                    "evaluation.run: failed dataset {}@{}", dataset.name, dataset.version
                )
                await session.rollback()
                case_results, summary = [], _summarize([])
                status, error = EvaluationRunStatus.FAILED, str(exc)[:2048]
            span.update(status=status.value, summary=summary)

        run = EvaluationRun(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            status=status,
            prompt_version=prompt_ref,
            llm_provider=self._settings.llm_provider,
            llm_model=llm_model,
            embedding_model=self._settings.embedding_model,
            results={"cases": case_results},
            summary=summary,
            error=error,
            triggered_by=triggered_by,
        )
        await runs.add(run)
        await session.commit()
        logger.info(
            "evaluation.run: persisted status={} cases={}",
            status.value,
            summary.get("cases"),
        )
        return run

    async def _run_dataset(
        self, dataset: EvalDataset, *, users: UserRepository, session: AsyncSession
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        eval_user = await self._ensure_eval_user(users)
        documents = DocumentRepository(session, eval_user.id)
        chunks = ChunkRepository(session, eval_user.id)
        progress_repo = ReadingProgressRepository(session, eval_user.id)
        events_repo = ReadingEventRepository(session, eval_user.id)

        fixtures = await self._ensure_fixtures(
            dataset, session=session, documents=documents, chunks=chunks
        )
        case_results = [
            await self._run_case(
                case,
                eval_user=eval_user,
                fixtures=fixtures,
                session=session,
                documents=documents,
                chunks=chunks,
                progress_repo=progress_repo,
                events_repo=events_repo,
            )
            for case in dataset.cases
        ]
        return case_results, _summarize(case_results)

    async def _ensure_eval_user(self, users: UserRepository) -> User:
        """Get or create the dedicated system user that owns every eval fixture.

        A fixed, deterministic id/email rather than one created per run: the
        isolation invariant requires a real ``users.id`` to own fixture
        documents/chunks, but eval fixtures are not personal data, so a single
        stable account (never exposed over any user-facing surface) is enough.
        """
        user_id = _deterministic_id("eval-system-user")
        existing = await users.get_by_id(user_id)
        if existing is not None:
            return existing
        return await users.add(
            User(
                id=user_id,
                email=_EVAL_SYSTEM_EMAIL,
                hashed_password=None,
                is_admin=False,
                spoiler_safe=False,
            )
        )

    async def _ensure_fixtures(
        self,
        dataset: EvalDataset,
        *,
        session: AsyncSession,
        documents: DocumentRepository,
        chunks: ChunkRepository,
    ) -> dict[str, tuple[UUID, list[UUID]]]:
        """Seed each dataset document as real rows + vectors, once per version.

        Returns a mapping of document key -> ``(document_id, chunk_ids)``, chunk
        ids ordered by ``ordinal`` so a case's ``expected_chunk_ordinals`` index
        into it directly.
        """
        fixtures: dict[str, tuple[UUID, list[UUID]]] = {}
        for doc in dataset.documents:
            document_id = _deterministic_id(dataset.name, dataset.version, doc.key)
            existing_doc = await documents.get(document_id)
            if existing_doc is not None:
                existing_chunks = await chunks.list_by_document(document_id)
                fixtures[doc.key] = (document_id, [c.id for c in existing_chunks])
                continue

            content = "\n".join(c.text for c in doc.chunks)
            await documents.add(
                Document(
                    id=document_id,
                    user_id=documents.user_id,
                    title=doc.title,
                    author=doc.author,
                    filename=f"{doc.key}.txt",
                    object_key=f"eval/{dataset.name}/{dataset.version}/{doc.key}",
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    format=DocumentFormat.PDF,
                    language=doc.language,
                    status=DocumentStatus.INDEXED,
                    embed_model=self._settings.embedding_model,
                )
            )

            chunk_rows = [
                Chunk(
                    id=_deterministic_id(dataset.name, dataset.version, doc.key, str(ordinal)),
                    document_id=document_id,
                    user_id=documents.user_id,
                    ordinal=ordinal,
                    page_start=fixture_chunk.page,
                    page_end=fixture_chunk.page,
                    chapter=fixture_chunk.chapter,
                    text=fixture_chunk.text,
                    content_hash=hashlib.sha256(fixture_chunk.text.encode()).hexdigest(),
                )
                for ordinal, fixture_chunk in enumerate(doc.chunks)
            ]
            for chunk in chunk_rows:
                chunk.vector_id = chunk_point_id(chunk.id)
            await chunks.add_many(chunk_rows)
            await session.flush()

            vectors = await self._embedder.embed([c.text for c in chunk_rows])
            await self._vector_store.upsert(
                ids=[c.vector_id for c in chunk_rows],
                vectors=vectors,
                payloads=[
                    build_chunk_payload(
                        c, title=doc.title, author=doc.author, language=doc.language
                    )
                    for c in chunk_rows
                ],
            )
            fixtures[doc.key] = (document_id, [c.id for c in chunk_rows])

        await session.commit()
        return fixtures

    async def _run_case(
        self,
        case: EvalCase,
        *,
        eval_user: User,
        fixtures: dict[str, tuple[UUID, list[UUID]]],
        session: AsyncSession,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        progress_repo: ReadingProgressRepository,
        events_repo: ReadingEventRepository,
    ) -> dict[str, Any]:
        document_id, chunk_ids = fixtures[case.document]
        expected_ids = {
            str(chunk_ids[ordinal])
            for ordinal in case.expected_chunk_ordinals
            if ordinal < len(chunk_ids)
        }

        include_unread = case.current_page is None
        if not include_unread:
            await self._progress_service.record_position(
                session=session,
                documents=documents,
                progress=progress_repo,
                events=events_repo,
                document_id=document_id,
                current_page=case.current_page,
            )

        retrieval = await self._retrieval_service.retrieve(
            query=case.query,
            user_id=eval_user.id,
            progress=progress_repo,
            chunks=chunks,
            document_id=document_id,
            include_unread=include_unread,
            user_spoiler_safe=eval_user.spoiler_safe,
        )
        retrieved_ids = [str(c.chunk_id) for c in retrieval.chunks]
        retrieval_scores = {
            "hit_rate": hit_rate(retrieved_ids, expected_ids),
            "recall": recall(retrieved_ids, expected_ids),
            "mrr": mean_reciprocal_rank(retrieved_ids, expected_ids),
        }

        tool_context = build_tool_context(
            session=session,
            user=eval_user,
            progress_service=self._progress_service,
            retrieval_service=self._retrieval_service,
            summarizer=self._summarizer,
            prompts=self._prompts,
            memory_service=self._memory_service,
            recommendation_service=self._recommendation_service,
            web_search=self._web_search,
            usage_service=self._usage_service,
        )
        turn = await self._agent_service.run(
            tool_context=tool_context,
            display_name="Evaluator",
            message=case.query,
            conversation_id=f"eval:{uuid.uuid4()}",
        )

        result: dict[str, Any] = {
            "case_id": case.id,
            "retrieval": retrieval_scores,
            "answer": turn.answer,
            "blocked": turn.blocked,
            "interrupted": turn.interrupted,
            "answer_quality": None,
        }
        if not turn.blocked and not turn.interrupted:
            context = "\n\n".join(c.text for c in retrieval.chunks)
            judgment = await self._judge_answer(
                query=case.query,
                context=context,
                reference_answer=case.reference_answer,
                answer=turn.answer,
            )
            result["answer_quality"] = judgment.model_dump()
        return result

    async def _judge_answer(
        self, *, query: str, context: str, reference_answer: str, answer: str
    ) -> EvaluationJudgment:
        prompt = self._prompts.get("evaluation_judge").render(
            query=query, context=context, reference_answer=reference_answer, answer=answer
        )
        with self._tracer.span("evaluation_judge"):
            return await self._judge_model.ainvoke(prompt)


def _summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Average each metric across cases; a run with zero cases scores all zeros."""
    n = len(case_results)
    if n == 0:
        return {
            "cases": 0,
            "retrieval": {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0},
            "answer_quality": {"faithfulness": 0.0, "relevance": 0.0, "citation_ok_rate": 0.0},
            "blocked": 0,
            "interrupted": 0,
        }

    retrieval_totals = {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0}
    judged = [r["answer_quality"] for r in case_results if r["answer_quality"] is not None]
    blocked = sum(1 for r in case_results if r["blocked"])
    interrupted = sum(1 for r in case_results if r["interrupted"])

    for r in case_results:
        for key in retrieval_totals:
            retrieval_totals[key] += r["retrieval"][key]

    answer_quality = {
        "faithfulness": sum(j["faithfulness"] for j in judged) / len(judged) if judged else 0.0,
        "relevance": sum(j["relevance"] for j in judged) / len(judged) if judged else 0.0,
        "citation_ok_rate": (sum(1 for j in judged if j["citation_ok"]) / len(judged))
        if judged
        else 0.0,
    }
    return {
        "cases": n,
        "retrieval": {key: total / n for key, total in retrieval_totals.items()},
        "answer_quality": answer_quality,
        "blocked": blocked,
        "interrupted": interrupted,
    }
