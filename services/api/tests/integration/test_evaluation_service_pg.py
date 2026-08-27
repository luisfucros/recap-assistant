"""Integration test for EvaluationService against real Postgres + Qdrant (FR-12).

Runs the shipped ``sample_v1`` dataset end-to-end: fixture documents/chunks are
seeded as real rows and real vectors, retrieval genuinely searches Qdrant and
hydrates from Postgres, and the run is persisted and re-fetchable — only the
agent turn and the LLM-as-judge are faked, since a real LLM call is an external
boundary the integration tier mocks like everywhere else in this suite.
"""

import uuid
from types import SimpleNamespace

import pytest
from api.evaluation.scorers.judge import EvaluationJudgment
from api.services.agent_service import AgentTurn
from api.services.evaluation_service import EvaluationService
from api.services.progress_service import ProgressService
from api.services.retrieval_service import RetrievalService

from shared.core.enums import EvaluationRunStatus
from shared.core.errors import NotFoundError
from shared.observability.tracing import NoOpTracer
from shared.prompt import get_prompt_registry
from shared.repositories import EvaluationRunRepository, UserRepository
from shared.vectorstore import ChunkVectorStore

pytestmark = pytest.mark.integration


class _FakeAgentService:
    """Returns one canned, non-blocked answer regardless of the turn's args."""

    async def run(self, **kwargs) -> AgentTurn:
        return AgentTurn(answer="A canned but on-topic answer.")


class _FakeJudgeModel:
    """Returns one canned judgment — the real LLM-as-judge call is an external boundary."""

    async def ainvoke(self, prompt: str) -> EvaluationJudgment:
        return EvaluationJudgment(faithfulness=0.9, relevance=0.8, citation_ok=True)


def _service(qdrant_client, test_settings, embedder) -> EvaluationService:
    store = ChunkVectorStore(
        qdrant_client, collection=test_settings.qdrant_chunks_collection, dim=8
    )
    return EvaluationService(
        agent_service=_FakeAgentService(),
        retrieval_service=RetrievalService(
            embedder=embedder, vector_store=store, settings=test_settings
        ),
        progress_service=ProgressService(),
        memory_service=SimpleNamespace(),
        recommendation_service=SimpleNamespace(),
        usage_service=SimpleNamespace(),
        web_search=lambda: SimpleNamespace(),
        summarizer=SimpleNamespace(),
        embedder=embedder,
        vector_store=store,
        judge_model=_FakeJudgeModel(),
        prompts=get_prompt_registry(),
        tracer=NoOpTracer(),
        settings=test_settings,
    )


async def test_runs_the_sample_dataset_end_to_end_and_persists_scores(
    db_sessionmaker, qdrant_client, test_settings, fake_embedder
) -> None:
    store = ChunkVectorStore(
        qdrant_client, collection=test_settings.qdrant_chunks_collection, dim=8
    )
    await store.ensure_collection()
    service = _service(qdrant_client, test_settings, fake_embedder)

    async with db_sessionmaker() as session:
        pending = await service.enqueue_evaluation(
            dataset_name="sample_v1",
            session=session,
            runs=EvaluationRunRepository(session),
            dispatch=False,
        )
        run = await service.execute_evaluation(
            run_id=pending.id,
            session=session,
            users=UserRepository(session),
            runs=EvaluationRunRepository(session),
        )

    assert run.status is EvaluationRunStatus.COMPLETED
    assert run.dataset_name == "sample_v1"
    assert run.error is None
    cases = run.results["cases"]
    assert len(cases) == 3
    for case in cases:
        assert set(case["retrieval"]) == {"hit_rate", "recall", "mrr"}
        assert case["answer_quality"] == {
            "faithfulness": 0.9,
            "relevance": 0.8,
            "citation_ok": True,
            "reasoning": "",
        }
    assert run.summary["cases"] == 3

    # The persisted run is independently fetchable by id (the GET route's path).
    async with db_sessionmaker() as session:
        fetched = await EvaluationRunRepository(session).get(run.id)
    assert fetched is not None
    assert fetched.id == run.id


async def test_a_second_run_reuses_the_same_seeded_fixtures(
    db_sessionmaker, qdrant_client, test_settings, fake_embedder
) -> None:
    store = ChunkVectorStore(
        qdrant_client, collection=test_settings.qdrant_chunks_collection, dim=8
    )
    await store.ensure_collection()
    service = _service(qdrant_client, test_settings, fake_embedder)

    async with db_sessionmaker() as session:
        await service.run_evaluation(
            dataset_name="sample_v1",
            session=session,
            users=UserRepository(session),
            runs=EvaluationRunRepository(session),
        )
    async with db_sessionmaker() as session:
        second = await service.run_evaluation(
            dataset_name="sample_v1",
            session=session,
            users=UserRepository(session),
            runs=EvaluationRunRepository(session),
        )

    assert second.status is EvaluationRunStatus.COMPLETED
    assert len(second.results["cases"]) == 3


async def test_run_evaluation_raises_not_found_for_an_unknown_dataset(
    db_sessionmaker, qdrant_client, test_settings, fake_embedder
) -> None:

    service = _service(qdrant_client, test_settings, fake_embedder)

    async with db_sessionmaker() as session:
        with pytest.raises(NotFoundError):
            await service.enqueue_evaluation(
                dataset_name="does-not-exist-" + uuid.uuid4().hex,
                session=session,
                runs=EvaluationRunRepository(session),
                dispatch=False,
            )
