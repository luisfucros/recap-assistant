"""Unit tests for the parts of EvaluationService that don't need real infra.

``_ensure_fixtures``/``_run_case``/``run_evaluation`` construct real
``DocumentRepository``/``ChunkRepository`` objects bound to a real DB session
and upsert real Qdrant vectors — that orchestration is exercised end-to-end
against real Postgres/Qdrant in the integration tier instead of being faked
here layer by layer, which would just be testing the fakes. What's genuinely
pure — the run-level aggregation, the deterministic-id scheme, and the
get-or-create system user — is unit-tested directly.
"""

import uuid
from types import SimpleNamespace

import pytest
from api.evaluation.datasets.loader import load_dataset
from api.services.evaluation_service import (
    _EVAL_SYSTEM_EMAIL,
    EvaluationService,
    _deterministic_id,
    _summarize,
)

from shared.core.config import Settings
from shared.core.enums import EvaluationRunStatus
from shared.core.errors import NotFoundError
from shared.models.user import User

pytestmark = pytest.mark.unit


class _FakeTracer:
    def span(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, **kwargs):
        pass


class _FakeUserRepository:
    def __init__(self) -> None:
        self._by_id: dict[uuid.UUID, User] = {}
        self.add_calls = 0

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return self._by_id.get(user_id)

    async def add(self, user: User) -> User:
        self.add_calls += 1
        self._by_id[user.id] = user
        return user


def _service(*, enqueue=None) -> EvaluationService:
    """An ``EvaluationService`` with placeholder collaborators.

    Only ``_ensure_eval_user``/``_summarize``-adjacent behavior is under test
    in this file, none of which touches these — they only need to exist to
    satisfy the constructor.
    """
    return EvaluationService(
        agent_service=SimpleNamespace(),
        retrieval_service=SimpleNamespace(),
        progress_service=SimpleNamespace(),
        memory_service=SimpleNamespace(),
        recommendation_service=SimpleNamespace(),
        usage_service=SimpleNamespace(),
        web_search=lambda: SimpleNamespace(),
        summarizer=SimpleNamespace(),
        embedder=SimpleNamespace(),
        vector_store=SimpleNamespace(),
        judge_model=SimpleNamespace(),
        prompts=SimpleNamespace(get=lambda _name: SimpleNamespace(ref="generate@v5")),
        tracer=_FakeTracer(),
        settings=Settings(_env_file=None, jwt_secret="test-secret"),
        enqueue=enqueue,
    )


class TestDeterministicId:
    def test_is_stable_for_the_same_input(self) -> None:
        assert _deterministic_id("a", "b") == _deterministic_id("a", "b")

    def test_differs_for_different_input(self) -> None:
        assert _deterministic_id("a", "b") != _deterministic_id("a", "c")


class TestEnsureEvalUser:
    async def test_creates_the_system_user_on_first_call(self) -> None:
        service = _service()
        users = _FakeUserRepository()

        user = await service._ensure_eval_user(users)

        assert user.email == _EVAL_SYSTEM_EMAIL
        assert user.is_admin is False
        assert users.add_calls == 1

    async def test_reuses_the_existing_system_user_on_a_later_call(self) -> None:
        service = _service()
        users = _FakeUserRepository()

        first = await service._ensure_eval_user(users)
        second = await service._ensure_eval_user(users)

        assert first.id == second.id
        assert users.add_calls == 1


class TestSummarize:
    def test_summarizes_zero_cases_to_all_zeros(self) -> None:
        summary = _summarize([])

        assert summary["cases"] == 0
        assert summary["retrieval"] == {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0}
        assert summary["answer_quality"]["citation_ok_rate"] == 0.0

    def test_averages_retrieval_scores_across_cases(self) -> None:
        cases = [
            {
                "retrieval": {"hit_rate": 1.0, "recall": 1.0, "mrr": 1.0},
                "blocked": False,
                "interrupted": False,
                "answer_quality": None,
            },
            {
                "retrieval": {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0},
                "blocked": False,
                "interrupted": False,
                "answer_quality": None,
            },
        ]

        summary = _summarize(cases)

        assert summary["cases"] == 2
        assert summary["retrieval"] == {"hit_rate": 0.5, "recall": 0.5, "mrr": 0.5}

    def test_averages_answer_quality_only_over_judged_cases(self) -> None:
        cases = [
            {
                "retrieval": {"hit_rate": 1.0, "recall": 1.0, "mrr": 1.0},
                "blocked": False,
                "interrupted": False,
                "answer_quality": {"faithfulness": 1.0, "relevance": 1.0, "citation_ok": True},
            },
            {
                "retrieval": {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0},
                "blocked": True,
                "interrupted": False,
                "answer_quality": None,
            },
        ]

        summary = _summarize(cases)

        # Only the one judged case counts toward answer-quality averages.
        assert summary["answer_quality"]["faithfulness"] == 1.0
        assert summary["answer_quality"]["citation_ok_rate"] == 1.0
        assert summary["blocked"] == 1
        assert summary["interrupted"] == 0

    def test_counts_interrupted_cases_separately_from_blocked(self) -> None:
        cases = [
            {
                "retrieval": {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0},
                "blocked": False,
                "interrupted": True,
                "answer_quality": None,
            }
        ]

        summary = _summarize(cases)

        assert summary["interrupted"] == 1
        assert summary["blocked"] == 0


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


class _FakeRuns:
    def __init__(self) -> None:
        self.added: list = []
        self.by_id: dict = {}

    async def add(self, run):
        if run.id is None:
            run.id = uuid.uuid4()
        self.added.append(run)
        self.by_id[run.id] = run
        return run

    async def get_or_404(self, run_id: uuid.UUID):
        run = self.by_id.get(run_id)
        if run is None:
            raise NotFoundError()
        return run


class TestEnqueueEvaluation:
    async def test_persists_pending_and_dispatches_without_running_the_agent(self) -> None:
        dispatched: list[uuid.UUID] = []
        agent = SimpleNamespace(run=None)

        def _run(*_a, **_k):
            raise AssertionError("agent must not run on enqueue")

        agent.run = _run
        service = _service(enqueue=dispatched.append)
        service._agent_service = agent
        session = _FakeSession()
        runs = _FakeRuns()

        run = await service.enqueue_evaluation(dataset_name="sample_v1", session=session, runs=runs)

        assert run.status is EvaluationRunStatus.PENDING
        assert run.dataset_name == "sample_v1"
        assert session.commits == 1
        assert dispatched == [run.id]
        assert len(runs.added) == 1

    async def test_unknown_dataset_404s_with_no_row(self) -> None:
        dispatched: list[uuid.UUID] = []
        service = _service(enqueue=dispatched.append)
        session = _FakeSession()
        runs = _FakeRuns()

        with pytest.raises(NotFoundError):
            await service.enqueue_evaluation(
                dataset_name="does-not-exist", session=session, runs=runs
            )

        assert runs.added == []
        assert dispatched == []
        assert session.commits == 0


class TestExecuteEvaluation:
    async def test_scores_a_pending_run(self) -> None:
        service = _service()
        session = _FakeSession()
        runs = _FakeRuns()
        pending = await service.enqueue_evaluation(
            dataset_name="sample_v1", session=session, runs=runs, dispatch=False
        )

        async def _fake_dataset(*_a, **_k):
            return (
                [
                    {
                        "case_id": "c1",
                        "retrieval": {"hit_rate": 1.0, "recall": 1.0, "mrr": 1.0},
                        "answer_quality": None,
                        "blocked": False,
                        "interrupted": False,
                    }
                ],
                {"cases": 1},
            )

        service._run_dataset = _fake_dataset  # type: ignore[method-assign]

        result = await service.execute_evaluation(
            run_id=pending.id,
            session=session,
            users=_FakeUserRepository(),
            runs=runs,
        )

        assert result.status is EvaluationRunStatus.COMPLETED
        assert result.summary == {"cases": 1}
        assert result.results["cases"][0]["case_id"] == "c1"

    async def test_terminal_run_is_a_noop(self) -> None:
        service = _service()
        session = _FakeSession()
        runs = _FakeRuns()
        pending = await service.enqueue_evaluation(
            dataset_name="sample_v1", session=session, runs=runs, dispatch=False
        )
        pending.status = EvaluationRunStatus.COMPLETED
        pending.summary = {"cases": 3}

        async def _must_not_run(*_a, **_k):
            raise AssertionError("must not score a terminal run")

        service._run_dataset = _must_not_run  # type: ignore[method-assign]

        result = await service.execute_evaluation(
            run_id=pending.id,
            session=session,
            users=_FakeUserRepository(),
            runs=runs,
        )

        assert result.status is EvaluationRunStatus.COMPLETED
        assert result.summary == {"cases": 3}


class TestEnsureFixtures:
    async def test_creates_the_chunks_collection_before_seeding(self) -> None:
        """Eval is often the first writer; Qdrant 404s if the collection is missing."""
        from unittest.mock import AsyncMock

        store = SimpleNamespace(ensure_collection=AsyncMock())
        service = _service()
        service._vector_store = store
        documents = SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(title=None, author=None, language=None)),
            user_id=uuid.uuid4(),
        )
        chunks = SimpleNamespace(list_by_document=AsyncMock(return_value=[]))
        session = SimpleNamespace(commit=AsyncMock())

        await service._ensure_fixtures(
            load_dataset("sample_v1"),
            session=session,
            documents=documents,
            chunks=chunks,
        )

        store.ensure_collection.assert_awaited_once()
        assert documents.get.await_count >= 1
