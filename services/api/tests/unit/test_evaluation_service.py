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
from api.services.evaluation_service import (
    _EVAL_SYSTEM_EMAIL,
    EvaluationService,
    _deterministic_id,
    _summarize,
)

from shared.core.config import Settings
from shared.models.user import User

pytestmark = pytest.mark.unit


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


def _service() -> EvaluationService:
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
        prompts=SimpleNamespace(),
        tracer=SimpleNamespace(),
        settings=Settings(_env_file=None, jwt_secret="test-secret"),
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
