"""Functional tests for the evaluation routes (FR-12), admin-only.

The evaluation service/run repository are faked at the boundary — only the
routes' HTTP contract (admin gating, request/response shape, 404) is under
test here; the real orchestration is covered in `test_evaluation_service.py`
(unit) and `test_evaluation_service_pg.py` (integration).
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from api.deps import get_evaluation_run_repository, get_evaluation_service
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

from shared.core.enums import EvaluationRunStatus
from shared.core.errors import NotFoundError
from shared.models.evaluation import EvaluationRun

pytestmark = pytest.mark.functional


def _run(**overrides: Any) -> EvaluationRun:
    base = {
        "id": uuid.uuid4(),
        "dataset_name": "sample_v1",
        "dataset_version": "v1",
        "status": EvaluationRunStatus.COMPLETED,
        "prompt_version": "generate@v1",
        "llm_provider": "anthropic",
        "llm_model": "claude-x",
        "embedding_model": "text-embedding-3-small",
        "results": {"cases": []},
        "summary": {"cases": 0},
        "error": None,
        "triggered_by": None,
        "created_at": datetime.now(UTC),
    }
    base.update(overrides)
    return EvaluationRun(**base)


class _FakeEvaluationService:
    def __init__(self, run: EvaluationRun | None = None) -> None:
        self._run = run or _run()
        self.calls: list[dict[str, Any]] = []

    async def run_evaluation(self, **kwargs: Any) -> EvaluationRun:
        self.calls.append(kwargs)
        return self._run


class _FakeEvaluationRunRepository:
    def __init__(self, run: EvaluationRun | None = None) -> None:
        self._run = run

    async def get_or_404(self, run_id: uuid.UUID) -> EvaluationRun:
        if self._run is None or self._run.id != run_id:
            raise NotFoundError()
        return self._run


@pytest.fixture
def evaluation_service(app: FastAPI) -> Iterator[_FakeEvaluationService]:
    """Override the evaluation-run boundary with an in-memory fake."""
    service = _FakeEvaluationService()
    app.dependency_overrides[get_evaluation_service] = lambda: service
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_evaluation_service, None)


@pytest.fixture
def evaluation_run_repo(app: FastAPI) -> Iterator[_FakeEvaluationRunRepository]:
    """Override the run-lookup boundary the GET route queries directly."""
    repo = _FakeEvaluationRunRepository()
    app.dependency_overrides[get_evaluation_run_repository] = lambda: repo
    try:
        yield repo
    finally:
        app.dependency_overrides.pop(get_evaluation_run_repository, None)


def _login_as_admin(
    client: TestClient, user_repo: FakeUserRepository, email: str = "admin@example.com"
) -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text
    user_repo._by_email[email].is_admin = True


def _login_as_reader(client: TestClient, email: str = "reader@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text


# --- POST /evaluations/run ---------------------------------------------------- #


def test_run_evaluation_returns_the_persisted_run(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_service: _FakeEvaluationService,
) -> None:
    _login_as_admin(client, user_repo)
    evaluation_service._run = _run(dataset_name="sample_v1", summary={"cases": 3})

    resp = client.post("/api/v1/evaluations/run", json={"dataset_name": "sample_v1"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset_name"] == "sample_v1"
    assert body["status"] == "completed"
    assert body["summary"] == {"cases": 3}
    assert evaluation_service.calls[0]["dataset_name"] == "sample_v1"


def test_run_evaluation_requires_admin(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_service: _FakeEvaluationService,
) -> None:
    _login_as_reader(client)

    resp = client.post("/api/v1/evaluations/run", json={"dataset_name": "sample_v1"})

    assert resp.status_code == 403


def test_run_evaluation_requires_authentication(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_service: _FakeEvaluationService,
) -> None:
    resp = client.post("/api/v1/evaluations/run", json={"dataset_name": "sample_v1"})

    assert resp.status_code == 401


def test_run_evaluation_rejects_an_empty_dataset_name(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_service: _FakeEvaluationService,
) -> None:
    _login_as_admin(client, user_repo)

    resp = client.post("/api/v1/evaluations/run", json={"dataset_name": ""})

    assert resp.status_code == 422


# --- GET /evaluations/{id} ----------------------------------------------------- #


def test_get_evaluation_run_returns_the_run(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_run_repo: _FakeEvaluationRunRepository,
) -> None:
    _login_as_admin(client, user_repo)
    run = _run()
    evaluation_run_repo._run = run

    resp = client.get(f"/api/v1/evaluations/{run.id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == str(run.id)


def test_get_evaluation_run_404s_for_an_unknown_id(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_run_repo: _FakeEvaluationRunRepository,
) -> None:
    _login_as_admin(client, user_repo)

    resp = client.get(f"/api/v1/evaluations/{uuid.uuid4()}")

    assert resp.status_code == 404


def test_get_evaluation_run_requires_admin(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_run_repo: _FakeEvaluationRunRepository,
) -> None:
    _login_as_reader(client)

    resp = client.get(f"/api/v1/evaluations/{uuid.uuid4()}")

    assert resp.status_code == 403


def test_get_evaluation_run_requires_authentication(
    client: TestClient,
    user_repo: FakeUserRepository,
    evaluation_run_repo: _FakeEvaluationRunRepository,
) -> None:
    resp = client.get(f"/api/v1/evaluations/{uuid.uuid4()}")

    assert resp.status_code == 401
