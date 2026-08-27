"""Unit tests for the eval Celery task body (no broker)."""

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from api.eval_worker.tasks import _execute

pytestmark = pytest.mark.unit


async def test_execute_helper_calls_execute_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    execute = AsyncMock()
    session = SimpleNamespace()

    @asynccontextmanager
    async def _sessionmaker():
        yield session

    resources = SimpleNamespace(
        evaluation_service=SimpleNamespace(execute_evaluation=execute),
        sessionmaker=_sessionmaker,
    )
    monkeypatch.setattr("api.eval_worker.tasks.get_eval_resources", lambda: resources)
    monkeypatch.setattr("api.eval_worker.tasks.UserRepository", lambda s: "users")
    monkeypatch.setattr("api.eval_worker.tasks.EvaluationRunRepository", lambda s: "runs")

    await _execute(run_id)

    execute.assert_awaited_once()
    kwargs = execute.await_args.kwargs
    assert kwargs["run_id"] == run_id
    assert kwargs["session"] is session
    assert kwargs["users"] == "users"
    assert kwargs["runs"] == "runs"
