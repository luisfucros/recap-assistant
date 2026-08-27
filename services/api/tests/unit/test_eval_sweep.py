"""Unit tests for the stuck-eval-run sweep's fan-out logic (no DB, no broker)."""

import uuid
from datetime import UTC, datetime

import pytest
from api.eval_worker import sweep as sweep_module

pytestmark = pytest.mark.unit


async def test_sweep_dispatches_once_per_stuck_run(monkeypatch: pytest.MonkeyPatch) -> None:
    stuck = [uuid.uuid4(), uuid.uuid4()]

    async def _fake_find(resources, *, now, stuck_after_seconds):
        return stuck

    monkeypatch.setattr(sweep_module, "find_stuck_runs", _fake_find)
    dispatched: list[uuid.UUID] = []

    count = await sweep_module.sweep_stuck_runs(
        resources=object(),  # type: ignore[arg-type]
        now=datetime.now(UTC),
        stuck_after_seconds=900,
        dispatch=dispatched.append,
    )

    assert count == 2
    assert dispatched == stuck


async def test_sweep_is_a_noop_with_nothing_stuck(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_find(resources, *, now, stuck_after_seconds):
        return []

    monkeypatch.setattr(sweep_module, "find_stuck_runs", _fake_find)
    dispatched: list[uuid.UUID] = []

    count = await sweep_module.sweep_stuck_runs(
        resources=object(),  # type: ignore[arg-type]
        now=datetime.now(UTC),
        stuck_after_seconds=900,
        dispatch=dispatched.append,
    )

    assert count == 0
    assert dispatched == []
