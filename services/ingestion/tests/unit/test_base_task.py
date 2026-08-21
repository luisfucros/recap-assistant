"""Unit tests for the AsyncTask Celery base (metric label + coroutine runner).

Covers the cross-cutting behavior the base owns — deriving the low-cardinality
metric label and running a coroutine to completion while timing it under a
success/error outcome. Domain status transitions live in the pipeline and are
tested there, not here.
"""

import asyncio

import pytest
from ingestion.base_task import AsyncTask, close_process_loop, get_process_loop
from prometheus_client import REGISTRY

pytestmark = pytest.mark.unit


def _task(name: str | None) -> AsyncTask:
    """An unbound AsyncTask with a chosen ``name`` (no Celery app / broker needed)."""
    task = AsyncTask()
    task.name = name
    return task


def _task_count(task_label: str, outcome: str) -> float:
    """The histogram observation count for a (task, outcome) label pair, or 0."""
    return (
        REGISTRY.get_sample_value(
            "recap_ingestion_task_seconds_count", {"task": task_label, "outcome": outcome}
        )
        or 0.0
    )


def test_metric_name_strips_dotted_namespace() -> None:
    assert _task("ingestion.ingest_document")._metric_name() == "ingest_document"


def test_metric_name_falls_back_to_class_name_without_a_name() -> None:
    assert _task(None)._metric_name() == "AsyncTask"


def test_run_async_runs_coroutine_returns_value_and_times_success() -> None:
    task = _task("ingestion.unit_probe_success")

    async def work() -> int:
        return 21 * 2

    before = _task_count("unit_probe_success", "success")
    assert task.run_async(work()) == 42
    assert _task_count("unit_probe_success", "success") == before + 1


def test_run_async_records_error_outcome_and_reraises() -> None:
    task = _task("ingestion.unit_probe_error")

    async def boom() -> None:
        raise ValueError("nope")

    before = _task_count("unit_probe_error", "error")
    with pytest.raises(ValueError, match="nope"):
        task.run_async(boom())
    assert _task_count("unit_probe_error", "error") == before + 1


def test_consecutive_runs_share_one_persistent_loop() -> None:
    """The regression guard: every run uses the same per-process loop.

    Pooled resources (asyncpg / Qdrant / Redis) bind their connections to the
    loop that opened them; a fresh loop per task would make the second task fail
    with "Future attached to a different loop". Here we assert the loop the
    coroutine actually runs on is stable across calls and identical to
    ``get_process_loop()``.
    """
    task = _task("ingestion.unit_probe_loop")

    async def running_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    first = task.run_async(running_loop())
    second = task.run_async(running_loop())

    assert first is second is get_process_loop()
    assert not first.is_closed()


def test_close_process_loop_is_idempotent_and_reopens() -> None:
    """Shutdown closes the loop; a later run transparently opens a fresh one."""
    loop = get_process_loop()
    close_process_loop()
    assert loop.is_closed()
    close_process_loop()  # second close is a no-op, not an error

    reopened = get_process_loop()
    assert reopened is not loop
    assert not reopened.is_closed()
