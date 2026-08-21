"""Unit tests for the shared Prometheus metrics helpers (no I/O)."""

import pytest
from prometheus_client import REGISTRY

from shared.observability.metrics import (
    observe_document_chunks,
    record_document_ingested,
    record_tokens,
    render_metrics,
    set_outbox_pending,
    time_operation,
    time_task,
)


def _op_count(operation: str, outcome: str) -> float:
    """Histogram sample count for one (operation, outcome), 0.0 if unseen."""
    return (
        REGISTRY.get_sample_value(
            "recap_operation_seconds_count",
            {"operation": operation, "outcome": outcome},
        )
        or 0.0
    )


@pytest.mark.unit
def test_time_operation_records_success() -> None:
    before = _op_count("retrieval", "success")
    with time_operation("retrieval"):
        pass
    assert _op_count("retrieval", "success") == before + 1


@pytest.mark.unit
def test_time_operation_labels_error_and_reraises() -> None:
    before = _op_count("embedding", "error")
    with pytest.raises(ValueError), time_operation("embedding"):
        raise ValueError("boom")
    assert _op_count("embedding", "error") == before + 1


@pytest.mark.unit
def test_record_tokens_and_render() -> None:
    record_tokens("prompt", 10)
    payload, content_type = render_metrics()
    assert b"recap_llm_tokens_total" in payload
    assert "text/plain" in content_type


def _task_count(task: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "recap_ingestion_task_seconds_count", {"task": task, "outcome": outcome}
        )
        or 0.0
    )


@pytest.mark.unit
def test_time_task_records_success_and_error() -> None:
    before_ok = _task_count("ingest_document", "success")
    with time_task("ingest_document"):
        pass
    assert _task_count("ingest_document", "success") == before_ok + 1

    before_err = _task_count("ingest_document", "error")
    with pytest.raises(RuntimeError), time_task("ingest_document"):
        raise RuntimeError("boom")
    assert _task_count("ingest_document", "error") == before_err + 1


@pytest.mark.unit
def test_record_document_ingested_counts_by_outcome() -> None:
    def count(outcome: str) -> float:
        return (
            REGISTRY.get_sample_value("recap_documents_ingested_total", {"outcome": outcome}) or 0.0
        )

    before = count("indexed")
    record_document_ingested("indexed")
    assert count("indexed") == before + 1


@pytest.mark.unit
def test_observe_document_chunks_and_set_outbox_pending() -> None:
    before = REGISTRY.get_sample_value("recap_document_chunks_count") or 0.0
    observe_document_chunks(7)
    assert (REGISTRY.get_sample_value("recap_document_chunks_count") or 0.0) == before + 1

    set_outbox_pending(42)
    assert REGISTRY.get_sample_value("recap_outbox_pending") == 42.0
