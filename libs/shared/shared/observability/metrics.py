"""Prometheus metrics — always on, no external dependency.

Metrics are the operational SLI layer (distinct from optional Langfuse tracing):
they must never require credentials or a network peer. HTTP request metrics are
added per-service by an ASGI instrumentator; this module defines the custom
RAG-pipeline metrics (shared by both services) and a helper to render the
registry for a ``/metrics`` scrape endpoint. Process CPU/memory come for free
from ``prometheus_client``'s default collectors.

Labels are kept low-cardinality (operation kind, outcome) — never raw user input,
ids, or free text, which would explode the time-series count.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Duration of a RAG pipeline step. `operation` ∈ {retrieval, embedding, llm, tool};
# `outcome` ∈ {success, error}. Count/sum derive from the histogram, so no
# separate counter is needed for call volume.
OPERATION_SECONDS = Histogram(
    "recap_operation_seconds",
    "Duration of a RAG pipeline operation.",
    labelnames=("operation", "outcome"),
)

# LLM token usage (per-user spend is aggregated in dashboards, not labeled here).
LLM_TOKENS = Counter(
    "recap_llm_tokens_total",
    "LLM tokens processed.",
    labelnames=("kind",),  # prompt | completion
)

# --- Ingestion pipeline metrics ------------------------------------------- #
# Duration of a whole Celery task; count/sum give throughput. `task` is the task
# name (e.g. "ingest_document", "relay_outbox"), `outcome` ∈ {success, error}.
INGESTION_TASK_SECONDS = Histogram(
    "recap_ingestion_task_seconds",
    "Duration of an ingestion Celery task.",
    labelnames=("task", "outcome"),
)

# Terminal outcomes of document ingestion; `outcome` ∈ {indexed, failed}.
DOCUMENTS_INGESTED = Counter(
    "recap_documents_ingested_total",
    "Documents that reached a terminal ingestion state.",
    labelnames=("outcome",),
)

# Chunks produced per successfully-indexed document (bucketed to size the corpus).
DOCUMENT_CHUNKS = Histogram(
    "recap_document_chunks",
    "Number of chunks produced per indexed document.",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

# Outbox backlog (unprocessed events) — the relay refreshes this each tick, so it
# doubles as the ingestion queue depth.
OUTBOX_PENDING = Gauge(
    "recap_outbox_pending",
    "Unprocessed transactional-outbox events (ingestion queue depth).",
)


@contextmanager
def time_operation(operation: str) -> Generator[None]:
    """Time a RAG operation, recording duration under a success/error outcome.

    Args:
        operation: Low-cardinality operation name (e.g. "retrieval", "embedding",
            "llm", "tool").
    """
    start = time.perf_counter()
    outcome = "success"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        OPERATION_SECONDS.labels(operation=operation, outcome=outcome).observe(
            time.perf_counter() - start
        )


def record_tokens(kind: str, count: int) -> None:
    """Record LLM token usage. ``kind`` is "prompt" or "completion"."""
    LLM_TOKENS.labels(kind=kind).inc(count)


@contextmanager
def time_task(task: str) -> Generator[None]:
    """Time a Celery task, recording duration under a success/error outcome.

    Args:
        task: Low-cardinality task name (e.g. "ingest_document", "relay_outbox").
    """
    start = time.perf_counter()
    outcome = "success"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        INGESTION_TASK_SECONDS.labels(task=task, outcome=outcome).observe(
            time.perf_counter() - start
        )


def record_document_ingested(outcome: str) -> None:
    """Count a document reaching a terminal state. ``outcome`` is "indexed"/"failed"."""
    DOCUMENTS_INGESTED.labels(outcome=outcome).inc()


def observe_document_chunks(count: int) -> None:
    """Record how many chunks an indexed document produced."""
    DOCUMENT_CHUNKS.observe(count)


def set_outbox_pending(count: int) -> None:
    """Publish the current outbox backlog size (ingestion queue depth)."""
    OUTBOX_PENDING.set(count)


def render_metrics() -> tuple[bytes, str]:
    """Return ``(payload, content_type)`` for a Prometheus scrape endpoint.

    Used by services without an ASGI instrumentator (e.g. the ingestion worker's
    metrics endpoint); the API exposes ``/metrics`` via its instrumentator.
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
