"""Always-on Prometheus metrics and optional, no-op-capable Langfuse tracing."""

from shared.observability.metrics import (
    observe_document_chunks,
    record_document_ingested,
    record_tokens,
    render_metrics,
    set_outbox_pending,
    time_operation,
    time_task,
)
from shared.observability.tracing import NoOpTracer, Tracer, build_tracer

__all__ = [  # noqa: RUF022 — grouped: metrics helpers, then tracing
    "record_tokens",
    "render_metrics",
    "time_operation",
    "time_task",
    "record_document_ingested",
    "observe_document_chunks",
    "set_outbox_pending",
    "build_tracer",
    "Tracer",
    "NoOpTracer",
]
