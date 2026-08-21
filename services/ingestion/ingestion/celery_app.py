"""Celery application for the ingestion pipeline service.

The ingestion service runs document processing (parse → chunk → embed → upsert)
off the API's request path, so it is driven by Celery rather than the web stack.
This module builds and configures the Celery app from the shared settings; task
modules and the beat schedule (the outbox relay) are registered here as they are
implemented.

Entry points (used by the container):
    celery -A ingestion.celery_app:app worker   # process ingestion tasks
    celery -A ingestion.celery_app:app beat      # periodic outbox relay
"""

from celery import Celery

from shared.core.config import get_settings


def create_celery_app() -> Celery:
    """Build and configure the Celery app from application settings.

    Returns:
        A configured ``Celery`` instance bound to the Redis broker. Broker
        connection is lazy, so importing this module never requires Redis to be
        up (safe for tooling, tests, and container start ordering).
    """
    settings = get_settings()
    app = Celery("recap_ingestion", broker=settings.celery_broker_url)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Ack tasks only after they finish, so a worker crash re-delivers the task.
        # Safe because ingestion is idempotent (a re-run replaces a document's
        # chunks/vectors) — see the ingestion pipeline design.
        task_acks_late=True,
        # Reserve one task at a time per worker process: ingestion tasks are
        # heavy (parsing, embedding), so fair dispatch beats prefetch throughput.
        worker_prefetch_multiplier=1,
        # Keep retrying the broker at startup instead of crashing if Redis is not
        # yet reachable (compose start ordering / transient restarts).
        broker_connection_retry_on_startup=True,
        # Beat drives the outbox relay: poll pending events and enqueue tasks.
        # MUST run as a single beat instance (multiple would double-enqueue).
        beat_schedule={
            "relay-outbox": {
                "task": "ingestion.relay_outbox",
                "schedule": settings.outbox_relay_interval_seconds,
            },
            # Safety net alongside the outbox relay: re-enqueues any document
            # stuck pending/processing past ingest_stuck_threshold_seconds. See
            # ingestion.sweep for why the outbox relay alone can't cover this.
            "sweep-stuck-documents": {
                "task": "ingestion.sweep_stuck_documents",
                "schedule": settings.ingest_sweep_interval_seconds,
            },
        },
    )
    return app


app = create_celery_app()

# Register worker lifecycle hooks (metrics server, resource disposal) and the
# task modules (ingest_document + the outbox relay) as a side effect of importing
# the Celery app, so `celery -A ingestion.celery_app:app` sees every task.
from ingestion import bootstrap, outbox_relay, reembed, sweep, tasks  # noqa: E402, F401
