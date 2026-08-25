"""Stuck-document sweep (Celery beat safety net).

The outbox relay (:mod:`ingestion.outbox_relay`) marks an event "processed" the
moment its task is *enqueued* — not when ingestion actually finishes — because
that is the correct boundary for the outbox pattern itself (see its module
docstring). But that also means the outbox never looks at a document again once
its event is dispatched: if the enqueued message is ever lost (a broker restart,
or any other gap ``task_acks_late``/idempotent retries don't cover) there is no
other path back to it.

This module closes that gap from the other side: on a beat schedule, it scans
``documents`` directly for rows sitting in ``pending``/``processing`` well past
how long a healthy run (including its own retry backoff) should take, and
re-enqueues them. Status — not a one-shot dispatch — is the source of truth for
"does this document still need work". Re-enqueuing a document whose task is
actually still in flight is harmless: :func:`ingestion.pipeline.run_ingestion` is
idempotent (a re-run replaces the prior attempt's chunks/vectors).

Runs as a trusted, cross-user system query (mirrors :func:`ingestion.reembed.reembed_all`'s
own corpus-wide scan) — each stuck document is still only ever re-enqueued for its
own ``user_id``, so no per-user isolation is crossed.
"""

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from loguru import logger
from sqlalchemy import select

from ingestion.base_task import AsyncTask
from ingestion.celery_app import app
from ingestion.resources import IngestionResources, get_ingestion_resources
from ingestion.tasks import ingest_document
from shared.core.config import get_settings
from shared.core.enums import DocumentStatus
from shared.models.document import Document

_STUCK_STATUSES = (DocumentStatus.PENDING, DocumentStatus.PROCESSING)

# What re-enqueuing a stuck document does, injected so the fan-out logic below is
# unit-testable without a real Celery app (mirrors outbox_relay.Dispatch).
Dispatch = Callable[[str, str], None]


def _dispatch(document_id: str, user_id: str) -> None:
    """Enqueue an ``ingest_document`` task for one stuck document."""
    ingest_document.delay(document_id, user_id)


async def find_stuck_documents(
    resources: IngestionResources, *, now: datetime, stuck_after_seconds: int
) -> Sequence[tuple[uuid.UUID, uuid.UUID]]:
    """Return ``(document_id, user_id)`` for rows stale past the threshold.

    ``now`` is injected rather than read from the clock here, so the cutoff is
    deterministic and testable.
    """
    cutoff = now - timedelta(seconds=stuck_after_seconds)
    async with resources.sessionmaker() as session:
        result = await session.execute(
            select(Document.id, Document.user_id).where(
                Document.status.in_(_STUCK_STATUSES),
                Document.updated_at < cutoff,
            )
        )
        return result.all()


async def sweep_stuck_documents(
    resources: IngestionResources,
    *,
    now: datetime,
    stuck_after_seconds: int,
    dispatch: Dispatch = _dispatch,
) -> int:
    """Re-enqueue every stuck document found; return how many were dispatched."""
    stuck = await find_stuck_documents(resources, now=now, stuck_after_seconds=stuck_after_seconds)
    if not stuck:
        logger.debug("ingest.sweep: no stuck documents")
        return 0
    for document_id, user_id in stuck:
        logger.bind(document_id=str(document_id), user_id=str(user_id)).warning(
            "ingest.sweep: re-enqueuing stuck document past {}s", stuck_after_seconds
        )
        dispatch(str(document_id), str(user_id))
    logger.info("ingest.sweep: re-enqueued {} documents", len(stuck))
    return len(stuck)


@app.task(base=AsyncTask, bind=True, name="ingestion.sweep_stuck_documents")
def sweep_stuck_documents_task(self: AsyncTask) -> int:
    """Beat-scheduled sweep tick; returns how many stuck documents were re-enqueued."""
    settings = get_settings()
    return self.run_async(
        sweep_stuck_documents(
            get_ingestion_resources(),
            now=datetime.now(tz=UTC),
            stuck_after_seconds=settings.ingest_stuck_threshold_seconds,
        )
    )
