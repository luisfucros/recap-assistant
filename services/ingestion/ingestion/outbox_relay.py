"""The transactional-outbox relay (Celery beat).

Runs on a schedule: poll the unprocessed outbox backlog oldest-first, dispatch a
Celery task per event, and stamp each event processed — all in one transaction,
so an event is marked delivered only once its task is enqueued. Delivery is
at-least-once (a crash between enqueue and commit re-delivers), which is safe
because the downstream tasks are idempotent.

**Must run as a single beat instance** — two beats would double-enqueue the
backlog.
"""

from collections.abc import Callable

from loguru import logger

from ingestion.base_task import AsyncTask
from ingestion.celery_app import app
from ingestion.resources import IngestionResources, get_ingestion_resources
from ingestion.tasks import ingest_document
from shared.core.events import DOCUMENT_UPLOADED
from shared.observability import set_outbox_pending
from shared.repositories import OutboxRepository

# What each event type triggers. Types with no consumer yet (e.g. document.indexed,
# which memory-indexing will consume in a later milestone) are acknowledged and
# dropped rather than left to accumulate in the backlog.
Dispatch = Callable[[str, dict], None]


def _dispatch(event_type: str, payload: dict) -> None:
    """Enqueue the Celery task a committed outbox event should trigger."""
    if event_type == DOCUMENT_UPLOADED:
        ingest_document.delay(payload["document_id"], payload["user_id"])
        return
    logger.debug("outbox relay: no consumer for event type {!r}; acknowledging", event_type)


async def drain_outbox(outbox: OutboxRepository, *, batch_size: int, dispatch: Dispatch) -> int:
    """Dispatch and mark-processed one batch of unprocessed events; return the count.

    Pure of Celery/DB construction (its collaborators are injected), so the relay
    logic — dispatch once per event, then mark processed — is unit-testable.
    """
    events = await outbox.fetch_unprocessed(limit=batch_size)
    for event in events:
        dispatch(event.event_type, event.payload)
        await outbox.mark_processed(event.id)
    return len(events)


async def _relay(resources: IngestionResources) -> int:
    async with resources.sessionmaker() as session:
        outbox = OutboxRepository(session)
        count = await drain_outbox(
            outbox,
            batch_size=resources.settings.outbox_relay_batch_size,
            dispatch=_dispatch,
        )
        await session.commit()
        # Publish the remaining backlog as the ingestion queue depth.
        set_outbox_pending(await outbox.count_unprocessed())
    return count


@app.task(base=AsyncTask, bind=True, name="ingestion.relay_outbox")
def relay_outbox(self: AsyncTask) -> int:
    """Beat-scheduled relay tick; returns how many events were dispatched."""
    return self.run_async(_relay(get_ingestion_resources()))
