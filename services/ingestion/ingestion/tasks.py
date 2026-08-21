"""Celery tasks for the ingestion pipeline.

Thin wrappers around the async pipeline in :mod:`ingestion.pipeline`: they run
the coroutine to completion and translate outcomes into Celery's retry model.
Failure handling follows the design:

* A **permanent** failure — a :class:`ParseError` (corrupt/unreadable bytes) —
  marks the document ``failed`` immediately, without raising: retrying can't fix
  bad input, and this is a handled domain outcome, not a Celery task failure.
* A **transient** failure — anything else (storage/embedding/Qdrant/DB
  connectivity) — is retried with bounded exponential backoff. Once retries are
  exhausted the task re-raises and lets it actually fail; :class:`IngestDocumentTask`'s
  ``on_failure`` is the single place that then marks the document ``failed`` —
  covering not just that exhausted-retries case but any other exception that
  might escape this function's own handling unanticipated, which would otherwise
  leave a document stuck ``processing`` with nothing ever recording why.

The task is idempotent (see the pipeline), so ``task_acks_late`` re-delivery on a
worker crash is safe.
"""

import uuid
from typing import Any

from ingestion.base_task import AsyncTask, run_on_process_loop
from ingestion.celery_app import app
from ingestion.pipeline import fail_document, run_ingestion
from ingestion.resources import get_ingestion_resources
from shared.core.config import get_settings
from shared.ingestion_core.parsing import ParseError


def _backoff_seconds(retries: int) -> int:
    """Exponential backoff (1s, 2s, 4s, …) for the ``retries``-th attempt."""
    return 2**retries


def _ids_from_task_args(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[uuid.UUID, uuid.UUID]:
    """Recover ``(document_id, user_id)`` from a task's positional or keyword call args."""
    document_id, user_id = args[:2] if args else (kwargs["document_id"], kwargs["user_id"])
    return uuid.UUID(document_id), uuid.UUID(user_id)


class IngestDocumentTask(AsyncTask):
    """Adds ``ingest_document``'s terminal-failure side effect to the shared base.

    ``on_failure`` fires exactly once per task, only once Celery has genuinely
    given up on it (the exhausted-retries exception below, or anything else that
    escapes ``ingest_document``'s own try/except entirely) — never on a
    ``self.retry()`` call, which Celery routes to ``on_retry`` instead. That makes
    it a safety net beyond the one retry path this module explicitly anticipates.
    """

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        super().on_failure(exc, task_id, args, kwargs, einfo)
        document_id, user_id = _ids_from_task_args(args, kwargs)
        run_on_process_loop(
            fail_document(
                get_ingestion_resources(),
                document_id=document_id,
                user_id=user_id,
                reason=str(exc),
            )
        )


@app.task(base=IngestDocumentTask, bind=True, name="ingestion.ingest_document")
def ingest_document(self: IngestDocumentTask, document_id: str, user_id: str) -> None:
    """Process one uploaded document end-to-end (enqueued by the outbox relay)."""
    resources = get_ingestion_resources()
    ids = {"document_id": uuid.UUID(document_id), "user_id": uuid.UUID(user_id)}
    try:
        self.run_async(run_ingestion(resources, **ids))
    except ParseError as exc:
        # Permanent: the bytes won't parse on a retry either. Marking the document
        # failed is a domain outcome, not a task metric — run it untimed, but on
        # the same persistent loop so it reuses the worker's DB pool. Deliberately
        # not re-raised, so Celery still records this as a task success.
        run_on_process_loop(fail_document(resources, **ids, reason=f"parse failed: {exc}"))
    except Exception as exc:
        max_retries = get_settings().ingest_max_retries
        if self.request.retries < max_retries:
            raise self.retry(
                exc=exc, countdown=_backoff_seconds(self.request.retries), max_retries=max_retries
            ) from exc
        # Retries exhausted: let it actually fail — IngestDocumentTask.on_failure
        # marks the document failed.
        raise
