"""Shared Celery base task for the ingestion service.

Centralizes the plumbing every task repeats — driving the async pipeline to
completion on the worker's event loop, timing it as a Prometheus metric, and
emitting structured start/success/retry/failure logs — so each task body holds
only its domain logic.

Deliberately *cross-cutting only*: the base never touches document status.
Failure and success **semantics** (marking a document ``failed`` / ``indexed``)
stay in the pipeline, where the ``indexed`` transition is welded into the atomic
terminal transaction and the ``failed`` transition follows the permanent-vs-
transient policy — neither of which a generic base could enforce correctly.

Event-loop model
-----------------
Every coroutine runs on **one persistent event loop per worker process**, not a
throwaway ``asyncio.run`` loop per task. The cached resources
(:class:`~ingestion.resources.IngestionResources`) hold pooled connections —
SQLAlchemy's asyncpg pool, and the Qdrant/Redis HTTP pools — that bind to the
event loop which first opened them. A new loop per task would leave the second
task reusing a connection from the first (now-closed) loop, raising *"Future
attached to a different loop"*. A stable per-process loop keeps those pools valid
across tasks. This is sound because the worker uses the prefork pool with a
single prefetch, so one task runs at a time per process — the loop is never
driven concurrently.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from celery import Task
from loguru import logger

from shared.observability import time_task

# One event loop per worker process, created on first use. Guarded by a lock only
# as defense-in-depth; the prefork pool runs tasks sequentially within a process.
_process_loop: asyncio.AbstractEventLoop | None = None
_process_loop_lock = threading.Lock()


def get_process_loop() -> asyncio.AbstractEventLoop:
    """Return this process's persistent event loop, creating it on first use."""
    global _process_loop
    with _process_loop_lock:
        if _process_loop is None or _process_loop.is_closed():
            _process_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_process_loop)
        return _process_loop


def run_on_process_loop(coro: Coroutine[Any, Any, Any]) -> Any:
    """Drive ``coro`` to completion on the process's persistent event loop.

    Used both by tasks (via :meth:`AsyncTask.run_async`) and by the worker
    shutdown hook, so resource disposal runs on the same loop the connections
    were opened on.
    """
    return get_process_loop().run_until_complete(coro)


def close_process_loop() -> None:
    """Close the persistent loop (worker shutdown), after resources are disposed."""
    global _process_loop
    with _process_loop_lock:
        if _process_loop is not None and not _process_loop.is_closed():
            _process_loop.close()
        _process_loop = None


class AsyncTask(Task):
    """Celery base task adding a coroutine runner + uniform metrics and logging.

    Applied via the decorator's ``base=`` argument; a bound task body runs its
    pipeline coroutine through :meth:`run_async`::

        @app.task(base=AsyncTask, bind=True, name="ingestion.ingest_document")
        def ingest_document(self, document_id: str, user_id: str) -> None:
            return self.run_async(run_ingestion(...))
    """

    def _metric_name(self) -> str:
        """Prometheus ``task`` label: the task name without its dotted prefix.

        Keeps labels stable and low-cardinality (``ingest_document`` rather than
        ``ingestion.ingest_document``).
        """
        return (self.name or type(self).__name__).rsplit(".", 1)[-1]

    def run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run the task's primary coroutine to completion, timed as a task metric.

        Runs on the worker's persistent event loop (see the module docstring), so
        pooled connections in the cached resources stay valid across tasks. Timing
        covers only this call, recording a success/error outcome on the
        ``recap_ingestion_task_seconds`` histogram. Cleanup coroutines a task may
        run afterwards (e.g. marking a document failed) are intentionally left
        untimed so a task records exactly one primary duration per attempt.
        """
        with time_task(self._metric_name()):
            return run_on_process_loop(coro)

    def before_start(self, task_id: str, args: tuple, kwargs: dict) -> None:
        logger.bind(task=self.name, task_id=task_id).info("celery task started")

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        logger.bind(task=self.name, task_id=task_id).info("celery task succeeded")

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        logger.bind(task=self.name, task_id=task_id).warning("celery task retrying: {}", exc)

    def on_failure(
        self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any
    ) -> None:
        logger.bind(task=self.name, task_id=task_id).error("celery task failed: {}", exc)
