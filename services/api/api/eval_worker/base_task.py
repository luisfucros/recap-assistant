"""Shared Celery base task for the eval worker.

Mirrors ``ingestion.base_task`` (services must not import each other): one
persistent event loop per prefork child so ``Resources`` pools stay valid
across tasks. Metrics use the shared ``time_task`` histogram with
``task=run_evaluation``; Prometheus scrapes this process as job ``eval``.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

from celery import Task
from loguru import logger

from shared.observability import time_task

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
    """Drive ``coro`` to completion on the process's persistent event loop."""
    return get_process_loop().run_until_complete(coro)


def close_process_loop() -> None:
    """Close the persistent loop (worker shutdown), after resources are disposed."""
    global _process_loop
    with _process_loop_lock:
        if _process_loop is not None and not _process_loop.is_closed():
            _process_loop.close()
        _process_loop = None


class AsyncTask(Task):
    """Celery base task adding a coroutine runner + uniform metrics and logging."""

    def _metric_name(self) -> str:
        return (self.name or type(self).__name__).rsplit(".", 1)[-1]

    def run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run the task's primary coroutine, timed as a Celery task metric."""
        with time_task(self._metric_name()):
            return run_on_process_loop(coro)

    def before_start(self, task_id: str, args: tuple, kwargs: dict) -> None:
        logger.bind(task=self.name, task_id=task_id).info("celery.task: started")

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        logger.bind(task=self.name, task_id=task_id).info("celery.task: succeeded")

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        logger.bind(task=self.name, task_id=task_id).warning("celery.task: retrying: {}", exc)

    def on_failure(
        self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any
    ) -> None:
        logger.bind(task=self.name, task_id=task_id).opt(exception=exc).error("celery.task: failed")
