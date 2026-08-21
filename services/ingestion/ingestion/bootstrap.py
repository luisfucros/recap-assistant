"""Celery worker lifecycle hooks: warm heavy resources, expose metrics, dispose.

Imported for its signal registrations (a side effect). On process start each pool
child warms its heavy singletons (the embedder — a local model loads weights
here); the main process serves the worker's Prometheus ``/metrics``; on shutdown
each process disposes its per-process resources.

Caveat: with the prefork pool, tasks run in child processes whose metric samples
live in separate memory from this (main-process) server. Full cross-process
aggregation needs prometheus_client multiprocess mode — deferred to hardening.
"""

from contextlib import suppress

from celery.signals import worker_process_init, worker_ready, worker_shutdown
from loguru import logger
from prometheus_client import start_http_server

import shared.observability.metrics  # noqa: F401 — registers custom metrics on import
from ingestion.base_task import close_process_loop, run_on_process_loop
from ingestion.resources import get_ingestion_resources
from shared.core.config import get_settings


@worker_process_init.connect
def _warm_heavy_resources(**_: object) -> None:
    """Force-build heavy resources in each prefork child as it starts.

    The heavy singletons are declared on
    :attr:`~ingestion.resources.IngestionResources.HEAVY_RESOURCES`; their
    *construction* is expensive (a local embedding model loads weights into
    memory), so paying it once at boot keeps the first ingestion from eating that
    latency. Skipped when ``warm_up_on_start`` is off (tests/CI).

    Runs in the child (not the main process), because prefork tasks execute in
    forked children with their own copy of the process-cached resources — warming
    the main process would not populate them. Failures are logged, not raised: the
    ``cached_property`` only caches on success, so a genuine config error (e.g. a
    hosted provider's missing key, or the local extra not installed) is surfaced
    normally on the first task instead of crashing the worker at boot.
    """
    if not get_settings().warm_up_on_start:
        return
    resources = get_ingestion_resources()
    for name in type(resources).HEAVY_RESOURCES:
        try:
            getattr(resources, name)
            logger.info("Warmed up ingestion resources.{} at process start", name)
        except Exception:
            logger.exception("Warm-up of resources.{} failed; will retry lazily", name)


@worker_ready.connect
def _start_metrics_server(**_: object) -> None:
    # Best-effort: a port clash must not prevent the worker from processing tasks.
    with suppress(OSError):
        start_http_server(get_settings().ingestion_metrics_port)


@worker_shutdown.connect
def _dispose_resources(**_: object) -> None:
    # Only dispose if resources were actually built (avoid constructing them just
    # to tear them down). Dispose on the persistent loop the connections were
    # opened on, then close that loop.
    if get_ingestion_resources.cache_info().currsize:
        run_on_process_loop(get_ingestion_resources().aclose())
    close_process_loop()
