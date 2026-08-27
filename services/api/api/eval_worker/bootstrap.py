"""Celery worker lifecycle: warm Resources, expose metrics, dispose.

Imported for its signal registrations. Prefork children warm loop-bound/heavy
singletons on the persistent process loop (same split as the API lifespan).
"""

import asyncio
from contextlib import suppress

from celery.signals import worker_process_init, worker_ready, worker_shutdown
from loguru import logger
from prometheus_client import start_http_server

import shared.observability.metrics  # noqa: F401
from api.eval_worker.base_task import close_process_loop, run_on_process_loop
from api.eval_worker.resources import get_eval_resources
from shared.core.config import get_settings


async def _warm_eval_resources() -> None:
    """Open the checkpointer pool and force-build heavy/loop-bound singletons."""
    resources = get_eval_resources()
    try:
        await resources.checkpointer_pool.open(wait=False)
        logger.info("Opened the agent checkpointer pool at eval worker start")
    except Exception:
        logger.exception("Opening the checkpointer pool failed; will connect lazily")

    if not get_settings().warm_up_on_start:
        return

    loop_bound = type(resources).LOOP_BOUND_RESOURCES
    names = (*type(resources).HEAVY_RESOURCES, "evaluation_service")
    for name in names:
        try:
            if name in loop_bound or name == "evaluation_service":
                getattr(resources, name)
            else:
                await asyncio.to_thread(getattr, resources, name)
            logger.info("Warmed up eval resources.{} at process start", name)
        except Exception:
            logger.exception("Warm-up of resources.{} failed; will retry lazily", name)


@worker_process_init.connect
def _warm_heavy_resources(**_: object) -> None:
    run_on_process_loop(_warm_eval_resources())


@worker_ready.connect
def _start_metrics_server(**_: object) -> None:
    with suppress(OSError):
        start_http_server(get_settings().eval_metrics_port)


@worker_shutdown.connect
def _dispose_resources(**_: object) -> None:
    if get_eval_resources.cache_info().currsize:
        run_on_process_loop(get_eval_resources().aclose())
    close_process_loop()
