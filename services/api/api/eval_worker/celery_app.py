"""Celery application for evaluation dataset runs (FR-12.5).

Entry points:
    celery -A api.eval_worker.celery_app:app worker --queues=eval
    celery -A api.eval_worker.celery_app:app beat
"""

from celery import Celery

from shared.core.config import get_settings


def create_celery_app() -> Celery:
    """Build the eval Celery app. Broker connect is lazy (safe without Redis)."""
    settings = get_settings()
    app = Celery("recap_eval", broker=settings.celery_broker_url)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        task_default_queue="eval",
        task_routes={
            "eval.run_evaluation": {"queue": "eval"},
            "eval.sweep_stuck_runs": {"queue": "eval"},
        },
        beat_schedule={
            "sweep-stuck-eval-runs": {
                "task": "eval.sweep_stuck_runs",
                "schedule": settings.eval_sweep_interval_seconds,
            },
        },
    )
    return app


app = create_celery_app()

from api.eval_worker import bootstrap, sweep, tasks  # noqa: E402, F401
