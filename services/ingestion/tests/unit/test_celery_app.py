"""Unit tests for the ingestion Celery app construction (no broker I/O)."""

import pytest
from celery import Celery
from ingestion.celery_app import app, create_celery_app


@pytest.mark.unit
def test_create_celery_app_returns_configured_instance() -> None:
    """The factory builds a Celery app bound to the Redis broker with safe defaults."""
    built = create_celery_app()

    assert isinstance(built, Celery)
    assert built.main == "recap_ingestion"
    assert built.conf.broker_url.startswith("redis://")
    # Idempotency-critical defaults: late ack + single prefetch for heavy tasks.
    assert built.conf.task_acks_late is True
    assert built.conf.worker_prefetch_multiplier == 1
    assert built.conf.broker_connection_retry_on_startup is True


@pytest.mark.unit
def test_module_level_app_is_a_celery_instance() -> None:
    """The importable ``app`` (used by the `celery -A ...:app` entry point) exists."""
    assert isinstance(app, Celery)
