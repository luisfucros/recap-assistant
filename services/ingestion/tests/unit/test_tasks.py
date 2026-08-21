"""Unit tests for ``ingest_document``'s terminal-failure handling.

``IngestDocumentTask.on_failure`` is the single place a document is marked
``failed`` once Celery has genuinely given up on the task — the exhausted-retries
case ``ingest_document`` raises into, or anything else that might escape its own
try/except unanticipated. ``fail_document`` itself (the DB write) is faked here
so this stays a pure, no-I/O unit test; it is exercised for real in
``test_pipeline_pg.py``.
"""

import uuid

import pytest
from ingestion.tasks import IngestDocumentTask, _ids_from_task_args

from ingestion import tasks as tasks_module

pytestmark = pytest.mark.unit


def _task(name: str = "ingestion.ingest_document") -> IngestDocumentTask:
    task = IngestDocumentTask()
    task.name = name
    return task


def test_ids_from_task_args_recovers_positional_args() -> None:
    document_id, user_id = uuid.uuid4(), uuid.uuid4()
    assert _ids_from_task_args((str(document_id), str(user_id)), {}) == (document_id, user_id)


def test_ids_from_task_args_recovers_keyword_args() -> None:
    document_id, user_id = uuid.uuid4(), uuid.uuid4()
    result = _ids_from_task_args((), {"document_id": str(document_id), "user_id": str(user_id)})
    assert result == (document_id, user_id)


def test_on_failure_marks_the_document_failed_with_the_exception_reason(monkeypatch) -> None:
    calls: list[tuple[uuid.UUID, uuid.UUID, str]] = []

    async def _fake_fail_document(resources, *, document_id, user_id, reason):
        calls.append((document_id, user_id, reason))

    monkeypatch.setattr(tasks_module, "fail_document", _fake_fail_document)
    monkeypatch.setattr(tasks_module, "get_ingestion_resources", lambda: object())

    document_id, user_id = uuid.uuid4(), uuid.uuid4()
    task = _task()
    task.on_failure(
        RuntimeError("qdrant unreachable"),
        "task-1",
        (str(document_id), str(user_id)),
        {},
        None,
    )

    assert calls == [(document_id, user_id, "qdrant unreachable")]


def test_on_failure_recovers_ids_from_keyword_args(monkeypatch) -> None:
    calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def _fake_fail_document(resources, *, document_id, user_id, reason):
        calls.append((document_id, user_id))

    monkeypatch.setattr(tasks_module, "fail_document", _fake_fail_document)
    monkeypatch.setattr(tasks_module, "get_ingestion_resources", lambda: object())

    document_id, user_id = uuid.uuid4(), uuid.uuid4()
    task = _task()
    task.on_failure(
        RuntimeError("boom"),
        "task-2",
        (),
        {"document_id": str(document_id), "user_id": str(user_id)},
        None,
    )

    assert calls == [(document_id, user_id)]
