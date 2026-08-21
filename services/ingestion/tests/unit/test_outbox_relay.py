"""Unit tests for the outbox relay's dispatch logic (no broker, no DB)."""

import uuid
from dataclasses import dataclass

import pytest
from ingestion.outbox_relay import drain_outbox

pytestmark = pytest.mark.unit


@dataclass
class _Event:
    id: uuid.UUID
    event_type: str
    payload: dict


class _FakeOutbox:
    def __init__(self, events: list[_Event]) -> None:
        self._events = events
        self.processed: list[uuid.UUID] = []

    async def fetch_unprocessed(self, *, limit: int) -> list[_Event]:
        return self._events[:limit]

    async def mark_processed(self, event_id: uuid.UUID) -> None:
        self.processed.append(event_id)


async def test_dispatches_once_per_event_and_marks_all_processed() -> None:
    events = [
        _Event(uuid.uuid4(), "document.uploaded", {"document_id": "d1", "user_id": "u1"}),
        _Event(uuid.uuid4(), "document.uploaded", {"document_id": "d2", "user_id": "u1"}),
    ]
    outbox = _FakeOutbox(events)
    dispatched: list[tuple[str, dict]] = []

    count = await drain_outbox(
        outbox, batch_size=100, dispatch=lambda t, p: dispatched.append((t, p))
    )

    assert count == 2
    assert [t for t, _ in dispatched] == ["document.uploaded", "document.uploaded"]
    # Every fetched event is acknowledged so it isn't re-delivered.
    assert outbox.processed == [e.id for e in events]


async def test_empty_backlog_is_a_noop() -> None:
    outbox = _FakeOutbox([])
    count = await drain_outbox(outbox, batch_size=100, dispatch=lambda t, p: None)
    assert count == 0
    assert outbox.processed == []


async def test_unhandled_event_type_is_still_acknowledged() -> None:
    # A type with no consumer yet must be marked processed, not left to accumulate.
    events = [_Event(uuid.uuid4(), "document.indexed", {"document_id": "d1"})]
    outbox = _FakeOutbox(events)
    count = await drain_outbox(outbox, batch_size=100, dispatch=lambda t, p: None)
    assert count == 1
    assert outbox.processed == [events[0].id]
