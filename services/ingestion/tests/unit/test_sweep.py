"""Unit tests for the stuck-document sweep's fan-out logic (no DB, no broker).

``find_stuck_documents`` issues a real query and is integration-tested against
Postgres; here it is monkeypatched so ``sweep_stuck_documents``'s own logic —
dispatch once per stuck row, return the count — is tested in isolation, mirroring
how ``test_outbox_relay.py`` tests ``drain_outbox`` against a fake outbox.
"""

import uuid
from datetime import UTC, datetime

import pytest

from ingestion import sweep as sweep_module

pytestmark = pytest.mark.unit


async def _patched(monkeypatch, stuck: list[tuple[uuid.UUID, uuid.UUID]]):
    async def _fake_find_stuck_documents(resources, *, now, stuck_after_seconds):
        return stuck

    monkeypatch.setattr(sweep_module, "find_stuck_documents", _fake_find_stuck_documents)


async def test_sweep_dispatches_once_per_stuck_document(monkeypatch) -> None:
    stuck = [(uuid.uuid4(), uuid.uuid4()), (uuid.uuid4(), uuid.uuid4())]
    await _patched(monkeypatch, stuck)
    dispatched: list[tuple[str, str]] = []

    count = await sweep_module.sweep_stuck_documents(
        resources=object(),
        now=datetime.now(UTC),
        stuck_after_seconds=900,
        dispatch=lambda doc_id, user_id: dispatched.append((doc_id, user_id)),
    )

    assert count == 2
    assert dispatched == [(str(d), str(u)) for d, u in stuck]


async def test_sweep_is_a_noop_with_nothing_stuck(monkeypatch) -> None:
    await _patched(monkeypatch, [])
    dispatched: list[tuple[str, str]] = []

    count = await sweep_module.sweep_stuck_documents(
        resources=object(),
        now=datetime.now(UTC),
        stuck_after_seconds=900,
        dispatch=lambda doc_id, user_id: dispatched.append((doc_id, user_id)),
    )

    assert count == 0
    assert dispatched == []
