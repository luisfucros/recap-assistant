"""Unit tests for the long-term-memory vector-store helpers (pure parts)."""

import uuid

import pytest

from shared.core.enums import MemoryType
from shared.models.memory import LongTermMemory
from shared.vectorstore import build_memory_payload, memory_point_id

pytestmark = pytest.mark.unit


def _memory(
    user_id: uuid.UUID,
    *,
    document_id: uuid.UUID | None = None,
    type: MemoryType = MemoryType.SUMMARY,
    page_start: int | None = 1,
    page_end: int | None = 10,
) -> LongTermMemory:
    return LongTermMemory(
        id=uuid.uuid4(),
        user_id=user_id,
        document_id=document_id,
        type=type,
        content="Odysseus leaves Troy.",
        page_start=page_start,
        page_end=page_end,
    )


def test_point_id_is_the_memory_uuid_string() -> None:
    memory_id = uuid.uuid4()
    assert memory_point_id(memory_id) == str(memory_id)


def test_payload_carries_owner_type_and_page_range() -> None:
    user_id, document_id = uuid.uuid4(), uuid.uuid4()
    memory = _memory(user_id, document_id=document_id, page_start=21, page_end=50)

    payload = build_memory_payload(memory)

    assert payload["user_id"] == str(user_id)
    assert payload["type"] == "summary"
    assert payload["document_id"] == str(document_id)
    assert payload["page_start"] == 21
    assert payload["page_end"] == 50


def test_payload_document_and_page_range_are_null_for_user_level_facts() -> None:
    memory = _memory(
        uuid.uuid4(), document_id=None, type=MemoryType.PREFERENCE, page_start=None, page_end=None
    )

    payload = build_memory_payload(memory)

    assert payload["document_id"] is None
    assert payload["page_start"] is None
    assert payload["page_end"] is None
    assert payload["type"] == "preference"
