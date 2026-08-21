"""Unit tests for the document-chunk vector-store helpers (pure parts)."""

import uuid

import pytest

from shared.core.enums import Language
from shared.models.document import Chunk
from shared.vectorstore import build_chunk_payload, chunk_point_id

pytestmark = pytest.mark.unit


def _chunk(user_id: uuid.UUID, document_id: uuid.UUID) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=document_id,
        user_id=user_id,
        ordinal=2,
        page_start=10,
        page_end=12,
        chapter="I",
        section="1",
        text="some text",
        token_count=3,
        content_hash="abc",
    )


def test_point_id_is_the_chunk_uuid_string() -> None:
    chunk_id = uuid.uuid4()
    assert chunk_point_id(chunk_id) == str(chunk_id)


def test_payload_carries_owner_and_metadata() -> None:
    user_id, document_id = uuid.uuid4(), uuid.uuid4()
    chunk = _chunk(user_id, document_id)

    payload = build_chunk_payload(chunk, title="Book", author="Author", language=Language.DE)

    # Isolation: the owner id rides in every payload as a string (Qdrant filter key).
    assert payload["user_id"] == str(user_id)
    assert payload["document_id"] == str(document_id)
    assert payload["ordinal"] == 2
    assert payload["page_start"] == 10 and payload["page_end"] == 12
    assert payload["chapter"] == "I" and payload["section"] == "1"
    assert payload["title"] == "Book" and payload["author"] == "Author"
    assert payload["content_hash"] == "abc"
    assert payload["language"] == "de"


def test_payload_language_may_be_absent() -> None:
    chunk = _chunk(uuid.uuid4(), uuid.uuid4())
    payload = build_chunk_payload(chunk, title=None, author=None, language=None)
    assert payload["language"] is None
    assert payload["title"] is None
