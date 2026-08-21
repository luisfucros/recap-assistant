"""Unit tests for the pipeline's pure helpers (chunk-row construction, backoff)."""

import uuid

import pytest
from ingestion.pipeline import _build_chunk_rows
from ingestion.tasks import _backoff_seconds

from shared.ingestion_core.chunking import ChunkData

pytestmark = pytest.mark.unit


def _data(ordinal: int) -> ChunkData:
    return ChunkData(
        ordinal=ordinal,
        text=f"chunk {ordinal}",
        page_start=ordinal + 1,
        page_end=ordinal + 1,
        token_count=2,
        content_hash=f"h{ordinal}",
    )


def test_build_chunk_rows_sets_ownership_and_vector_identity() -> None:
    document_id, user_id = uuid.uuid4(), uuid.uuid4()
    rows = _build_chunk_rows([_data(0), _data(1)], document_id=document_id, user_id=user_id)

    assert [r.ordinal for r in rows] == [0, 1]
    for row in rows:
        assert row.document_id == document_id
        assert row.user_id == user_id
        # The Qdrant point id is the chunk's own uuid, so row and vector share identity.
        assert row.vector_id == str(row.id)
    # Distinct chunks get distinct ids.
    assert rows[0].id != rows[1].id


def test_build_chunk_rows_empty() -> None:
    assert _build_chunk_rows([], document_id=uuid.uuid4(), user_id=uuid.uuid4()) == []


def test_backoff_is_exponential() -> None:
    assert [_backoff_seconds(n) for n in range(4)] == [1, 2, 4, 8]
