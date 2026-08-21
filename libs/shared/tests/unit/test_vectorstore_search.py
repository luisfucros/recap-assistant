"""Unit tests for ``ChunkVectorStore.search`` filter construction.

The Qdrant client is faked at the boundary; the test inspects the ``query_filter``
the store builds. The load-bearing assertion is that **every** search carries a
``user_id`` condition (per-user isolation is enforced in the store, not the
caller), and that optional read-range / document / structure filters are added
only when requested.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from shared.vectorstore import ChunkVectorStore, ScoredChunk

pytestmark = pytest.mark.unit


class _FakeClient:
    """Captures ``query_points`` kwargs and returns preset scored points."""

    def __init__(self, points: list[Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._points = points or []

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(points=self._points)


def _conditions_by_key(query_filter: models.Filter) -> dict[str, models.FieldCondition]:
    return {cond.key: cond for cond in query_filter.must}  # type: ignore[union-attr]


async def test_search_always_filters_by_user_id() -> None:
    client = _FakeClient()
    store = ChunkVectorStore(client, collection="document_chunks")  # type: ignore[arg-type]
    owner = uuid.uuid4()

    await store.search(user_id=owner, query_vector=[0.1, 0.2], limit=5)

    kwargs = client.calls[-1]
    assert kwargs["collection_name"] == "document_chunks"
    assert kwargs["limit"] == 5
    conditions = _conditions_by_key(kwargs["query_filter"])
    assert "user_id" in conditions
    assert conditions["user_id"].match.value == str(owner)
    # With no other options, user_id is the only filter.
    assert set(conditions) == {"user_id"}


async def test_search_adds_read_range_and_document_conditions() -> None:
    client = _FakeClient()
    store = ChunkVectorStore(client, collection="document_chunks")  # type: ignore[arg-type]
    owner, document_id = uuid.uuid4(), uuid.uuid4()

    await store.search(
        user_id=owner,
        query_vector=[0.1],
        limit=8,
        document_id=document_id,
        max_page_end=84,
        chapter="5",
        section="intro",
        language="en",
    )

    conditions = _conditions_by_key(client.calls[-1]["query_filter"])
    assert conditions["document_id"].match.value == str(document_id)
    assert conditions["page_end"].range.lte == 84
    assert conditions["chapter"].match.value == "5"
    assert conditions["section"].match.value == "intro"
    assert conditions["language"].match.value == "en"


async def test_search_omits_page_bound_when_not_requested() -> None:
    client = _FakeClient()
    store = ChunkVectorStore(client, collection="document_chunks")  # type: ignore[arg-type]

    await store.search(user_id=uuid.uuid4(), query_vector=[0.1], limit=3)

    conditions = _conditions_by_key(client.calls[-1]["query_filter"])
    assert "page_end" not in conditions


async def test_search_maps_points_to_scored_chunks() -> None:
    point = SimpleNamespace(id=uuid.uuid4(), score=0.87, payload={"document_id": "d"})
    client = _FakeClient(points=[point])
    store = ChunkVectorStore(client, collection="document_chunks")  # type: ignore[arg-type]

    results = await store.search(user_id=uuid.uuid4(), query_vector=[0.1], limit=1)

    assert results == [ScoredChunk(id=str(point.id), score=0.87, payload={"document_id": "d"})]
