"""Unit tests for ``MemoryVectorStore.search``/``.delete`` filter construction.

The Qdrant client is faked at the boundary; the test inspects the filters the
store builds. The load-bearing assertion is that **every** search/delete carries
a ``user_id`` condition (per-user isolation is enforced in the store, not the
caller), and that optional type/document/spoiler-safe filters are added only
when requested.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from shared.core.enums import MemoryType
from shared.vectorstore import MemoryVectorStore, ScoredMemory

pytestmark = pytest.mark.unit


class _FakeClient:
    """Captures ``query_points``/``delete`` kwargs and returns preset points."""

    def __init__(self, points: list[Any] | None = None) -> None:
        self.query_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self._points = points or []

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self._points)

    async def delete(self, **kwargs: Any) -> None:
        self.delete_calls.append(kwargs)


def _conditions_by_key(query_filter: models.Filter) -> dict[str, Any]:
    return {cond.key: cond for cond in query_filter.must}  # type: ignore[union-attr]


async def test_search_always_filters_by_user_id() -> None:
    client = _FakeClient()
    store = MemoryVectorStore(client, collection="long_term_memory")  # type: ignore[arg-type]
    owner = uuid.uuid4()

    await store.search(user_id=owner, query_vector=[0.1, 0.2], limit=5)

    kwargs = client.query_calls[-1]
    assert kwargs["collection_name"] == "long_term_memory"
    assert kwargs["limit"] == 5
    conditions = _conditions_by_key(kwargs["query_filter"])
    assert conditions["user_id"].match.value == str(owner)
    assert set(conditions) == {"user_id"}


async def test_search_adds_type_document_and_spoiler_safe_conditions() -> None:
    client = _FakeClient()
    store = MemoryVectorStore(client, collection="long_term_memory")  # type: ignore[arg-type]
    owner, document_id = uuid.uuid4(), uuid.uuid4()

    await store.search(
        user_id=owner,
        query_vector=[0.1],
        limit=8,
        type=MemoryType.SUMMARY,
        document_id=document_id,
        max_page_end=50,
    )

    conditions = _conditions_by_key(client.query_calls[-1]["query_filter"])
    assert conditions["type"].match.value == "summary"
    assert conditions["document_id"].match.value == str(document_id)
    assert conditions["page_end"].range.lte == 50


async def test_search_omits_page_bound_when_not_requested() -> None:
    client = _FakeClient()
    store = MemoryVectorStore(client, collection="long_term_memory")  # type: ignore[arg-type]

    await store.search(user_id=uuid.uuid4(), query_vector=[0.1], limit=3)

    conditions = _conditions_by_key(client.query_calls[-1]["query_filter"])
    assert "page_end" not in conditions
    assert "type" not in conditions
    assert "document_id" not in conditions


async def test_search_maps_points_to_scored_memories() -> None:
    point = SimpleNamespace(id=uuid.uuid4(), score=0.77, payload={"type": "summary"})
    client = _FakeClient(points=[point])
    store = MemoryVectorStore(client, collection="long_term_memory")  # type: ignore[arg-type]

    results = await store.search(user_id=uuid.uuid4(), query_vector=[0.1], limit=1)

    assert results == [ScoredMemory(id=str(point.id), score=0.77, payload={"type": "summary"})]


async def test_delete_filters_by_user_id_and_point_id() -> None:
    client = _FakeClient()
    store = MemoryVectorStore(client, collection="long_term_memory")  # type: ignore[arg-type]
    owner, memory_id = uuid.uuid4(), uuid.uuid4()

    await store.delete(user_id=owner, memory_id=memory_id)

    kwargs = client.delete_calls[-1]
    assert kwargs["collection_name"] == "long_term_memory"
    conditions = kwargs["points_selector"].filter.must
    assert conditions[0].key == "user_id"
    assert conditions[0].match.value == str(owner)
    assert conditions[1].has_id == [str(memory_id)]
