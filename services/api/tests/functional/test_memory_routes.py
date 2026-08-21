"""Functional tests for the long-term memory view/delete routes (FR-4.5).

Boundaries are faked in-process: the memory repository and its vector store.
The real :class:`MemoryService` runs against the fakes, so its logic (routing
to ``list_recent``/``list_by_type``, deleting the vector point before the row)
is exercised end-to-end without infrastructure.
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from api.deps import CurrentUser, get_memory_repository, get_memory_service
from api.services.memory_service import MemoryService
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

from shared.core.enums import MemoryType
from shared.core.errors import NotFoundError
from shared.models.memory import LongTermMemory

pytestmark = pytest.mark.functional


class _FakeEmbedder:
    @property
    def dim(self) -> int:
        return 8

    async def embed(self, texts, *, batch_size=None) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError  # list/delete never embed


class _FakeMemoryVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def delete(self, *, user_id: uuid.UUID, memory_id: uuid.UUID) -> None:
        self.deleted.append((user_id, memory_id))


class _FakeMemoryRepo:
    """In-memory memory repo standing in for the user-scoped DB repository."""

    def __init__(self, user_id: uuid.UUID | None = None) -> None:
        self.user_id = user_id or uuid.uuid4()
        self.by_id: dict[uuid.UUID, LongTermMemory] = {}

    async def list_recent(self, *, limit: int, offset: int) -> list[LongTermMemory]:
        ordered = sorted(self.by_id.values(), key=lambda m: m.created_at, reverse=True)
        return ordered[offset : offset + limit]

    async def list_by_type(
        self, memory_type: MemoryType, *, limit: int, offset: int
    ) -> list[LongTermMemory]:
        matching = [m for m in self.by_id.values() if m.type == memory_type]
        ordered = sorted(matching, key=lambda m: m.created_at, reverse=True)
        return ordered[offset : offset + limit]

    async def count(self) -> int:
        return len(self.by_id)

    async def get_or_404(self, memory_id: uuid.UUID) -> LongTermMemory:
        memory = self.by_id.get(memory_id)
        if memory is None:
            raise NotFoundError()
        return memory

    async def delete(self, memory: LongTermMemory) -> None:
        self.by_id.pop(memory.id, None)


@dataclass
class _MemoryEnv:
    """The in-memory fakes wired into the app for a memory test."""

    memories: _FakeMemoryRepo
    vectors: _FakeMemoryVectorStore


@pytest.fixture
def memory_env(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> Iterator[_MemoryEnv]:
    """Override the memory repository/service boundaries with in-memory fakes.

    ``get_memory_repository`` is overridden with a function that still depends
    on ``CurrentUser``, so the endpoints keep enforcing authentication. The real
    ``MemoryService`` runs against the fakes.
    """
    memories = _FakeMemoryRepo()
    vectors = _FakeMemoryVectorStore()
    memory_service = MemoryService(embedder=_FakeEmbedder(), vector_store=vectors)  # type: ignore[arg-type]

    def _memories(_user: CurrentUser) -> _FakeMemoryRepo:
        return memories

    app.dependency_overrides[get_memory_repository] = _memories
    app.dependency_overrides[get_memory_service] = lambda: memory_service
    try:
        yield _MemoryEnv(memories=memories, vectors=vectors)
    finally:
        for dep in (get_memory_repository, get_memory_service):
            app.dependency_overrides.pop(dep, None)


def _login(client: TestClient, email: str = "reader@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text


def _seed(
    memories: _FakeMemoryRepo,
    *,
    type: MemoryType = MemoryType.PREFERENCE,
    content: str = "likes sci-fi",
    document_id: uuid.UUID | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    created_at: datetime | None = None,
) -> LongTermMemory:
    memory = LongTermMemory(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        type=type,
        content=content,
        document_id=document_id,
        page_start=page_start,
        page_end=page_end,
        created_at=created_at or datetime.now(tz=UTC),
    )
    memories.by_id[memory.id] = memory
    return memory


def test_list_memories_returns_a_page_newest_first(client: TestClient, memory_env) -> None:
    _login(client)
    older = _seed(memory_env.memories, content="older", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    newer = _seed(memory_env.memories, content="newer", created_at=datetime(2026, 2, 1, tzinfo=UTC))

    resp = client.get("/api/v1/memory")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert [item["id"] for item in body["items"]] == [str(newer.id), str(older.id)]
    # Internal fields never leak into the public view.
    assert "user_id" not in body["items"][0] and "embedding_id" not in body["items"][0]


def test_list_memories_filters_by_type(client: TestClient, memory_env) -> None:
    _login(client)
    _seed(memory_env.memories, type=MemoryType.PREFERENCE, content="likes sci-fi")
    summary = _seed(
        memory_env.memories,
        type=MemoryType.SUMMARY,
        content="recap",
        document_id=uuid.uuid4(),
        page_start=1,
        page_end=20,
    )

    resp = client.get("/api/v1/memory", params={"type": "summary"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [item["id"] for item in body["items"]] == [str(summary.id)]
    assert body["items"][0]["page_start"] == 1 and body["items"][0]["page_end"] == 20


def test_list_memories_respects_pagination(client: TestClient, memory_env) -> None:
    _login(client)
    for i in range(3):
        _seed(memory_env.memories, content=f"memory {i}")

    resp = client.get("/api/v1/memory", params={"page": 1, "page_size": 2})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["page"] == 1 and body["page_size"] == 2


def test_list_memories_requires_authentication(client: TestClient, memory_env) -> None:
    resp = client.get("/api/v1/memory")
    assert resp.status_code == 401


def test_delete_memory_removes_the_row_and_its_vector_point(client: TestClient, memory_env) -> None:
    _login(client)
    memory = _seed(memory_env.memories)

    resp = client.delete(f"/api/v1/memory/{memory.id}")

    assert resp.status_code == 204
    assert memory.id not in memory_env.memories.by_id
    # The vector delete is keyed by the repository's (authenticated caller's)
    # user_id, never the row's own — the owner is injected server-side.
    assert memory_env.vectors.deleted == [(memory_env.memories.user_id, memory.id)]


def test_delete_memory_unknown_id_is_404(client: TestClient, memory_env) -> None:
    _login(client)
    resp = client.delete(f"/api/v1/memory/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_delete_memory_requires_authentication(client: TestClient, memory_env) -> None:
    memory = _seed(memory_env.memories)
    resp = client.delete(f"/api/v1/memory/{memory.id}")
    assert resp.status_code == 401
