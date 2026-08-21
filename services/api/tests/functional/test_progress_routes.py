"""Functional tests for the /progress and /analytics routes over real HTTP.

Boundaries are faked in-process: the auth DB (shared ``user_repo`` fixture), the
document/progress/event repositories, and Redis (for the analytics cache). The
real :class:`ProgressService`/:class:`AnalyticsService` run against the fakes, so
their logic — position/status updates, event emission, the per-document
spoiler-safe override, reading-list grouping, and analytics aggregation — is
exercised end-to-end without infrastructure.
"""

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from api.deps import (
    CurrentUser,
    get_analytics_service,
    get_document_repository,
    get_progress_repository,
    get_reading_event_repository,
)
from api.services.analytics_service import AnalyticsService
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

from shared.core.enums import DocumentFormat, ReadingStatus
from shared.core.errors import NotFoundError
from shared.models.document import Document
from shared.models.reading import ReadingEvent, ReadingProgress

pytestmark = pytest.mark.functional


class _FakeDocRepo:
    """Holds pre-seeded documents; only ``get_or_404`` is exercised here."""

    def __init__(self) -> None:
        self.by_id: dict[uuid.UUID, Document] = {}

    def seed(self, *, page_count: int | None = 100) -> Document:
        doc = Document(
            id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            filename="book.pdf",
            object_key="k",
            content_sha256=uuid.uuid4().hex,
            format=DocumentFormat.PDF,
            embed_model="m",
            page_count=page_count,
        )
        self.by_id[doc.id] = doc
        return doc

    async def get_or_404(self, document_id: uuid.UUID) -> Document:
        doc = self.by_id.get(document_id)
        if doc is None:
            raise NotFoundError()
        return doc


class _FakeProgressRepo:
    """In-memory reading-progress repo, keyed by document id (single test user)."""

    def __init__(self) -> None:
        self.user_id: uuid.UUID | None = None
        self.rows: dict[uuid.UUID, ReadingProgress] = {}

    async def get_by_document(self, document_id: uuid.UUID) -> ReadingProgress | None:
        return self.rows.get(document_id)

    async def add(self, row: ReadingProgress) -> ReadingProgress:
        self.rows[row.document_id] = row
        return row

    async def list_by_status(
        self, status: ReadingStatus, *, limit: int = 100, offset: int = 0
    ) -> Sequence[ReadingProgress]:
        rows = [r for r in self.rows.values() if r.status == status]
        rows.sort(key=lambda r: r.last_accessed_at or datetime.min, reverse=True)
        return rows[offset : offset + limit]

    async def list_recent(self, *, limit: int = 10, offset: int = 0) -> Sequence[ReadingProgress]:
        rows = sorted(
            self.rows.values(), key=lambda r: r.last_accessed_at or datetime.min, reverse=True
        )
        return rows[offset : offset + limit]

    async def count_by_status(self, status: ReadingStatus) -> int:
        return sum(1 for r in self.rows.values() if r.status == status)


class _FakeEventRepo:
    def __init__(self) -> None:
        self.user_id: uuid.UUID | None = None
        self.events: list[ReadingEvent] = []

    async def add(self, event: ReadingEvent) -> ReadingEvent:
        # The DB fills occurred_at via a server default; emulate it for the fake
        # so analytics (which filters by occurred_at) sees the event.
        if event.occurred_at is None:
            event.occurred_at = datetime.now(tz=UTC)
        self.events.append(event)
        return event

    async def list_since(self, since, *, limit: int = 10_000) -> list[ReadingEvent]:
        return [e for e in self.events if e.occurred_at is not None and e.occurred_at >= since]


class _FakeRedis:
    """Cache miss always: forces the analytics service to recompute each call."""

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        return None


@dataclass
class _ProgressEnv:
    docs: _FakeDocRepo
    progress: _FakeProgressRepo
    events: _FakeEventRepo
    document_id: uuid.UUID = field(init=False)

    def __post_init__(self) -> None:
        self.document_id = self.docs.seed().id


@pytest.fixture
def progress_env(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> Iterator[_ProgressEnv]:
    """Override the document/progress/event repos + analytics service with fakes."""
    docs, progress, events = _FakeDocRepo(), _FakeProgressRepo(), _FakeEventRepo()
    analytics = AnalyticsService(redis=_FakeRedis(), ttl_seconds=300)  # type: ignore[arg-type]

    def _docs(_user: CurrentUser) -> _FakeDocRepo:
        return docs

    def _progress(user: CurrentUser) -> _FakeProgressRepo:
        progress.user_id = user.id
        return progress

    def _events(user: CurrentUser) -> _FakeEventRepo:
        events.user_id = user.id
        return events

    app.dependency_overrides[get_document_repository] = _docs
    app.dependency_overrides[get_progress_repository] = _progress
    app.dependency_overrides[get_reading_event_repository] = _events
    app.dependency_overrides[get_analytics_service] = lambda: analytics
    try:
        yield _ProgressEnv(docs=docs, progress=progress, events=events)
    finally:
        for dep in (
            get_document_repository,
            get_progress_repository,
            get_reading_event_repository,
            get_analytics_service,
        ):
            app.dependency_overrides.pop(dep, None)


def _login(client: TestClient, email: str = "reader@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text


# --- update + read-back -------------------------------------------------- #


def test_update_position_promotes_status_and_reads_back(
    client: TestClient, progress_env: _ProgressEnv
) -> None:
    _login(client)
    doc_id = str(progress_env.document_id)

    resp = client.put(f"/api/v1/progress/{doc_id}", json={"current_page": 20})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["current_page"] == 20
    assert body["status"] == ReadingStatus.READING.value

    got = client.get(f"/api/v1/progress/{doc_id}")
    assert got.status_code == 200
    assert got.json()["current_page"] == 20
    # A forward move recorded a position event.
    assert len(progress_env.events.events) >= 1


def test_get_untracked_document_is_404(client: TestClient, progress_env: _ProgressEnv) -> None:
    _login(client)
    resp = client.get(f"/api/v1/progress/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_update_rejects_page_past_end(client: TestClient, progress_env: _ProgressEnv) -> None:
    _login(client)
    resp = client.put(f"/api/v1/progress/{progress_env.document_id}", json={"current_page": 500})
    assert resp.status_code == 422  # page_count is 100


def test_cancel_via_status_then_appears_in_reading_list(
    client: TestClient, progress_env: _ProgressEnv
) -> None:
    _login(client)
    doc_id = str(progress_env.document_id)
    client.put(f"/api/v1/progress/{doc_id}", json={"current_page": 10})

    resp = client.put(f"/api/v1/progress/{doc_id}", json={"status": "cancelled"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    listing = client.get("/api/v1/progress").json()
    assert [r["document_id"] for r in listing["cancelled"]] == [doc_id]
    assert listing["reading"] == []


def test_per_document_spoiler_override_roundtrips(
    client: TestClient, progress_env: _ProgressEnv
) -> None:
    _login(client)
    doc_id = str(progress_env.document_id)

    # Set the per-document override to False.
    resp = client.put(f"/api/v1/progress/{doc_id}", json={"spoiler_safe": False})
    assert resp.status_code == 200
    assert resp.json()["spoiler_safe"] is False

    # Explicit null clears it back to "defer to user default".
    cleared = client.put(f"/api/v1/progress/{doc_id}", json={"spoiler_safe": None})
    assert cleared.status_code == 200
    assert cleared.json()["spoiler_safe"] is None


def test_progress_requires_authentication(client: TestClient, progress_env: _ProgressEnv) -> None:
    assert client.get("/api/v1/progress").status_code == 401
    assert (
        client.put(f"/api/v1/progress/{uuid.uuid4()}", json={"current_page": 1}).status_code == 401
    )


# --- analytics ----------------------------------------------------------- #


def test_analytics_reflects_recorded_reading(
    client: TestClient, progress_env: _ProgressEnv
) -> None:
    _login(client)
    doc_id = str(progress_env.document_id)

    # Two forward moves today: 0→10 then 10→25 = 25 pages read.
    client.put(f"/api/v1/progress/{doc_id}", json={"current_page": 10})
    client.put(f"/api/v1/progress/{doc_id}", json={"current_page": 25})

    resp = client.get("/api/v1/analytics")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pages_read"] == 25
    assert body["documents_started"] == 1
    assert body["current_streak_days"] >= 1
    assert body["window_days"] == 30


def test_analytics_requires_authentication(client: TestClient, progress_env: _ProgressEnv) -> None:
    assert client.get("/api/v1/analytics").status_code == 401
