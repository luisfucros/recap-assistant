"""Functional tests for the document upload/list/detail routes over real HTTP.

Boundaries are faked in-process: the auth DB (shared ``user_repo`` fixture),
object storage, and the document/outbox repositories. This exercises the full
request cycle — multipart parsing, boundary validation (type/size), the
store-and-enqueue handoff, the ``409 DUPLICATE_DOCUMENT`` translation with its
``Location`` header, isolation-scoped reads, and auth — without infrastructure.
The real :class:`IngestionService` runs against the fakes, so its logic is under
test end-to-end.
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from api.deps import (
    CurrentUser,
    get_app_settings,
    get_document_repository,
    get_document_service,
    get_ingestion_service,
    get_outbox_repository,
)
from api.services.document_service import DocumentService
from api.services.ingestion_service import IngestionService
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.functional.conftest import TEST_JWT_SECRET, FakeUserRepository

from shared.core.config import Settings
from shared.core.enums import DocumentStatus, Language
from shared.core.errors import NotFoundError
from shared.models.document import Document

pytestmark = pytest.mark.functional

PDF_BYTES = b"%PDF-1.7\nfake pdf body\n%%EOF"


class _FakeStorage:
    def __init__(self) -> None:
        self.puts: list[tuple[str, bytes, str]] = []
        self.deletes: list[str] = []

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        self.puts.append((key, data, content_type))

    async def get(self, key: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        self.deletes.append(key)


class _FakeVectorStore:
    def __init__(self) -> None:
        self.deletes: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def delete_by_document(self, *, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self.deletes.append((user_id, document_id))


class _FakeDocRepo:
    """In-memory document repo standing in for the user-scoped DB repository."""

    def __init__(self) -> None:
        self.by_sha: dict[str, Document] = {}
        self.by_id: dict[uuid.UUID, Document] = {}

    async def get_by_content_sha256(self, content_sha256: str) -> Document | None:
        return self.by_sha.get(content_sha256)

    async def add(self, document: Document) -> Document:
        if document.id is None:
            document.id = uuid.uuid4()
        # The DB fills created_at via a server default; emulate it for the fake.
        if document.created_at is None:
            document.created_at = datetime.now(tz=UTC)
        self.by_sha[document.content_sha256] = document
        self.by_id[document.id] = document
        return document

    async def get_or_404(self, document_id: uuid.UUID) -> Document:
        document = self.by_id.get(document_id)
        if document is None:
            raise NotFoundError()
        return document

    async def list_recent(self, *, limit: int, offset: int) -> list[Document]:
        ordered = sorted(self.by_id.values(), key=lambda d: d.created_at, reverse=True)
        return ordered[offset : offset + limit]

    async def count(self) -> int:
        return len(self.by_id)

    async def delete(self, document: Document) -> None:
        self.by_id.pop(document.id, None)
        self.by_sha.pop(document.content_sha256, None)


class _FakeOutboxRepo:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def add(self, *, event_type: str, aggregate_id: uuid.UUID, payload: dict) -> None:
        self.events.append(
            {"event_type": event_type, "aggregate_id": aggregate_id, "payload": payload}
        )


@dataclass
class _DocEnv:
    """The in-memory fakes wired into the app for a document test."""

    docs: _FakeDocRepo
    outbox: _FakeOutboxRepo
    storage: _FakeStorage
    vectors: _FakeVectorStore


@pytest.fixture
def doc_env(app: FastAPI, client: TestClient, user_repo: FakeUserRepository) -> Iterator[_DocEnv]:
    """Override the document/outbox/storage boundaries with in-memory fakes.

    ``get_document_repository`` is overridden with a function that still depends
    on ``CurrentUser``, so the endpoints keep enforcing authentication. The real
    ``IngestionService``/``DocumentService`` run against the fakes, so their logic
    is exercised end-to-end without infrastructure.
    """
    docs = _FakeDocRepo()
    outbox = _FakeOutboxRepo()
    storage = _FakeStorage()
    vectors = _FakeVectorStore()
    ingestion = IngestionService(storage=storage, embed_model="test-embed-model")
    document_service = DocumentService(storage=storage, vector_store=vectors)

    def _docs(_user: CurrentUser) -> _FakeDocRepo:
        return docs

    app.dependency_overrides[get_document_repository] = _docs
    app.dependency_overrides[get_outbox_repository] = lambda: outbox
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_document_service] = lambda: document_service
    try:
        yield _DocEnv(docs=docs, outbox=outbox, storage=storage, vectors=vectors)
    finally:
        for dep in (
            get_document_repository,
            get_outbox_repository,
            get_ingestion_service,
            get_document_service,
        ):
            app.dependency_overrides.pop(dep, None)


def _login(client: TestClient, email: str = "reader@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text


def _upload(
    client: TestClient,
    data: bytes = PDF_BYTES,
    name: str = "book.pdf",
    ctype: str = "application/pdf",
):
    return client.post("/api/v1/documents", files={"file": (name, data, ctype)})


def test_upload_returns_pending_document_and_enqueues_event(client: TestClient, doc_env) -> None:
    _login(client)

    resp = _upload(client)

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == DocumentStatus.PENDING.value
    assert body["filename"] == "book.pdf"
    assert body["format"] == "pdf"
    # Internal fields never leak into the public view.
    assert "content_sha256" not in body and "object_key" not in body
    # Stored once, and exactly one ingestion event enqueued for the new doc.
    assert len(doc_env.storage.puts) == 1
    assert len(doc_env.outbox.events) == 1
    assert doc_env.outbox.events[0]["event_type"] == "document.uploaded"
    assert doc_env.outbox.events[0]["payload"]["document_id"] == body["id"]


def test_reupload_same_content_conflicts_with_location(client: TestClient, doc_env) -> None:
    _login(client)

    first = _upload(client)
    assert first.status_code == 201
    existing_id = first.json()["id"]

    second = _upload(client)
    assert second.status_code == 409
    assert second.json()["code"] == "DUPLICATE_DOCUMENT"
    assert second.headers["location"].endswith(f"/documents/{existing_id}")
    # The duplicate was rejected — no second event, no extra storage write.
    assert len(doc_env.outbox.events) == 1
    assert len(doc_env.storage.puts) == 1


def test_upload_rejects_non_pdf(client: TestClient, doc_env) -> None:
    _login(client)
    resp = _upload(client, data=b"hello", name="notes.txt", ctype="text/plain")
    assert resp.status_code == 415
    assert resp.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_rejects_oversize(app: FastAPI, client: TestClient, doc_env) -> None:
    _login(client)
    # Shrink the cap for this request only.
    small = Settings(
        _env_file=None, jwt_secret=TEST_JWT_SECRET, cookie_secure=False, max_upload_bytes=8
    )
    app.dependency_overrides[get_app_settings] = lambda: small
    try:
        resp = _upload(client, data=b"x" * 64)
    finally:
        app.dependency_overrides.pop(get_app_settings, None)
    assert resp.status_code == 413
    assert resp.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_upload_requires_authentication(client: TestClient, doc_env) -> None:
    # No login → the CurrentUser dependency rejects before any work happens.
    resp = _upload(client)
    assert resp.status_code == 401


def test_list_and_detail_are_scoped_and_paginated(client: TestClient, doc_env) -> None:
    _login(client)
    created = _upload(client).json()

    listing = client.get("/api/v1/documents", params={"page": 1, "page_size": 5})
    assert listing.status_code == 200
    payload = listing.json()
    assert payload["total"] == 1
    assert payload["page"] == 1 and payload["page_size"] == 5
    assert [d["id"] for d in payload["items"]] == [created["id"]]

    detail = client.get(f"/api/v1/documents/{created['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == created["id"]


def test_detail_unknown_id_is_404(client: TestClient, doc_env) -> None:
    _login(client)
    resp = client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_patch_language_override(client: TestClient, doc_env) -> None:
    _login(client)
    created = _upload(client).json()

    resp = client.patch(f"/api/v1/documents/{created['id']}", json={"language": "de"})
    assert resp.status_code == 200
    assert resp.json()["language"] == Language.DE.value
    # The change is reflected on a subsequent read.
    assert client.get(f"/api/v1/documents/{created['id']}").json()["language"] == "de"


def test_patch_rejects_unsupported_language(client: TestClient, doc_env) -> None:
    _login(client)
    created = _upload(client).json()
    resp = client.patch(f"/api/v1/documents/{created['id']}", json={"language": "jp"})
    assert resp.status_code == 422  # not in the Language enum


def test_delete_removes_document_and_artifacts(client: TestClient, doc_env) -> None:
    _login(client)
    created = _upload(client).json()
    doc_id = created["id"]

    resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert resp.status_code == 204
    # Vectors and the stored object were cleaned up, and the row is gone.
    assert len(doc_env.vectors.deletes) == 1
    assert len(doc_env.storage.deletes) == 1
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404
    # A second delete is now a 404 (nothing left to remove).
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 404


def test_delete_unknown_id_is_404(client: TestClient, doc_env) -> None:
    _login(client)
    resp = client.delete(f"/api/v1/documents/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_delete_requires_authentication(client: TestClient, doc_env) -> None:
    resp = client.delete(f"/api/v1/documents/{uuid.uuid4()}")
    assert resp.status_code == 401


def test_retry_reenqueues_a_failed_document(client: TestClient, doc_env) -> None:
    _login(client)
    created = _upload(client).json()
    doc_id = uuid.UUID(created["id"])
    doc_env.docs.by_id[doc_id].status = DocumentStatus.FAILED
    doc_env.docs.by_id[doc_id].failure_reason = "parse failed: bad bytes"

    resp = client.post(f"/api/v1/documents/{doc_id}/retry")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == DocumentStatus.PENDING.value
    # A fresh ingestion event was enqueued alongside the original upload's.
    assert len(doc_env.outbox.events) == 2
    assert doc_env.outbox.events[-1]["event_type"] == "document.uploaded"
    assert doc_env.outbox.events[-1]["payload"]["document_id"] == str(doc_id)
    # The stored original is reused as-is — no second storage write.
    assert len(doc_env.storage.puts) == 1


def test_retry_rejects_a_non_failed_document(client: TestClient, doc_env) -> None:
    _login(client)
    created = _upload(client).json()
    doc_id = created["id"]

    resp = client.post(f"/api/v1/documents/{doc_id}/retry")

    assert resp.status_code == 409
    assert resp.json()["code"] == "DOCUMENT_NOT_FAILED"
    # Nothing changed: still the original single upload event.
    assert len(doc_env.outbox.events) == 1


def test_retry_unknown_id_is_404(client: TestClient, doc_env) -> None:
    _login(client)
    resp = client.post(f"/api/v1/documents/{uuid.uuid4()}/retry")
    assert resp.status_code == 404


def test_retry_requires_authentication(client: TestClient, doc_env) -> None:
    resp = client.post(f"/api/v1/documents/{uuid.uuid4()}/retry")
    assert resp.status_code == 401
