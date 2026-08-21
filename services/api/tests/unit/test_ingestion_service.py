"""Unit tests for the API-side ingestion handoff.

The service's collaborators — object storage, the document/outbox repositories,
and the DB session — are faked at their boundaries. These assert the handoff
contract: content-addressed storage, a ``pending`` row plus a matching outbox
event committed together, and race-safe duplicate rejection.
"""

import uuid

import pytest
from api.services.ingestion_service import (
    DocumentNotFailedError,
    DuplicateDocumentError,
    IngestionService,
)
from sqlalchemy.exc import IntegrityError

from shared.core.enums import DocumentFormat, DocumentStatus
from shared.core.errors import NotFoundError
from shared.core.events import DOCUMENT_UPLOADED
from shared.ingestion_core.content_address import object_key, sha256_hexdigest
from shared.models.document import Document

pytestmark = pytest.mark.unit


class _FakeStorage:
    def __init__(self) -> None:
        self.puts: list[tuple[str, bytes, str]] = []

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        self.puts.append((key, data, content_type))

    async def get(self, key: str) -> bytes:  # pragma: no cover - unused here
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


class _FakeDocumentRepo:
    """Fakes the user-scoped document repo; ``by_sha`` scripts the lookups."""

    def __init__(
        self,
        user_id: uuid.UUID,
        by_sha: list[Document | None] | None = None,
        by_id: dict[uuid.UUID, Document] | None = None,
    ) -> None:
        self._user_id = user_id
        self._by_sha = list(by_sha or [None])
        self._by_id = dict(by_id or {})
        self.added: list[Document] = []
        self.add_raises: Exception | None = None

    async def get_by_content_sha256(self, content_sha256: str) -> Document | None:
        # Pop scripted results so pre-check and post-conflict lookup can differ.
        return self._by_sha.pop(0) if self._by_sha else None

    async def add(self, document: Document) -> Document:
        if self.add_raises is not None:
            raise self.add_raises
        if document.id is None:
            document.id = uuid.uuid4()
        self.added.append(document)
        return document

    async def get_or_404(self, document_id: uuid.UUID) -> Document:
        document = self._by_id.get(document_id)
        if document is None:
            raise NotFoundError()
        return document


class _FakeOutboxRepo:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def add(self, *, event_type: str, aggregate_id: uuid.UUID, payload: dict) -> None:
        self.events.append(
            {"event_type": event_type, "aggregate_id": aggregate_id, "payload": payload}
        )


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def _integrity_error() -> IntegrityError:
    return IntegrityError("INSERT ...", {}, Exception("duplicate key"))


def _service() -> IngestionService:
    return IngestionService(storage=_FakeStorage(), embed_model="text-embedding-3-small")


async def test_upload_stores_and_enqueues_pending_document() -> None:
    user_id = uuid.uuid4()
    storage = _FakeStorage()
    service = IngestionService(storage=storage, embed_model="text-embedding-3-small")
    documents = _FakeDocumentRepo(user_id)
    outbox = _FakeOutboxRepo()
    session = _FakeSession()
    data = b"%PDF-1.7 fake"

    document = await service.upload(
        session=session,
        documents=documents,
        outbox=outbox,
        user_id=user_id,
        filename="book.pdf",
        content_type="application/pdf",
        document_format=DocumentFormat.PDF,
        data=data,
    )

    sha = sha256_hexdigest(data)
    assert document.status is DocumentStatus.PENDING
    assert document.content_sha256 == sha
    assert document.embed_model == "text-embedding-3-small"
    assert document.user_id == user_id
    # Stored content-addressed, exactly once.
    assert storage.puts == [(object_key(user_id, sha, "pdf"), data, "application/pdf")]
    # Exactly one matching outbox event, carrying both ids for the consumer.
    assert len(outbox.events) == 1
    event = outbox.events[0]
    assert event["event_type"] == DOCUMENT_UPLOADED
    assert event["aggregate_id"] == document.id
    assert event["payload"] == {"document_id": str(document.id), "user_id": str(user_id)}
    # Row + event committed together.
    assert session.commits == 1


async def test_upload_rejects_known_duplicate_before_storing() -> None:
    user_id = uuid.uuid4()
    existing = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="orig.pdf",
        object_key="k",
        content_sha256="sha",
        format=DocumentFormat.PDF,
        embed_model="m",
    )
    storage = _FakeStorage()
    service = IngestionService(storage=storage, embed_model="m")
    documents = _FakeDocumentRepo(user_id, by_sha=[existing])
    outbox = _FakeOutboxRepo()
    session = _FakeSession()

    with pytest.raises(DuplicateDocumentError) as excinfo:
        await service.upload(
            session=session,
            documents=documents,
            outbox=outbox,
            user_id=user_id,
            filename="dup.pdf",
            content_type="application/pdf",
            document_format=DocumentFormat.PDF,
            data=b"whatever",
        )

    assert excinfo.value.existing_id == existing.id
    # Fast path: no storage write, no event, no commit for a known duplicate.
    assert storage.puts == []
    assert outbox.events == []
    assert session.commits == 0


async def test_upload_race_maps_integrity_error_to_duplicate() -> None:
    user_id = uuid.uuid4()
    winner = Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="winner.pdf",
        object_key="k",
        content_sha256="sha",
        format=DocumentFormat.PDF,
        embed_model="m",
    )
    service = _service()
    # Pre-check sees nothing; the post-conflict lookup finds the race winner.
    documents = _FakeDocumentRepo(user_id, by_sha=[None, winner])
    documents.add_raises = _integrity_error()
    outbox = _FakeOutboxRepo()
    session = _FakeSession()

    with pytest.raises(DuplicateDocumentError) as excinfo:
        await service.upload(
            session=session,
            documents=documents,
            outbox=outbox,
            user_id=user_id,
            filename="loser.pdf",
            content_type="application/pdf",
            document_format=DocumentFormat.PDF,
            data=b"racing bytes",
        )

    assert excinfo.value.existing_id == winner.id
    # Lost the race: rolled back, nothing committed.
    assert session.rollbacks == 1
    assert session.commits == 0


def _failed_document(user_id: uuid.UUID) -> Document:
    return Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="book.pdf",
        object_key="k",
        content_sha256="sha",
        format=DocumentFormat.PDF,
        embed_model="m",
        status=DocumentStatus.FAILED,
        failure_reason="parse failed: bad bytes",
    )


async def test_retry_resets_a_failed_document_and_reenqueues() -> None:
    user_id = uuid.uuid4()
    document = _failed_document(user_id)
    service = _service()
    documents = _FakeDocumentRepo(user_id, by_id={document.id: document})
    outbox = _FakeOutboxRepo()
    session = _FakeSession()

    result = await service.retry(
        session=session, documents=documents, outbox=outbox, document_id=document.id
    )

    assert result is document
    assert document.status is DocumentStatus.PENDING
    assert document.failure_reason is None
    assert len(outbox.events) == 1
    event = outbox.events[0]
    assert event["event_type"] == DOCUMENT_UPLOADED
    assert event["aggregate_id"] == document.id
    assert event["payload"] == {"document_id": str(document.id), "user_id": str(user_id)}
    assert session.commits == 1


async def test_retry_rejects_a_non_failed_document() -> None:
    user_id = uuid.uuid4()
    document = _failed_document(user_id)
    document.status = DocumentStatus.INDEXED
    service = _service()
    documents = _FakeDocumentRepo(user_id, by_id={document.id: document})
    outbox = _FakeOutboxRepo()
    session = _FakeSession()

    with pytest.raises(DocumentNotFailedError) as excinfo:
        await service.retry(
            session=session, documents=documents, outbox=outbox, document_id=document.id
        )

    assert excinfo.value.document_id == document.id
    assert outbox.events == []
    assert session.commits == 0


async def test_retry_unknown_document_is_not_found() -> None:
    user_id = uuid.uuid4()
    service = _service()
    documents = _FakeDocumentRepo(user_id)
    outbox = _FakeOutboxRepo()
    session = _FakeSession()

    with pytest.raises(NotFoundError):
        await service.retry(
            session=session, documents=documents, outbox=outbox, document_id=uuid.uuid4()
        )
    assert session.commits == 0
