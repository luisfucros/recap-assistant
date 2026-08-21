"""Unit tests for the document lifecycle service (delete + metadata update)."""

import uuid

import pytest
from api.services.document_service import DocumentService

from shared.core.enums import DocumentFormat, Language
from shared.core.errors import NotFoundError
from shared.models.document import Document

pytestmark = pytest.mark.unit


class _FakeStorage:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def put(self, key, data, content_type) -> None:
        raise NotImplementedError

    async def get(self, key) -> bytes:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        self.deleted.append(key)


class _FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def delete_by_document(self, *, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
        self.deleted.append((user_id, document_id))


class _FakeDocRepo:
    def __init__(self, document: Document | None) -> None:
        self._document = document
        self.deleted: list[Document] = []

    async def get_or_404(self, document_id: uuid.UUID) -> Document:
        if self._document is None or self._document.id != document_id:
            raise NotFoundError()
        return self._document

    async def delete(self, document: Document) -> None:
        self.deleted.append(document)


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


def _document(user_id: uuid.UUID) -> Document:
    return Document(
        id=uuid.uuid4(),
        user_id=user_id,
        filename="book.pdf",
        object_key=f"{user_id}/sha256/abc.pdf",
        content_sha256="abc",
        format=DocumentFormat.PDF,
        embed_model="m",
    )


async def test_delete_removes_vectors_object_and_row_then_commits() -> None:
    user_id = uuid.uuid4()
    doc = _document(user_id)
    storage, vectors = _FakeStorage(), _FakeVectorStore()
    repo, session = _FakeDocRepo(doc), _FakeSession()
    service = DocumentService(storage=storage, vector_store=vectors)

    await service.delete(session=session, documents=repo, user_id=user_id, document_id=doc.id)

    assert vectors.deleted == [(user_id, doc.id)]
    assert storage.deleted == [doc.object_key]
    assert repo.deleted == [doc]
    assert session.commits == 1


async def test_delete_unknown_document_is_404_and_touches_nothing() -> None:
    storage, vectors = _FakeStorage(), _FakeVectorStore()
    repo, session = _FakeDocRepo(None), _FakeSession()
    service = DocumentService(storage=storage, vector_store=vectors)

    with pytest.raises(NotFoundError):
        await service.delete(
            session=session, documents=repo, user_id=uuid.uuid4(), document_id=uuid.uuid4()
        )
    assert vectors.deleted == [] and storage.deleted == [] and session.commits == 0


async def test_update_language_sets_and_commits() -> None:
    user_id = uuid.uuid4()
    doc = _document(user_id)
    service = DocumentService(storage=_FakeStorage(), vector_store=_FakeVectorStore())
    session = _FakeSession()

    updated = await service.update_language(
        session=session, documents=_FakeDocRepo(doc), document_id=doc.id, language=Language.DE
    )
    assert updated.language is Language.DE
    assert session.commits == 1


async def test_update_language_none_is_noop_but_still_commits() -> None:
    user_id = uuid.uuid4()
    doc = _document(user_id)
    doc.language = Language.EN
    service = DocumentService(storage=_FakeStorage(), vector_store=_FakeVectorStore())

    updated = await service.update_language(
        session=_FakeSession(), documents=_FakeDocRepo(doc), document_id=doc.id, language=None
    )
    assert updated.language is Language.EN
