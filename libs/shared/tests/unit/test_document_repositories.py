"""Unit tests for the document/chunk/outbox repositories.

No database: a capturing fake session records the statements the repositories
build, which are compiled to SQL and inspected. This asserts the load-bearing
per-user filter is present on every user-scoped query, and that the outbox poll
selects only unprocessed rows — the boundary (``session.execute``) is mocked,
the query-building logic under test is real.
"""

import uuid
from typing import Any

import pytest

from shared.core.enums import DocumentFormat, DocumentStatus, Language
from shared.models.document import Chunk
from shared.repositories import (
    ChunkRepository,
    DocumentRepository,
    OutboxRepository,
)

pytestmark = pytest.mark.unit


class _FakeResult:
    """Stand-in for a SQLAlchemy ``Result`` returning preset values."""

    def __init__(self, *, value: Any = None, seq: tuple[Any, ...] = ()) -> None:
        self._value = value
        self._seq = seq

    def scalar_one_or_none(self) -> Any:
        return self._value

    def scalar_one(self) -> Any:
        return self._value

    def scalars(self) -> "_FakeScalars":
        return _FakeScalars(self._seq)


class _FakeScalars:
    def __init__(self, seq: tuple[Any, ...]) -> None:
        self._seq = seq

    def all(self) -> list[Any]:
        return list(self._seq)


class _CapturingSession:
    """Async session double that records executed statements."""

    def __init__(self, result: _FakeResult | None = None) -> None:
        self.statements: list[Any] = []
        self.added: list[Any] = []
        self._result = result or _FakeResult()

    async def execute(self, statement: Any) -> _FakeResult:
        self.statements.append(statement)
        return self._result

    def add(self, entity: Any) -> None:
        self.added.append(entity)

    def add_all(self, entities: Any) -> None:
        self.added.extend(entities)

    async def flush(self) -> None:
        return None


def _sql(statement: Any) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True}))


# --- enums --------------------------------------------------------------- #


def test_document_status_values() -> None:
    assert [s.value for s in DocumentStatus] == ["pending", "processing", "indexed", "failed"]


def test_document_format_values() -> None:
    assert [f.value for f in DocumentFormat] == ["pdf"]


# --- DocumentRepository -------------------------------------------------- #


async def test_get_by_content_sha256_scopes_user_and_hash() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(value=None))
    repo = DocumentRepository(session, owner)  # type: ignore[arg-type]

    await repo.get_by_content_sha256("deadbeef")

    sql = _sql(session.statements[-1])
    assert "documents.user_id" in sql
    assert "content_sha256" in sql
    assert "deadbeef" in sql
    assert owner.hex in sql.replace("-", "")


async def test_list_recent_is_user_scoped_and_ordered() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = DocumentRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_recent(limit=5, offset=10)

    sql = _sql(session.statements[-1]).lower()
    assert "documents.user_id" in sql
    assert "order by documents.created_at desc" in sql
    assert "limit 5" in sql
    assert "offset 10" in sql


async def test_count_is_user_scoped() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(value=0))
    repo = DocumentRepository(session, owner)  # type: ignore[arg-type]

    total = await repo.count()

    assert total == 0
    sql = _sql(session.statements[-1]).lower()
    assert "count(" in sql
    assert "user_id" in sql


# --- ChunkRepository ----------------------------------------------------- #


def _chunk(user_id: uuid.UUID) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        user_id=user_id,
        ordinal=0,
        text="x",
        content_hash="h",
    )


async def test_add_many_rejects_foreign_user_id() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession()
    repo = ChunkRepository(session, owner)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="does not match"):
        await repo.add_many([_chunk(owner), _chunk(uuid.uuid4())])
    # The batch is rejected before anything is staged.
    assert session.added == []


async def test_add_many_stages_owned_chunks() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession()
    repo = ChunkRepository(session, owner)  # type: ignore[arg-type]

    chunks = [_chunk(owner), _chunk(owner)]
    await repo.add_many(chunks)
    assert session.added == chunks


async def test_list_by_document_scopes_user_and_orders_by_ordinal() -> None:
    owner = uuid.uuid4()
    document_id = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = ChunkRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_by_document(document_id)

    sql = _sql(session.statements[-1]).lower()
    assert "chunks.user_id" in sql
    assert "chunks.document_id" in sql
    assert "order by chunks.ordinal asc" in sql


async def test_delete_by_document_scopes_user() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession()
    repo = ChunkRepository(session, owner)  # type: ignore[arg-type]

    await repo.delete_by_document(uuid.uuid4())

    sql = _sql(session.statements[-1]).lower()
    assert sql.startswith("delete from chunks")
    assert "user_id" in sql
    assert "document_id" in sql


# --- OutboxRepository ---------------------------------------------------- #


async def test_outbox_add_stages_event() -> None:
    session = _CapturingSession()
    repo = OutboxRepository(session)  # type: ignore[arg-type]

    aggregate = uuid.uuid4()
    event = await repo.add(
        event_type="document.uploaded",
        aggregate_id=aggregate,
        payload={"user_id": "u1"},
    )

    assert event.event_type == "document.uploaded"
    assert event.aggregate_id == aggregate
    assert event.payload == {"user_id": "u1"}
    assert session.added == [event]


async def test_fetch_unprocessed_selects_only_unprocessed_oldest_first() -> None:
    session = _CapturingSession(_FakeResult(seq=()))
    repo = OutboxRepository(session)  # type: ignore[arg-type]

    await repo.fetch_unprocessed(limit=50)

    sql = _sql(session.statements[-1]).lower()
    assert "processed_at is null" in sql
    assert "order by outbox.created_at asc" in sql
    assert "limit 50" in sql


async def test_mark_processed_stamps_and_increments() -> None:
    session = _CapturingSession()
    repo = OutboxRepository(session)  # type: ignore[arg-type]

    await repo.mark_processed(uuid.uuid4())

    sql = _sql(session.statements[-1]).lower()
    assert sql.startswith("update outbox set")
    assert "processed_at=now()" in sql.replace(" ", "")
    assert "attempts=" in sql.replace(" ", "")


def test_language_enum_unchanged() -> None:
    # documents.language reuses the same shared Language enum as users.
    assert [lang.value for lang in Language] == ["en", "es", "de", "fr", "it"]
