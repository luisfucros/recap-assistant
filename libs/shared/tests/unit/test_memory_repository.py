"""Unit tests for :class:`LongTermMemoryRepository`.

No database: a capturing fake session records the statements the repository
builds, compiled to SQL and inspected. The load-bearing assertion is that the
per-user ``user_id`` filter is present on every query, including the
document/page-range lookups a caller might otherwise assume are already
scoped by the document id alone.
"""

import uuid
from typing import Any

import pytest

from shared.core.enums import MemoryType
from shared.repositories import LongTermMemoryRepository

pytestmark = pytest.mark.unit


class _FakeResult:
    def __init__(self, *, value: Any = None, seq: tuple[Any, ...] = ()) -> None:
        self._value = value
        self._seq = seq

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
    def __init__(self, result: _FakeResult | None = None) -> None:
        self.statements: list[Any] = []
        self._result = result or _FakeResult()

    async def execute(self, statement: Any) -> _FakeResult:
        self.statements.append(statement)
        return self._result

    async def delete(self, entity: Any) -> None:
        self.statements.append(("delete", entity))

    async def flush(self) -> None:
        pass


def _sql(statement: Any) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True})).lower()


async def test_list_recent_scopes_user_and_orders_by_recency() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = LongTermMemoryRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_recent(limit=5, offset=0)

    sql = _sql(session.statements[-1])
    assert "long_term_memory.user_id" in sql
    assert owner.hex in sql.replace("-", "")
    assert "order by long_term_memory.created_at desc" in sql


async def test_list_by_type_scopes_user_and_orders_by_recency() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = LongTermMemoryRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_by_type(MemoryType.PREFERENCE, limit=5, offset=0)

    sql = _sql(session.statements[-1])
    assert "long_term_memory.user_id" in sql
    assert owner.hex in sql.replace("-", "")
    assert "'preference'" in sql
    assert "order by long_term_memory.created_at desc" in sql


async def test_list_by_document_scopes_user_and_document() -> None:
    owner = uuid.uuid4()
    document_id = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = LongTermMemoryRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_by_document(document_id)

    sql = _sql(session.statements[-1])
    assert "long_term_memory.user_id" in sql
    assert "long_term_memory.document_id" in sql
    assert owner.hex in sql.replace("-", "")
    assert document_id.hex in sql.replace("-", "")


async def test_list_summaries_covering_filters_type_and_orders_by_page_start() -> None:
    owner = uuid.uuid4()
    document_id = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = LongTermMemoryRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_summaries_covering(document_id, max_page_end=50)

    sql = _sql(session.statements[-1])
    assert "long_term_memory.user_id" in sql
    assert "'summary'" in sql
    assert "page_end <= 50" in sql
    assert "order by long_term_memory.page_start asc" in sql


async def test_list_summaries_covering_omits_bound_when_not_requested() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = LongTermMemoryRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_summaries_covering(uuid.uuid4())

    sql = _sql(session.statements[-1])
    assert "page_end <=" not in sql


async def test_count_is_user_scoped() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(value=0))
    repo = LongTermMemoryRepository(session, owner)  # type: ignore[arg-type]

    await repo.count()

    sql = _sql(session.statements[-1])
    assert "count(" in sql
    assert "long_term_memory.user_id" in sql
    assert owner.hex in sql.replace("-", "")


async def test_delete_removes_and_flushes() -> None:
    session = _CapturingSession()
    repo = LongTermMemoryRepository(session, uuid.uuid4())  # type: ignore[arg-type]
    memory = object()

    await repo.delete(memory)  # type: ignore[arg-type]

    assert session.statements == [("delete", memory)]
