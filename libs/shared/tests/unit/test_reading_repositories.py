"""Unit tests for the reading-progress and reading-event repositories.

No database: a capturing fake session records the statements the repositories
build, which are compiled to SQL and inspected. This asserts the load-bearing
per-user filter is present on every query, and that append (event insert) rejects
a foreign ``user_id`` — the boundary (``session.execute``) is mocked, the
query-building logic under test is real.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from shared.core.enums import ReadingEventType, ReadingStatus
from shared.models.reading import ReadingEvent
from shared.repositories import ReadingEventRepository, ReadingProgressRepository

pytestmark = pytest.mark.unit


class _FakeResult:
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
    def __init__(self, result: _FakeResult | None = None) -> None:
        self.statements: list[Any] = []
        self.added: list[Any] = []
        self._result = result or _FakeResult()

    async def execute(self, statement: Any) -> _FakeResult:
        self.statements.append(statement)
        return self._result

    def add(self, entity: Any) -> None:
        self.added.append(entity)

    async def flush(self) -> None:
        return None


def _sql(statement: Any) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True}))


# --- enums --------------------------------------------------------------- #


def test_reading_status_values() -> None:
    assert [s.value for s in ReadingStatus] == [
        "not_started",
        "reading",
        "completed",
        "cancelled",
    ]


def test_reading_event_type_values() -> None:
    assert [t.value for t in ReadingEventType] == [
        "position_advanced",
        "status_changed",
        "session",
        "completed",
    ]


# --- ReadingProgressRepository ------------------------------------------- #


async def test_get_by_document_scopes_user_and_document() -> None:
    owner = uuid.uuid4()
    document_id = uuid.uuid4()
    session = _CapturingSession(_FakeResult(value=None))
    repo = ReadingProgressRepository(session, owner)  # type: ignore[arg-type]

    await repo.get_by_document(document_id)

    sql = _sql(session.statements[-1]).lower()
    assert "reading_progress.user_id" in sql
    assert "reading_progress.document_id" in sql
    assert owner.hex in sql.replace("-", "")
    assert document_id.hex in sql.replace("-", "")


async def test_list_by_status_filters_status_and_orders_by_last_accessed() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = ReadingProgressRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_by_status(ReadingStatus.READING, limit=5, offset=10)

    sql = _sql(session.statements[-1]).lower()
    assert "reading_progress.user_id" in sql
    assert "reading_progress.status" in sql
    assert "reading" in sql
    assert "order by reading_progress.last_accessed_at desc" in sql
    assert "limit 5" in sql
    assert "offset 10" in sql


async def test_list_recent_is_user_scoped_and_ordered() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = ReadingProgressRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_recent(limit=3)

    sql = _sql(session.statements[-1]).lower()
    assert "reading_progress.user_id" in sql
    assert "order by reading_progress.last_accessed_at desc" in sql
    assert "limit 3" in sql


async def test_count_by_status_is_user_and_status_scoped() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(value=7))
    repo = ReadingProgressRepository(session, owner)  # type: ignore[arg-type]

    total = await repo.count_by_status(ReadingStatus.COMPLETED)

    assert total == 7
    sql = _sql(session.statements[-1]).lower()
    assert "count(" in sql
    assert "reading_progress.user_id" in sql
    assert "reading_progress.status" in sql
    assert "completed" in sql


# --- ReadingEventRepository ---------------------------------------------- #


def _event(user_id: uuid.UUID) -> ReadingEvent:
    return ReadingEvent(
        id=uuid.uuid4(),
        user_id=user_id,
        document_id=uuid.uuid4(),
        type=ReadingEventType.POSITION_ADVANCED,
        from_page=0,
        to_page=5,
    )


async def test_event_add_rejects_foreign_user_id() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession()
    repo = ReadingEventRepository(session, owner)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="does not match"):
        await repo.add(_event(uuid.uuid4()))
    assert session.added == []


async def test_event_add_stages_owned_event() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession()
    repo = ReadingEventRepository(session, owner)  # type: ignore[arg-type]

    event = _event(owner)
    await repo.add(event)
    assert session.added == [event]


async def test_list_since_scopes_user_and_bounds_time() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = ReadingEventRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_since(datetime(2026, 1, 1, tzinfo=UTC), limit=100)

    sql = _sql(session.statements[-1]).lower()
    assert "reading_events.user_id" in sql
    assert "reading_events.occurred_at >=" in sql
    assert "order by reading_events.occurred_at asc" in sql
    assert "limit 100" in sql
