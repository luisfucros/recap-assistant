"""Unit tests for the conversation and message repositories.

No database: a capturing fake session records the statements the repositories
build, compiled to SQL and inspected. The load-bearing assertion is that the
per-user ``user_id`` filter is present on every query — including message reads
scoped by ``conversation_id``, which must *also* be owner-filtered so another
user's conversation id can never surface messages. The boundary
(``session.execute``) is mocked; the query-building logic is real.
"""

import uuid
from typing import Any

import pytest

from shared.repositories import ConversationRepository, MessageRepository

pytestmark = pytest.mark.unit


class _FakeResult:
    def __init__(self, *, value: Any = None, seq: tuple[Any, ...] = ()) -> None:
        self._value = value
        self._seq = seq

    def scalar_one(self) -> Any:
        return self._value

    def scalar_one_or_none(self) -> Any:
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


def _sql(statement: Any) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True})).lower()


# --- ConversationRepository -------------------------------------------------- #


async def test_list_recent_scopes_user_and_orders_by_updated() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = ConversationRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_recent(limit=5, offset=10)

    sql = _sql(session.statements[-1])
    assert "conversations.user_id" in sql
    assert owner.hex in sql.replace("-", "")
    assert "order by conversations.updated_at desc" in sql


async def test_conversation_count_is_user_scoped() -> None:
    owner = uuid.uuid4()
    session = _CapturingSession(_FakeResult(value=0))
    repo = ConversationRepository(session, owner)  # type: ignore[arg-type]

    await repo.count()

    sql = _sql(session.statements[-1])
    assert "count(" in sql
    assert "conversations.user_id" in sql
    assert owner.hex in sql.replace("-", "")


# --- MessageRepository ------------------------------------------------------- #


async def test_list_by_conversation_scopes_user_and_conversation_in_order() -> None:
    owner = uuid.uuid4()
    conversation_id = uuid.uuid4()
    session = _CapturingSession(_FakeResult(seq=()))
    repo = MessageRepository(session, owner)  # type: ignore[arg-type]

    await repo.list_by_conversation(conversation_id, limit=20, offset=0)

    sql = _sql(session.statements[-1])
    # Owner filter AND conversation filter both present — neither alone is safe.
    assert "messages.user_id" in sql
    assert "messages.conversation_id" in sql
    assert owner.hex in sql.replace("-", "")
    assert conversation_id.hex in sql.replace("-", "")
    assert "order by messages.created_at asc" in sql


async def test_count_by_conversation_is_user_scoped() -> None:
    owner = uuid.uuid4()
    conversation_id = uuid.uuid4()
    session = _CapturingSession(_FakeResult(value=0))
    repo = MessageRepository(session, owner)  # type: ignore[arg-type]

    await repo.count_by_conversation(conversation_id)

    sql = _sql(session.statements[-1])
    assert "count(" in sql
    assert "messages.user_id" in sql
    assert "messages.conversation_id" in sql
    assert owner.hex in sql.replace("-", "")
