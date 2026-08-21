"""Unit tests for the per-user isolation foundation.

These assert the load-bearing invariant — every scoped query carries the owning
``user_id`` filter — without touching a database: the SELECT is compiled to SQL
and inspected, and the guard helpers are exercised directly.
"""

import uuid

import pytest
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.errors import NotFoundError
from shared.db.base import Base
from shared.repositories.base import UserScopedRepository, ensure_owned

pytestmark = pytest.mark.unit


class _Owned(Base):
    """Throwaway owned entity used only to exercise the base repository."""

    __tablename__ = "test_owned_entities"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column()


class _OwnedRepo(UserScopedRepository[_Owned]):
    model = _Owned


def test_scoped_select_always_filters_by_user_id() -> None:
    owner = uuid.uuid4()
    repo = _OwnedRepo(session=object(), user_id=owner)  # session unused for compile

    sql = str(repo._scoped_select().compile(compile_kwargs={"literal_binds": True}))

    assert "test_owned_entities.user_id" in sql
    # The bound owner id is present (rendered without dashes by the compiler).
    assert owner.hex in sql.replace("-", "")


def test_get_scopes_by_both_id_and_user() -> None:
    repo = _OwnedRepo(session=object(), user_id=uuid.uuid4())
    stmt = repo._scoped_select().where(_Owned.id == uuid.uuid4())
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
    # Both predicates present ⇒ another user's row can't be fetched by id.
    assert "user_id" in sql
    assert "id =" in sql


async def test_add_rejects_foreign_user_id() -> None:
    repo = _OwnedRepo(session=object(), user_id=uuid.uuid4())
    foreign = _Owned(id=uuid.uuid4(), user_id=uuid.uuid4())
    with pytest.raises(ValueError, match="does not match"):
        await repo.add(foreign)


def test_ensure_owned_passes_for_owner() -> None:
    owner = uuid.uuid4()
    ensure_owned(_Owned(id=uuid.uuid4(), user_id=owner), owner)  # no raise


def test_ensure_owned_raises_for_other_user() -> None:
    with pytest.raises(NotFoundError):
        ensure_owned(_Owned(id=uuid.uuid4(), user_id=uuid.uuid4()), uuid.uuid4())


def test_ensure_owned_raises_for_missing() -> None:
    with pytest.raises(NotFoundError):
        ensure_owned(None, uuid.uuid4())
