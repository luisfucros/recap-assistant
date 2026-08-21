"""Unit tests for the chat model schema (``Conversation`` / ``Message``).

No database: these assert the load-bearing schema contract straight off
``Base.metadata`` — that both tables carry a cascading ``user_id`` (the per-user
isolation invariant), that messages order within a conversation, and that the
``message_role`` vocabulary is what the app writes. Behavioral coverage (scoped
queries, cascade deletes) lands with the repositories and the integration suite.
"""

import pytest

from shared.core.enums import MessageRole
from shared.models import Conversation, Message

pytestmark = pytest.mark.unit


def test_message_role_vocabulary() -> None:
    assert [r.value for r in MessageRole] == ["user", "assistant", "system", "tool"]


def test_both_tables_carry_a_cascading_user_id() -> None:
    # Every user-owned table filters by user_id and cascades with the user; a
    # missing/ non-cascading owner column would break isolation and leave orphans.
    for model in (Conversation, Message):
        user_fk = next(fk for fk in model.__table__.c["user_id"].foreign_keys)
        assert user_fk.column.table.name == "users"
        assert user_fk.ondelete == "CASCADE"


def test_message_cascades_with_its_conversation() -> None:
    conv_fk = next(fk for fk in Message.__table__.c["conversation_id"].foreign_keys)
    assert conv_fk.column.table.name == "conversations"
    assert conv_fk.ondelete == "CASCADE"


def test_conversation_lists_by_recent_activity() -> None:
    # The conversation list orders by (user_id, updated_at); that composite index
    # is its access path.
    indexes = {tuple(c.name for c in ix.columns) for ix in Conversation.__table__.indexes}
    assert ("user_id", "updated_at") in indexes


def test_messages_order_within_a_conversation() -> None:
    indexes = {tuple(c.name for c in ix.columns) for ix in Message.__table__.indexes}
    assert ("conversation_id", "created_at") in indexes
