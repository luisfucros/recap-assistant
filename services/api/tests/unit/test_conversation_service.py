"""Unit tests for :class:`ConversationService` (repositories/session faked).

The repositories and DB session are faked at the boundary; the service logic
under test is real — ownership re-checks, title derivation from the first user
message, the recency bump, and that a turn persists exactly a user message plus an
assistant reply. No database or LLM is involved.
"""

import uuid

import pytest
from api.services.conversation_service import ConversationService

from shared.core.enums import MessageRole
from shared.core.errors import NotFoundError
from shared.models.conversation import Conversation, Message

pytestmark = pytest.mark.unit

OWNER = uuid.uuid4()
CONVERSATION_ID = uuid.uuid4()


class _FakeConversations:
    """Fake ConversationRepository: owner-bound, canned get/list/count."""

    def __init__(self, conversation: Conversation | None = None) -> None:
        self.user_id = OWNER
        self._conversation = conversation
        self.added: list[Conversation] = []
        self.deleted: list[Conversation] = []

    async def add(self, conversation: Conversation) -> Conversation:
        self.added.append(conversation)
        return conversation

    async def delete(self, conversation: Conversation) -> None:
        self.deleted.append(conversation)

    async def get_or_404(self, conversation_id: uuid.UUID) -> Conversation:
        if self._conversation is None:
            raise NotFoundError()
        return self._conversation

    async def list_recent(self, *, limit: int, offset: int) -> list[Conversation]:
        return [self._conversation] if self._conversation else []

    async def count(self) -> int:
        return 1 if self._conversation else 0


class _FakeMessages:
    """Fake MessageRepository capturing appended messages."""

    def __init__(self, existing: list[Message] | None = None) -> None:
        self.user_id = OWNER
        self.added: list[Message] = []
        self._existing = existing or []

    async def add(self, message: Message) -> Message:
        self.added.append(message)
        return message

    async def list_by_conversation(
        self, conversation_id: uuid.UUID, *, limit: int, offset: int
    ) -> list[Message]:
        return self._existing

    async def count_by_conversation(self, conversation_id: uuid.UUID) -> int:
        return len(self._existing)


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _FakeCheckpointer:
    """Fake LangGraph checkpointer, capturing deleted thread ids."""

    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


def _service(checkpointer: _FakeCheckpointer | None = None) -> ConversationService:
    return ConversationService(checkpointer=checkpointer)  # type: ignore[arg-type]


# --- create ------------------------------------------------------------------ #


async def test_create_assigns_owner_and_commits() -> None:
    conversations = _FakeConversations()
    session = _FakeSession()
    conversation = await _service().create(conversations=conversations, session=session)  # type: ignore[arg-type]
    assert conversation.user_id == OWNER
    assert session.commits == 1


# --- list -------------------------------------------------------------------- #


async def test_list_conversations_returns_items_and_total() -> None:
    conv = Conversation(user_id=OWNER, title="Odyssey")
    conversations = _FakeConversations(conv)
    items, total = await _service().list_conversations(conversations=conversations)  # type: ignore[arg-type]
    assert list(items) == [conv]
    assert total == 1


async def test_list_messages_requires_ownership() -> None:
    # An unowned/absent conversation → 404 before any messages are read.
    conversations = _FakeConversations(None)
    messages = _FakeMessages()
    with pytest.raises(NotFoundError):
        await _service().list_messages(
            conversations=conversations,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            conversation_id=CONVERSATION_ID,
        )


# --- record_turn ------------------------------------------------------------- #


async def test_record_turn_persists_user_then_assistant_and_titles_first_turn() -> None:
    conv = Conversation(user_id=OWNER, title=None)
    conversations = _FakeConversations(conv)
    messages = _FakeMessages()
    session = _FakeSession()

    user_msg, assistant_msg = await _service().record_turn(
        conversations=conversations,  # type: ignore[arg-type]
        messages=messages,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
        user_text="Who is the narrator of the Odyssey?",
        assistant_text="Odysseus narrates much of it.",
        tool_calls={"steps": [{"name": "retrieve_chunks"}]},
    )

    # Exactly two messages, user first, both owner-scoped and on this conversation.
    assert [m.role for m in messages.added] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert all(m.user_id == OWNER and m.conversation_id == CONVERSATION_ID for m in messages.added)
    assert user_msg.content == "Who is the narrator of the Odyssey?"
    assert assistant_msg.tool_calls == {"steps": [{"name": "retrieve_chunks"}]}
    # First turn names the conversation from the user message; recency bumped; committed.
    assert conv.title == "Who is the narrator of the Odyssey?"
    assert conv.updated_at is not None
    assert session.commits == 1


async def test_record_turn_stamps_assistant_created_at_strictly_after_user() -> None:
    # Both inserts land in the same DB transaction, where Postgres's `now()` is
    # fixed for the whole transaction — a server-default timestamp would tie,
    # leaving message order to an unrelated (random-UUID) tiebreaker. Stamping
    # both explicitly here, offset by a tick, keeps the transcript ordered
    # user-then-assistant regardless of what the DB's clock does.
    conv = Conversation(user_id=OWNER, title=None)
    messages = _FakeMessages()

    user_msg, assistant_msg = await _service().record_turn(
        conversations=_FakeConversations(conv),  # type: ignore[arg-type]
        messages=messages,  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
        user_text="hi",
        assistant_text="hello",
    )

    assert user_msg.created_at is not None
    assert assistant_msg.created_at is not None
    assert assistant_msg.created_at > user_msg.created_at


async def test_record_turn_keeps_existing_title() -> None:
    conv = Conversation(user_id=OWNER, title="Existing title")
    conversations = _FakeConversations(conv)
    await _service().record_turn(
        conversations=conversations,  # type: ignore[arg-type]
        messages=_FakeMessages(),  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
        user_text="a follow-up question",
        assistant_text="an answer",
    )
    assert conv.title == "Existing title"


async def test_record_turn_titles_are_clipped() -> None:
    conv = Conversation(user_id=OWNER, title=None)
    long_text = "word " * 40  # well over the 60-char cap
    await _service().record_turn(
        conversations=_FakeConversations(conv),  # type: ignore[arg-type]
        messages=_FakeMessages(),  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
        user_text=long_text,
        assistant_text="ok",
    )
    assert len(conv.title) <= 60
    assert conv.title.endswith("…")


async def test_record_turn_on_unowned_conversation_raises() -> None:
    with pytest.raises(NotFoundError):
        await _service().record_turn(
            conversations=_FakeConversations(None),  # type: ignore[arg-type]
            messages=_FakeMessages(),  # type: ignore[arg-type]
            session=_FakeSession(),  # type: ignore[arg-type]
            conversation_id=CONVERSATION_ID,
            user_text="hi",
            assistant_text="hello",
        )


# --- record_user_message / record_assistant_reply (HITL pause/resume) -------- #


async def test_record_user_message_persists_only_the_user_side() -> None:
    conv = Conversation(user_id=OWNER, title=None)
    conversations = _FakeConversations(conv)
    messages = _FakeMessages()
    session = _FakeSession()

    user_msg = await _service().record_user_message(
        conversations=conversations,  # type: ignore[arg-type]
        messages=messages,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
        user_text="Search the web for the sequel's release date.",
    )

    assert [m.role for m in messages.added] == [MessageRole.USER]
    assert user_msg.content == "Search the web for the sequel's release date."
    # Same first-turn conveniences as record_turn: titled and bumped.
    assert conv.title == "Search the web for the sequel's release date."
    assert conv.updated_at is not None
    assert session.commits == 1


async def test_record_user_message_keeps_existing_title() -> None:
    conv = Conversation(user_id=OWNER, title="Existing title")
    await _service().record_user_message(
        conversations=_FakeConversations(conv),  # type: ignore[arg-type]
        messages=_FakeMessages(),  # type: ignore[arg-type]
        session=_FakeSession(),  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
        user_text="a follow-up question",
    )
    assert conv.title == "Existing title"


async def test_record_user_message_on_unowned_conversation_raises() -> None:
    with pytest.raises(NotFoundError):
        await _service().record_user_message(
            conversations=_FakeConversations(None),  # type: ignore[arg-type]
            messages=_FakeMessages(),  # type: ignore[arg-type]
            session=_FakeSession(),  # type: ignore[arg-type]
            conversation_id=CONVERSATION_ID,
            user_text="hi",
        )


async def test_record_assistant_reply_persists_only_the_assistant_side() -> None:
    conv = Conversation(user_id=OWNER, title="Existing title")
    conversations = _FakeConversations(conv)
    messages = _FakeMessages()
    session = _FakeSession()

    assistant_msg = await _service().record_assistant_reply(
        conversations=conversations,  # type: ignore[arg-type]
        messages=messages,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
        assistant_text="Here's what I found.",
        tool_calls={"steps": [{"name": "web_search"}]},
    )

    assert [m.role for m in messages.added] == [MessageRole.ASSISTANT]
    assert assistant_msg.content == "Here's what I found."
    assert assistant_msg.tool_calls == {"steps": [{"name": "web_search"}]}
    assert conv.updated_at is not None
    assert session.commits == 1


async def test_record_assistant_reply_on_unowned_conversation_raises() -> None:
    with pytest.raises(NotFoundError):
        await _service().record_assistant_reply(
            conversations=_FakeConversations(None),  # type: ignore[arg-type]
            messages=_FakeMessages(),  # type: ignore[arg-type]
            session=_FakeSession(),  # type: ignore[arg-type]
            conversation_id=CONVERSATION_ID,
            assistant_text="hello",
        )


# --- delete ------------------------------------------------------------------- #


async def test_delete_removes_the_conversation_and_its_checkpoint_thread() -> None:
    conv = Conversation(user_id=OWNER, title="Odyssey")
    conversations = _FakeConversations(conv)
    session = _FakeSession()
    checkpointer = _FakeCheckpointer()

    await _service(checkpointer).delete(
        conversations=conversations,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
    )

    assert conversations.deleted == [conv]
    assert checkpointer.deleted_threads == [str(CONVERSATION_ID)]
    assert session.commits == 1


async def test_delete_skips_checkpoint_cleanup_when_no_checkpointer_is_wired() -> None:
    conv = Conversation(user_id=OWNER, title="Odyssey")
    conversations = _FakeConversations(conv)
    session = _FakeSession()

    await _service(None).delete(
        conversations=conversations,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        conversation_id=CONVERSATION_ID,
    )

    assert conversations.deleted == [conv]
    assert session.commits == 1


async def test_delete_on_unowned_conversation_raises_before_any_cleanup() -> None:
    conversations = _FakeConversations(None)
    checkpointer = _FakeCheckpointer()
    with pytest.raises(NotFoundError):
        await _service(checkpointer).delete(
            conversations=conversations,  # type: ignore[arg-type]
            session=_FakeSession(),  # type: ignore[arg-type]
            conversation_id=CONVERSATION_ID,
        )
    assert conversations.deleted == []
    assert checkpointer.deleted_threads == []
