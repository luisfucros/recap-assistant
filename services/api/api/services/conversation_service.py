"""Chat-history business logic: create threads, list them, persist turns.

:class:`ConversationService` is the single writer of the product-facing chat
transcript (:class:`~shared.models.conversation.Conversation` /
:class:`~shared.models.conversation.Message`). The ``/chat`` routes delegate the
persistence side of a turn here while the agent itself runs in
:class:`~api.services.agent_service.AgentService`; keeping the two apart means the
transcript store and the LangGraph checkpointer (which owns the internal run
state, keyed by the same conversation id) evolve independently.

Ownership is enforced end-to-end: conversations and messages are read and written
through user-scoped repositories, and every append re-loads the target
conversation through the owner's repository (404 if it isn't theirs), so a caller
can only ever touch their own history — the ``user_id`` never comes from client
input.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.enums import MessageRole
from shared.models.conversation import Conversation, Message
from shared.repositories import ConversationRepository, MessageRepository

# A new conversation's title is derived from its first user message, clipped to
# this many characters so the sidebar label stays short.
_TITLE_MAX_CHARS = 60


def _now() -> datetime:
    """Current UTC time (isolated so tests can patch it cheaply)."""
    return datetime.now(UTC)


def _derive_title(text: str) -> str:
    """Build a short conversation title from the first user message."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _TITLE_MAX_CHARS:
        return collapsed
    return collapsed[: _TITLE_MAX_CHARS - 1].rstrip() + "…"


class ConversationService:
    """Create/list/delete chat threads and persist the turns exchanged in them."""

    def __init__(self, *, checkpointer: BaseCheckpointSaver | None = None) -> None:
        """Wire the service to the agent's checkpointer, needed only for deletion.

        Args:
            checkpointer: The LangGraph checkpointer whose thread state must be
                dropped alongside a deleted conversation. ``None`` in contexts
                that never delete (e.g. most tests), in which case a delete call
                skips checkpoint cleanup rather than failing.
        """
        self._checkpointer = checkpointer

    async def create(
        self,
        *,
        conversations: ConversationRepository,
        session: AsyncSession,
        title: str | None = None,
    ) -> Conversation:
        """Start a new conversation for the owning user and commit it."""
        conversation = await conversations.add(
            Conversation(user_id=conversations.user_id, title=title)
        )
        await session.commit()
        logger.info("conversation.create: created {}", conversation.id)
        return conversation

    async def list_conversations(
        self, *, conversations: ConversationRepository, limit: int = 10, offset: int = 0
    ) -> tuple[Sequence[Conversation], int]:
        """Return a page of the user's conversations (recent first) and the total."""
        items = await conversations.list_recent(limit=limit, offset=offset)
        total = await conversations.count()
        logger.debug("conversation.list: {} of {}", len(items), total)
        return items, total

    async def list_messages(
        self,
        *,
        conversations: ConversationRepository,
        messages: MessageRepository,
        conversation_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Sequence[Message], int]:
        """Return a conversation's messages (chronological) and the total.

        Raises:
            NotFoundError: The conversation doesn't exist or isn't the caller's.
        """
        await conversations.get_or_404(conversation_id)
        items = await messages.list_by_conversation(conversation_id, limit=limit, offset=offset)
        total = await messages.count_by_conversation(conversation_id)
        logger.debug("conversation.list_messages: {} of {}", len(items), total)
        return items, total

    async def delete(
        self,
        *,
        conversations: ConversationRepository,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> None:
        """Delete a conversation, its messages, and its agent checkpoint state.

        The checkpointer's thread state is dropped first (idempotent — safe to
        retry), then the Postgres row as the authority; messages cascade via the
        DB FK. Mirrors how a document's vectors/storage are cleaned up before its
        row.

        Raises:
            NotFoundError: The conversation doesn't exist or isn't the caller's.
        """
        conversation = await conversations.get_or_404(conversation_id)
        if self._checkpointer is not None:
            await self._checkpointer.adelete_thread(str(conversation_id))
        await conversations.delete(conversation)
        await session.commit()
        logger.info("conversation.delete: deleted {}", conversation_id)

    async def record_turn(
        self,
        *,
        conversations: ConversationRepository,
        messages: MessageRepository,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        user_text: str,
        assistant_text: str,
        tool_calls: dict[str, Any] | None = None,
    ) -> tuple[Message, Message]:
        """Append a turn (user message + assistant reply) and commit.

        Persists the user's (already-normalized/redacted) input and the assistant's
        answer — a guardrail refusal is just an assistant message with no
        ``tool_calls``. Names the conversation from its first user message and bumps
        ``updated_at`` so it rises to the top of the recent list. This is the
        transcript write only; the agent run and its checkpoint happen elsewhere.

        Raises:
            NotFoundError: The conversation doesn't exist or isn't the caller's.
        """
        conversation = await conversations.get_or_404(conversation_id)
        owner = conversations.user_id
        # Stamped explicitly (not the column's `server_default=func.now()`):
        # both inserts land in the same transaction, and Postgres's `now()` is
        # fixed for the whole transaction — a server-default timestamp would
        # tie, leaving `list_by_conversation`'s ordering to its (random-UUID)
        # tiebreaker and the transcript to show the reply before the question.
        turn_time = _now()
        user_message = await messages.add(
            Message(
                user_id=owner,
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_text,
                created_at=turn_time,
            )
        )
        assistant_message = await messages.add(
            Message(
                user_id=owner,
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=assistant_text,
                tool_calls=tool_calls,
                created_at=turn_time + timedelta(microseconds=1),
            )
        )
        if conversation.title is None:
            conversation.title = _derive_title(user_text)
        conversation.updated_at = _now()
        await session.commit()
        logger.info("conversation.record_turn: persisted {}", conversation_id)
        return user_message, assistant_message

    async def record_user_message(
        self,
        *,
        conversations: ConversationRepository,
        messages: MessageRepository,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        user_text: str,
    ) -> Message:
        """Append just the user's message and commit — no assistant reply yet.

        Used when a turn pauses on a gated tool call (HITL): the user's question
        is real and worth showing immediately, but there is no answer to pair it
        with until :meth:`record_assistant_reply` completes the turn after
        resume. Titles a still-untitled conversation and bumps recency, same as
        :meth:`record_turn`.

        Raises:
            NotFoundError: The conversation doesn't exist or isn't the caller's.
        """
        conversation = await conversations.get_or_404(conversation_id)
        user_message = await messages.add(
            Message(
                user_id=conversations.user_id,
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_text,
                created_at=_now(),
            )
        )
        if conversation.title is None:
            conversation.title = _derive_title(user_text)
        conversation.updated_at = _now()
        await session.commit()
        logger.info("conversation.record_user_message: persisted {}", conversation_id)
        return user_message

    async def record_assistant_reply(
        self,
        *,
        conversations: ConversationRepository,
        messages: MessageRepository,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        assistant_text: str,
        tool_calls: dict[str, Any] | None = None,
    ) -> Message:
        """Append just the assistant's reply and commit — completes a HITL-paused turn.

        Pairs with an earlier :meth:`record_user_message` call for the same
        turn (the resume path never re-sends the user's original text). Bumps
        recency so the conversation rises to the top once answered.

        Raises:
            NotFoundError: The conversation doesn't exist or isn't the caller's.
        """
        conversation = await conversations.get_or_404(conversation_id)
        assistant_message = await messages.add(
            Message(
                user_id=conversations.user_id,
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=assistant_text,
                tool_calls=tool_calls,
                created_at=_now(),
            )
        )
        conversation.updated_at = _now()
        await session.commit()
        logger.info("conversation.record_assistant_reply: persisted {}", conversation_id)
        return assistant_message
