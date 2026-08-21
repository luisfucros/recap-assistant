"""Chat models: ``Conversation`` and ``Message`` — the product-facing transcript.

These persist *what the reader and assistant said*, so the UI can list past
chats and reopen one:

* :class:`Conversation` is a chat thread owned by one user. Its ``id`` doubles as
  the LangGraph checkpointer's ``thread_id``, so the durable agent state for a
  thread and its human-readable transcript share one key.
* :class:`Message` is one turn's user input or assistant reply, in order.

This table is deliberately **not** the agent's internal message graph. The
LangGraph checkpointer owns the full run state — the system prompt, the
tool-call/observation loop, intermediate ``AIMessage``/``ToolMessage`` chatter —
which is what a follow-up turn resumes from. What lands here is the curated,
user-visible transcript: the human message and the final answer, with the turn's
tool steps and citations kept as structured metadata on the assistant message
(for rendering the tool-step timeline) rather than as separate rows. Keeping the
two separate stops the transcript from bloating with internal chatter and lets
each evolve independently.

Both carry ``user_id`` for per-user isolation and cascade-delete with the user;
messages also cascade with their conversation.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.enums import MessageRole
from shared.db.base import Base
from shared.models.types import MESSAGE_ROLE_TYPE


class Conversation(Base):
    """A chat thread between one user and the assistant.

    The primary key is reused as the LangGraph checkpointer ``thread_id``, so the
    durable agent state and this transcript are addressed by the same id.
    """

    __tablename__ = "conversations"
    __table_args__ = (
        # The conversation list is "my threads, most-recently-active first"; this
        # composite index is that query's access path.
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # A short, human-readable label (typically derived from the first message);
    # null until one is set, so the UI can fall back to a placeholder.
    title: Mapped[str | None] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Bumped whenever a turn is appended, so the conversation list can order by
    # recency of activity rather than creation.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    """One turn in a conversation: a user input or an assistant reply.

    ``tool_calls`` holds the turn's tool-step timeline and citations as structured
    JSON for assistant messages (null for user messages) — enough to re-render the
    tool steps and sources in history without replaying the agent.
    """

    __tablename__ = "messages"
    __table_args__ = (
        # Messages are always read as "this conversation's messages, in order";
        # this composite index serves that fetch directly.
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    # Denormalized owner: carried on every message so the isolation invariant
    # (every relational query filters by user_id) holds without a join back to
    # the conversation — the same pattern as chunks carrying their user_id.
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    role: Mapped[MessageRole] = mapped_column(MESSAGE_ROLE_TYPE)
    content: Mapped[str] = mapped_column(Text)
    # Structured tool-step/citation metadata for assistant turns; null otherwise.
    tool_calls: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
