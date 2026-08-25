"""The agent's turn-scoped working memory (FR-7.8), backed by Redis.

For a multi-step turn the agent keeps a **scratchpad** — its plan, running
findings, and open questions — *outside* the model context window, so a long
research turn doesn't re-send everything each step and bloat the prompt (or
trigger premature compaction). The ``plan`` node writes the plan, tool steps
append findings, and ``generate`` recalls only the **relevant** slices back into
context.

The scratchpad is ephemeral and strictly turn/conversation-scoped: entries live
under a ``(user_id, conversation_id, turn_id)`` key and expire after
``SCRATCHPAD_TTL_SECONDS`` (distinct from short-term conversation state and
long-term memory). The ``user_id`` is part of the key, so one user's notes can
never surface in another's turn — the isolation invariant, on the cache.

Recall is **relevance-gated**: the turn's plan is always returned (it frames the
whole turn), and findings/questions are returned only when they share salient
words with the query, most-relevant-and-recent first. This is a deliberately
simple lexical gate — no embeddings on the hot path — sufficient to keep only
pertinent slices in context; salient conclusions can later be promoted to
long-term memory (M5).
"""

import json
import uuid
from dataclasses import dataclass

from loguru import logger
from redis.asyncio import Redis

from shared.core.enums import ScratchpadKind

# Words too common to signal relevance; ignored when scoring lexical overlap.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "do",
        "does",
        "did",
        "i",
        "you",
        "they",
        "them",
        "this",
        "these",
        "those",
        "with",
    ]
)


@dataclass(slots=True)
class ScratchpadNote:
    """One entry in a turn's scratchpad: a plan, a finding, or an open question."""

    kind: ScratchpadKind
    text: str

    def to_json(self) -> str:
        """Serialize for storage in the Redis list."""
        return json.dumps({"kind": self.kind.value, "text": self.text})

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ScratchpadNote":
        """Rebuild a note from its stored JSON."""
        data = json.loads(raw)
        return cls(kind=ScratchpadKind(data["kind"]), text=data["text"])


def _tokens(text: str) -> set[str]:
    """Lowercased, stopword-stripped word set used for lexical relevance scoring."""
    normalized = "".join(c.lower() if c.isalnum() else " " for c in text)
    return set(normalized.split()) - _STOPWORDS


class ScratchpadService:
    """Read/write the agent's turn-scoped working memory in Redis."""

    def __init__(self, *, redis: Redis, ttl_seconds: int) -> None:
        """Wire the service to Redis and the per-turn TTL (seconds)."""
        self._redis = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(user_id: uuid.UUID, conversation_id: str, turn_id: str) -> str:
        """The Redis key for one turn's scratchpad (user id first — isolation)."""
        return f"scratchpad:{user_id}:{conversation_id}:{turn_id}"

    async def append(
        self,
        *,
        user_id: uuid.UUID,
        conversation_id: str,
        turn_id: str,
        note: ScratchpadNote,
    ) -> None:
        """Append one note to the turn's scratchpad and (re)set its TTL.

        Every write refreshes the expiry, so an active turn keeps its notes while
        an abandoned one lets them lapse — the scratchpad is never long-lived.
        """
        key = self._key(user_id, conversation_id, turn_id)
        await self._redis.rpush(key, note.to_json())
        await self._redis.expire(key, self._ttl)
        logger.debug("scratchpad.append: {}", note.kind.value)

    async def recall(
        self,
        *,
        user_id: uuid.UUID,
        conversation_id: str,
        turn_id: str,
        query: str,
        limit: int = 5,
    ) -> list[ScratchpadNote]:
        """Return the turn's plan plus the findings/questions relevant to ``query``.

        The plan (if any) always comes first — it frames the turn. The remaining
        notes are lexically scored against the query and only those that overlap
        are returned, most-relevant-then-most-recent first, capped at ``limit``.
        An empty query or empty scratchpad yields just the plan (or nothing).
        """
        key = self._key(user_id, conversation_id, turn_id)
        raw_entries = await self._redis.lrange(key, 0, -1)
        notes = [ScratchpadNote.from_json(raw) for raw in raw_entries]

        plans = [n for n in notes if n.kind is ScratchpadKind.PLAN]
        others = [n for n in notes if n.kind is not ScratchpadKind.PLAN]

        query_tokens = _tokens(query)
        # (original index → recency) preserved so ties break toward newer notes.
        scored = [
            (len(query_tokens & _tokens(note.text)), index, note)
            for index, note in enumerate(others)
        ]
        relevant = [entry for entry in scored if entry[0] > 0]
        relevant.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        recalled = plans + [note for _, _, note in relevant[:limit]]
        logger.debug("scratchpad.recall: {} notes", len(recalled))
        return recalled
