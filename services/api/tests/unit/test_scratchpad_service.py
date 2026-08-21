"""Unit tests for the agent scratchpad (Redis faked in memory).

The load-bearing behaviors: append persists notes and refreshes the TTL; recall
is relevance-gated (the plan always returns, findings/questions only when they
overlap the query); and the key is user-scoped so one reader's notes never surface
in another's turn. Redis is a small in-memory fake; the service logic is real.
"""

import uuid
from typing import Any

import pytest
from api.services.scratchpad_service import ScratchpadNote, ScratchpadService

from shared.core.enums import ScratchpadKind

pytestmark = pytest.mark.unit

USER_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
USER_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
CONV = uuid.UUID("33333333-3333-3333-3333-333333333333")
TURN = "turn-1"


class _FakeRedis:
    """Minimal in-memory stand-in for the Redis list ops the service uses."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.expires: dict[str, int] = {}

    async def rpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).append(value)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    async def expire(self, key: str, ttl: int) -> None:
        self.expires[key] = ttl


def _service(redis: Any, ttl: int = 1800) -> ScratchpadService:
    return ScratchpadService(redis=redis, ttl_seconds=ttl)


async def _seed(service: ScratchpadService, user_id: uuid.UUID = USER_A) -> None:
    for note in (
        ScratchpadNote(ScratchpadKind.PLAN, "Recap the Odyssey for the reader"),
        ScratchpadNote(ScratchpadKind.FINDING, "Odysseus sails home from Troy"),
        ScratchpadNote(ScratchpadKind.FINDING, "A treatise on Roman tax policy"),
    ):
        await service.append(user_id=user_id, conversation_id=CONV, turn_id=TURN, note=note)


# --- note serialization ------------------------------------------------------ #


def test_note_json_round_trip() -> None:
    note = ScratchpadNote(ScratchpadKind.QUESTION, "Which pages were read?")
    assert ScratchpadNote.from_json(note.to_json()) == note


# --- append ------------------------------------------------------------------ #


async def test_append_persists_and_sets_ttl() -> None:
    redis = _FakeRedis()
    await _service(redis, ttl=1234).append(
        user_id=USER_A,
        conversation_id=CONV,
        turn_id=TURN,
        note=ScratchpadNote(ScratchpadKind.FINDING, "x"),
    )
    key = f"scratchpad:{USER_A}:{CONV}:{TURN}"
    assert redis.lists[key]  # one entry stored
    assert redis.expires[key] == 1234  # TTL refreshed on write


# --- recall ------------------------------------------------------------------ #


async def test_recall_returns_plan_plus_only_relevant_findings() -> None:
    service = _service(_FakeRedis())
    await _seed(service)
    recalled = await service.recall(
        user_id=USER_A, conversation_id=CONV, turn_id=TURN, query="who is Odysseus"
    )
    texts = [n.text for n in recalled]
    # The plan always comes first; the Odysseus finding is relevant, tax is not.
    assert recalled[0].kind is ScratchpadKind.PLAN
    assert any("Odysseus sails home" in t for t in texts)
    assert not any("tax" in t for t in texts)


async def test_recall_empty_query_returns_only_the_plan() -> None:
    service = _service(_FakeRedis())
    await _seed(service)
    recalled = await service.recall(user_id=USER_A, conversation_id=CONV, turn_id=TURN, query="")
    assert [n.kind for n in recalled] == [ScratchpadKind.PLAN]


async def test_recall_honors_the_limit_on_relevant_notes() -> None:
    redis = _FakeRedis()
    service = _service(redis)
    for i in range(6):
        await service.append(
            user_id=USER_A,
            conversation_id=CONV,
            turn_id=TURN,
            note=ScratchpadNote(ScratchpadKind.FINDING, f"Odysseus detail number {i}"),
        )
    recalled = await service.recall(
        user_id=USER_A, conversation_id=CONV, turn_id=TURN, query="Odysseus", limit=3
    )
    assert len(recalled) == 3  # no plan seeded, so all three are findings


async def test_recall_is_user_scoped() -> None:
    service = _service(_FakeRedis())
    await _seed(service, user_id=USER_A)
    # User B shares the conversation/turn ids but has a distinct key → sees nothing.
    recalled = await service.recall(
        user_id=USER_B, conversation_id=CONV, turn_id=TURN, query="Odysseus"
    )
    assert recalled == []
