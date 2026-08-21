"""Functional tests for the chat routes (agent + transcript faked at the boundary).

Full HTTP cycles against the running app: auth is real (register + login), while
the agent run and the transcript store are replaced with fakes via dependency
overrides — so these assert the *route* contract (status codes, response shapes,
SSE frame ordering, guardrail blocks, ownership 404, auth) without a real LLM,
checkpointer, or database. The one boundary left real is HTTP itself.
"""

import base64
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from api.agent.events import (
    AgentEvent,
    BlockedEvent,
    DoneEvent,
    InterruptEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from api.deps import (
    get_agent_service,
    get_conversation_repository,
    get_conversation_service,
    get_rate_limit_service,
    get_tool_context,
)
from api.services.agent_service import AgentTurn, NoPendingInterruptError, ToolStep
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

from shared.core.enums import MessageRole
from shared.core.errors import NotFoundError, RateLimitExceededError
from shared.models.conversation import Conversation, Message

pytestmark = pytest.mark.functional

CONV_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")


def _login(client: TestClient, email: str = "reader@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})


class _FakeAgentService:
    """Stands in for AgentService: a canned turn, event stream, and resume outcome."""

    def __init__(
        self,
        *,
        turn: AgentTurn,
        events: list[AgentEvent],
        resume_turn: AgentTurn | None = None,
        resume_error: Exception | None = None,
    ) -> None:
        self._turn = turn
        self._events = events
        self._resume_turn = resume_turn
        self._resume_error = resume_error
        self.resume_calls: list[dict] = []

    async def run(self, **kwargs: Any) -> AgentTurn:
        return self._turn

    async def stream(self, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        for event in self._events:
            yield event

    async def resume(self, **kwargs: Any) -> AgentTurn:
        self.resume_calls.append(kwargs)
        if self._resume_error is not None:
            raise self._resume_error
        assert self._resume_turn is not None
        return self._resume_turn


class _FakeConversationService:
    """Stands in for ConversationService: records turns, serves canned history."""

    def __init__(
        self,
        *,
        conversations: list[Conversation] | None = None,
        messages: list[Message] | None = None,
        messages_owned: bool = True,
        deleted_owned: bool = True,
    ) -> None:
        self._conversations = conversations or []
        self._messages = messages or []
        self._messages_owned = messages_owned
        self._deleted_owned = deleted_owned
        self.recorded: list[dict] = []
        self.recorded_user_messages: list[dict] = []
        self.recorded_assistant_replies: list[dict] = []
        self.deleted: list[dict] = []

    async def create(self, **kwargs: Any) -> Conversation:
        return _conversation(CONV_ID)

    async def record_turn(self, **kwargs: Any) -> tuple[None, None]:
        self.recorded.append(kwargs)
        return None, None

    async def record_user_message(self, **kwargs: Any) -> None:
        self.recorded_user_messages.append(kwargs)

    async def record_assistant_reply(self, **kwargs: Any) -> None:
        self.recorded_assistant_replies.append(kwargs)

    async def list_conversations(self, **kwargs: Any) -> tuple[list[Conversation], int]:
        return self._conversations, len(self._conversations)

    async def list_messages(self, **kwargs: Any) -> tuple[list[Message], int]:
        if not self._messages_owned:
            raise NotFoundError()
        return self._messages, len(self._messages)

    async def delete(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs)
        if not self._deleted_owned:
            raise NotFoundError()


class _FakeConversationRepository:
    """Stands in for a user-scoped ``ConversationRepository``: canned ownership.

    ``chat_resume`` looks up the target conversation directly through the
    repository (the authorization gate — a paused turn must be resumed only by
    its owner), unlike the other routes which resolve it via the fake
    ``ConversationService`` instead. ``owned=False`` simulates an unknown or
    another user's conversation id.
    """

    def __init__(self, *, owned: bool = True) -> None:
        self._owned = owned

    async def get_or_404(self, conversation_id: uuid.UUID) -> Conversation:
        if not self._owned:
            raise NotFoundError()
        return _conversation(conversation_id)


def _conversation(conversation_id: uuid.UUID, title: str | None = "Odyssey") -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=conversation_id, user_id=uuid.uuid4(), title=title, created_at=now, updated_at=now
    )


def _message(role: MessageRole, content: str) -> Message:
    return Message(
        id=uuid.uuid4(),
        conversation_id=CONV_ID,
        user_id=uuid.uuid4(),
        role=role,
        content=content,
        tool_calls=None,
        created_at=datetime.now(UTC),
    )


class _AlwaysRateLimited:
    """A rate limiter that always reports the caller as over the limit."""

    async def enforce(self, **_: Any) -> None:
        raise RateLimitExceededError()


def _override(
    app: FastAPI,
    *,
    agent: Any = None,
    conversations: Any = None,
    conversation_repo: Any = None,
    rate_limited: bool = False,
) -> None:
    """Install fakes for the agent, conversation service/repository, and tool context."""
    if agent is not None:
        app.dependency_overrides[get_agent_service] = lambda: agent
    if conversations is not None:
        app.dependency_overrides[get_conversation_service] = lambda: conversations
    if conversation_repo is not None:
        app.dependency_overrides[get_conversation_repository] = lambda: conversation_repo
    if rate_limited:
        app.dependency_overrides[get_rate_limit_service] = lambda: _AlwaysRateLimited()
    app.dependency_overrides[get_tool_context] = lambda: SimpleNamespace()


@pytest.fixture(autouse=True)
def _clear_chat_overrides(app: FastAPI) -> Any:
    yield
    for dep in (
        get_agent_service,
        get_conversation_service,
        get_conversation_repository,
        get_rate_limit_service,
        get_tool_context,
    ):
        app.dependency_overrides.pop(dep, None)
    # Also undo any `_override_ws`/`_no_llm_tool_context` shadowing of the real
    # Resources singleton's cached_property values.
    resources = app.state.resources
    for name in (
        "agent_service",
        "conversation_service",
        "chat_model",
        "retrieval_service",
        "progress_service",
        "memory_service",
        "recommendation_service",
        "web_search",
    ):
        resources.__dict__.pop(name, None)


# --- POST /chat -------------------------------------------------------------- #


def test_chat_requires_authentication(client: TestClient, user_repo: FakeUserRepository) -> None:
    resp = client.post("/api/v1/chat", json={"message": "hi"})
    assert resp.status_code == 401


def test_chat_returns_answer_and_conversation_id(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(
            answer="Odysseus narrates.",
            tool_steps=[ToolStep("retrieve_chunks", {"q": "x"}, "[1] …")],
            trace_id="trace-abc",
        ),
        events=[],
    )
    convo = _FakeConversationService()
    _override(app, agent=agent, conversations=convo)

    resp = client.post("/api/v1/chat", json={"message": "who narrates?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == str(CONV_ID)
    assert body["answer"] == "Odysseus narrates."
    assert body["blocked"] is False
    assert body["tool_steps"][0]["name"] == "retrieve_chunks"
    assert body["trace_id"] == "trace-abc"  # correlates the turn to its trace
    # The turn was persisted (user text + assistant answer).
    assert convo.recorded and convo.recorded[0]["assistant_text"] == "Odysseus narrates."


def test_chat_returns_429_when_rate_limited(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    _override(
        app,
        agent=_FakeAgentService(turn=AgentTurn(answer="unused"), events=[]),
        conversations=_FakeConversationService(),
        rate_limited=True,
    )

    resp = client.post("/api/v1/chat", json={"message": "hello"})

    assert resp.status_code == 429
    assert resp.json()["code"] == "RATE_LIMITED"


def test_chat_blocked_turn_reports_blocked(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="I only help with your reading.", blocked=True), events=[]
    )
    _override(app, agent=agent, conversations=_FakeConversationService())

    resp = client.post("/api/v1/chat", json={"message": "write a poem about taxes"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["blocked"] is True
    assert body["answer"] == "I only help with your reading."


def test_chat_rejects_empty_message(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    # Overrides installed so dependency resolution succeeds and the empty-message
    # body is what fails — a 422 from validation, not a 500 from a missing LLM key.
    _override(
        app,
        agent=_FakeAgentService(turn=AgentTurn(answer="x"), events=[]),
        conversations=_FakeConversationService(),
    )
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_accepts_an_audio_attachment(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(turn=AgentTurn(answer="You asked about chapter one."), events=[])
    convo = _FakeConversationService()
    _override(app, agent=agent, conversations=convo)

    resp = client.post(
        "/api/v1/chat",
        json={
            "message": "",
            "parts": [
                {
                    "kind": "audio",
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(b"RIFF....").decode(),
                }
            ],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "You asked about chapter one."
    # The persisted user turn notes the attachment (the transcript lives in the
    # agent's reasoning, not the curated visible history).
    assert "audio attachment" in convo.recorded[0]["user_text"]


def test_chat_rejects_a_mismatched_attachment_type(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    _override(
        app,
        agent=_FakeAgentService(turn=AgentTurn(answer="x"), events=[]),
        conversations=_FakeConversationService(),
    )
    resp = client.post(
        "/api/v1/chat",
        json={
            "message": "look",
            "parts": [
                {
                    "kind": "audio",
                    "mime_type": "image/png",  # image mime on an audio part
                    "data": base64.b64encode(b"x").decode(),
                }
            ],
        },
    )
    assert resp.status_code == 422


# --- HITL: interrupt on /chat, resume via /chat/{id}/resume ------------------ #

_TOOL_CALL = {"name": "web_search", "args": {"query": "release date"}, "id": "call_1"}
_REASON = "'web_search' reaches beyond your stored data and needs your approval."


def test_chat_returns_interrupt_when_the_turn_pauses_for_approval(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(
            answer="", interrupted=True, interrupt={"tool_call": _TOOL_CALL, "reason": _REASON}
        ),
        events=[],
    )
    convo = _FakeConversationService()
    _override(app, agent=agent, conversations=convo)

    resp = client.post("/api/v1/chat", json={"message": "search for the release date"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == ""
    assert body["blocked"] is False
    assert body["interrupt"] == {"tool_call": _TOOL_CALL, "reason": _REASON}
    # The question is persisted now; there's no answer yet to pair it with.
    assert convo.recorded_user_messages
    assert convo.recorded_user_messages[0]["user_text"] == "search for the release date"
    assert not convo.recorded


def test_chat_stream_emits_interrupt_and_defers_persistence(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused for stream"),
        events=[
            ToolCallEvent(name="web_search", args={"query": "release date"}, id="call_1"),
            InterruptEvent(
                payload={"kind": "tool_approval", "tool_call": _TOOL_CALL, "reason": _REASON}
            ),
        ],
    )
    convo = _FakeConversationService()
    _override(app, agent=agent, conversations=convo)

    resp = client.post("/api/v1/chat/stream", json={"message": "search for the release date"})

    assert resp.status_code == 200
    assert _parse_sse(resp.text) == ["conversation", "tool_call", "interrupt"]
    assert convo.recorded_user_messages
    assert not convo.recorded


_SPOILER_INTERRUPT = {
    "kind": "spoiler_warning",
    "document_id": "22222222-2222-2222-2222-222222222222",
    "document_title": "The Odyssey",
    "current_page": 50,
    "reason": "reveals who Odysseus fights in the end",
}


def test_chat_returns_a_spoiler_warning_interrupt(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    # A generation-time spoiler flag (FR-18.3/18.4) pauses the turn the same way
    # a gated tool call does — the route's interrupt handling is kind-agnostic,
    # so this proves the new kind flows through unchanged end to end.
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="", interrupted=True, interrupt=_SPOILER_INTERRUPT), events=[]
    )
    convo = _FakeConversationService()
    _override(app, agent=agent, conversations=convo)

    resp = client.post("/api/v1/chat", json={"message": "what happens at the end?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == ""
    assert body["interrupt"] == _SPOILER_INTERRUPT
    assert convo.recorded_user_messages
    assert not convo.recorded


def test_chat_resume_approve_persists_the_assistant_reply(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused"),
        events=[],
        resume_turn=AgentTurn(
            answer="The sequel releases in March.",
            tool_steps=[ToolStep("web_search", {"query": "release date"}, "March.")],
            trace_id="trace-resume",
        ),
    )
    convo = _FakeConversationService()
    _override(
        app, agent=agent, conversations=convo, conversation_repo=_FakeConversationRepository()
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"decision": "approve"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The sequel releases in March."
    assert body["interrupt"] is None
    assert body["trace_id"] == "trace-resume"
    assert agent.resume_calls[0]["decision"] == {"decision": "approve"}
    assert convo.recorded_assistant_replies
    assert convo.recorded_assistant_replies[0]["assistant_text"] == "The sequel releases in March."


def test_chat_resume_deny_is_a_valid_decision(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused"),
        events=[],
        resume_turn=AgentTurn(answer="I could not search, so here's what I know."),
    )
    _override(
        app,
        agent=agent,
        conversations=_FakeConversationService(),
        conversation_repo=_FakeConversationRepository(),
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"decision": "deny"})

    assert resp.status_code == 200
    assert agent.resume_calls[0]["decision"] == {"decision": "deny"}


def test_chat_resume_edit_requires_args(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    _override(
        app,
        agent=_FakeAgentService(turn=AgentTurn(answer="unused"), events=[]),
        conversations=_FakeConversationService(),
        conversation_repo=_FakeConversationRepository(),
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"decision": "edit"})

    assert resp.status_code == 422


def test_chat_resume_empty_payload_is_rejected(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    _override(
        app,
        agent=_FakeAgentService(turn=AgentTurn(answer="unused"), events=[]),
        conversations=_FakeConversationService(),
        conversation_repo=_FakeConversationRepository(),
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={})

    assert resp.status_code == 422


def test_chat_resume_page_range_answer_needs_no_decision(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused"),
        events=[],
        resume_turn=AgentTurn(answer="Telemachus searches for his father."),
    )
    _override(
        app,
        agent=agent,
        conversations=_FakeConversationService(),
        conversation_repo=_FakeConversationRepository(),
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"page_start": 1, "page_end": 42})

    assert resp.status_code == 200
    assert agent.resume_calls[0]["decision"] == {"page_start": 1, "page_end": 42}


def test_chat_resume_still_interrupted_does_not_persist(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused"),
        events=[],
        resume_turn=AgentTurn(
            answer="", interrupted=True, interrupt={"tool_call": _TOOL_CALL, "reason": _REASON}
        ),
    )
    convo = _FakeConversationService()
    _override(
        app, agent=agent, conversations=convo, conversation_repo=_FakeConversationRepository()
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"decision": "approve"})

    assert resp.status_code == 200
    assert resp.json()["interrupt"] == {"tool_call": _TOOL_CALL, "reason": _REASON}
    assert not convo.recorded_assistant_replies


def test_chat_resume_of_unowned_conversation_is_404(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    _override(
        app,
        agent=_FakeAgentService(turn=AgentTurn(answer="unused"), events=[]),
        conversations=_FakeConversationService(),
        conversation_repo=_FakeConversationRepository(owned=False),
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"decision": "approve"})

    assert resp.status_code == 404


def test_chat_resume_without_a_pending_interrupt_is_a_conflict(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused"), events=[], resume_error=NoPendingInterruptError()
    )
    _override(
        app,
        agent=agent,
        conversations=_FakeConversationService(),
        conversation_repo=_FakeConversationRepository(),
    )

    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"decision": "approve"})

    assert resp.status_code == 409
    assert resp.json()["code"] == "NO_PENDING_INTERRUPT"


def test_chat_resume_requires_authentication(
    client: TestClient, user_repo: FakeUserRepository
) -> None:
    resp = client.post(f"/api/v1/chat/{CONV_ID}/resume", json={"decision": "approve"})
    assert resp.status_code == 401


# --- POST /chat/stream (SSE) ------------------------------------------------- #


def _parse_sse(text: str) -> list[str]:
    """Return the ordered list of SSE event names in a response body."""
    return [line[len("event: ") :] for line in text.splitlines() if line.startswith("event: ")]


def test_chat_stream_emits_events_in_order(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused for stream"),
        events=[
            ToolCallEvent(name="retrieve_chunks", args={"q": "x"}, id="c1"),
            ToolResultEvent(name="retrieve_chunks", content="[1] …", id="c1"),
            TokenEvent(text="Odysseus "),
            TokenEvent(text="narrates."),
            DoneEvent(answer="Odysseus narrates.", trace_id="trace-stream"),
        ],
    )
    convo = _FakeConversationService()
    _override(app, agent=agent, conversations=convo)

    resp = client.post("/api/v1/chat/stream", json={"message": "who narrates?"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # A leading conversation frame, then tool step(s), then tokens, then done.
    assert _parse_sse(resp.text) == [
        "conversation",
        "tool_call",
        "tool_result",
        "token",
        "token",
        "done",
    ]
    assert str(CONV_ID) in resp.text
    # The terminal done frame carries the turn's trace id for client correlation.
    assert '"trace_id": "trace-stream"' in resp.text
    # Persisted with the sanitized done answer after the stream completed.
    assert convo.recorded[0]["assistant_text"] == "Odysseus narrates."


def test_chat_stream_blocked_emits_only_blocked(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="", blocked=True),
        events=[BlockedEvent(reason="I only help with your reading.")],
    )
    convo = _FakeConversationService()
    _override(app, agent=agent, conversations=convo)

    resp = client.post("/api/v1/chat/stream", json={"message": "off topic"})

    assert resp.status_code == 200
    assert _parse_sse(resp.text) == ["conversation", "blocked"]
    assert convo.recorded[0]["assistant_text"] == "I only help with your reading."


# --- WS /chat/ws --------------------------------------------------------------- #


def _override_ws(app: FastAPI, *, agent: Any, conversations: Any) -> None:
    """Patch the real ``Resources`` singleton's agent/conversation service.

    ``chat_ws`` reads ``resources.agent_service``/``.conversation_service``
    directly rather than as ``Depends`` params — deliberately, so an
    unauthenticated connection never pays to build the real agent (see the
    route's docstring) — so faking them means shadowing the ``cached_property``
    values on the one real ``Resources`` instance the app lifespan built. Plain
    attribute assignment shadows the descriptor via the instance ``__dict__``,
    exactly like a normal cache hit; ``_clear_chat_overrides`` undoes it.
    """
    resources = app.state.resources
    resources.agent_service = agent
    resources.conversation_service = conversations


@pytest.fixture
def _no_llm_tool_context(app: FastAPI) -> None:
    """Stub the ``Resources`` collaborators ``chat_ws`` reads to build its ``ToolContext``.

    ``chat_ws`` builds its own per-turn ``ToolContext`` from the real ``Resources``
    singleton (a WebSocket connection outlives any one request-scoped session,
    unlike the HTTP routes' ``get_tool_context`` dependency chain) — several of
    those collaborators are lazy, key- or embedder-backed providers
    (``chat_model``, the embedder-backed services) these tests don't configure.
    Plain instance-attribute assignment, like ``_override_ws`` uses for
    ``agent_service``/``conversation_service``, not ``monkeypatch.setattr``:
    the latter reads the *current* value first to remember it for teardown,
    which would itself trigger the real (failing) ``cached_property`` build.
    ``_clear_chat_overrides`` undoes the shadowing afterwards.
    """
    resources = app.state.resources
    resources.chat_model = lambda **_: SimpleNamespace()
    for name in (
        "retrieval_service",
        "progress_service",
        "memory_service",
        "recommendation_service",
        "web_search",
    ):
        setattr(resources, name, SimpleNamespace())


def test_chat_ws_requires_authentication(client: TestClient, user_repo: FakeUserRepository) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/api/v1/chat/ws"),
    ):
        pass
    assert exc_info.value.code == 4401


def test_chat_ws_streams_turn_and_persists(
    app: FastAPI,
    client: TestClient,
    user_repo: FakeUserRepository,
    _no_llm_tool_context: None,
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="unused for stream"),
        events=[
            ToolCallEvent(name="retrieve_chunks", args={"q": "x"}, id="c1"),
            ToolResultEvent(name="retrieve_chunks", content="[1] …", id="c1"),
            TokenEvent(text="Odysseus "),
            TokenEvent(text="narrates."),
            DoneEvent(answer="Odysseus narrates.", trace_id="trace-ws"),
        ],
    )
    convo = _FakeConversationService()
    _override_ws(app, agent=agent, conversations=convo)

    with client.websocket_connect("/api/v1/chat/ws") as ws:
        ws.send_json({"message": "who narrates?"})
        frames = [ws.receive_json() for _ in range(6)]

    assert [f["type"] for f in frames] == [
        "conversation",
        "tool_call",
        "tool_result",
        "token",
        "token",
        "done",
    ]
    assert frames[-1]["trace_id"] == "trace-ws"
    assert convo.recorded[0]["assistant_text"] == "Odysseus narrates."


def test_chat_ws_blocked_turn_emits_only_blocked(
    app: FastAPI,
    client: TestClient,
    user_repo: FakeUserRepository,
    _no_llm_tool_context: None,
) -> None:
    _login(client)
    agent = _FakeAgentService(
        turn=AgentTurn(answer="", blocked=True),
        events=[BlockedEvent(reason="I only help with your reading.")],
    )
    convo = _FakeConversationService()
    _override_ws(app, agent=agent, conversations=convo)

    with client.websocket_connect("/api/v1/chat/ws") as ws:
        ws.send_json({"message": "off topic"})
        frames = [ws.receive_json(), ws.receive_json()]

    assert [f["type"] for f in frames] == ["conversation", "blocked"]
    assert convo.recorded[0]["assistant_text"] == "I only help with your reading."


def test_chat_ws_supports_multiple_turns_on_one_connection(
    app: FastAPI,
    client: TestClient,
    user_repo: FakeUserRepository,
    _no_llm_tool_context: None,
) -> None:
    _login(client)
    agent = _FakeAgentService(turn=AgentTurn(answer="unused"), events=[DoneEvent(answer="ok")])
    convo = _FakeConversationService()
    _override_ws(app, agent=agent, conversations=convo)

    with client.websocket_connect("/api/v1/chat/ws") as ws:
        ws.send_json({"message": "first"})
        first = [ws.receive_json(), ws.receive_json()]
        ws.send_json({"message": "second"})
        second = [ws.receive_json(), ws.receive_json()]

    assert [f["type"] for f in first] == ["conversation", "done"]
    assert [f["type"] for f in second] == ["conversation", "done"]
    assert len(convo.recorded) == 2


def test_chat_ws_rejects_invalid_json_and_stays_open(
    app: FastAPI,
    client: TestClient,
    user_repo: FakeUserRepository,
    _no_llm_tool_context: None,
) -> None:
    _login(client)
    agent = _FakeAgentService(turn=AgentTurn(answer="unused"), events=[DoneEvent(answer="ok")])
    _override_ws(app, agent=agent, conversations=_FakeConversationService())

    with client.websocket_connect("/api/v1/chat/ws") as ws:
        ws.send_text("not json")
        error = ws.receive_json()
        assert error == {
            "type": "error",
            "detail": "Invalid JSON payload.",
            "code": "BAD_REQUEST",
        }
        # The connection survives a bad frame — the next turn still works.
        ws.send_json({"message": "hello"})
        frames = [ws.receive_json(), ws.receive_json()]

    assert [f["type"] for f in frames] == ["conversation", "done"]


def test_chat_ws_rejects_invalid_payload(
    app: FastAPI,
    client: TestClient,
    user_repo: FakeUserRepository,
    _no_llm_tool_context: None,
) -> None:
    _login(client)
    _override_ws(
        app,
        agent=_FakeAgentService(turn=AgentTurn(answer="x"), events=[]),
        conversations=_FakeConversationService(),
    )

    with client.websocket_connect("/api/v1/chat/ws") as ws:
        ws.send_json({"message": ""})  # neither text nor attachments
        error = ws.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "VALIDATION_ERROR"


# --- history ----------------------------------------------------------------- #


def test_list_conversations_returns_paginated_envelope(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    convo = _FakeConversationService(conversations=[_conversation(CONV_ID, "Odyssey")])
    _override(app, conversations=convo)

    resp = client.get("/api/v1/conversations")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["page"] == 1
    assert body["items"][0]["title"] == "Odyssey"


def test_list_messages_returns_transcript(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    convo = _FakeConversationService(
        messages=[
            _message(MessageRole.USER, "who narrates?"),
            _message(MessageRole.ASSISTANT, "Odysseus does."),
        ]
    )
    _override(app, conversations=convo)

    resp = client.get(f"/api/v1/conversations/{CONV_ID}/messages")

    assert resp.status_code == 200
    body = resp.json()
    assert [m["role"] for m in body["items"]] == ["user", "assistant"]


def test_list_messages_of_unowned_conversation_is_404(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    _override(app, conversations=_FakeConversationService(messages_owned=False))

    resp = client.get(f"/api/v1/conversations/{CONV_ID}/messages")

    assert resp.status_code == 404


def test_delete_conversation_requires_authentication(client: TestClient) -> None:
    resp = client.delete(f"/api/v1/conversations/{CONV_ID}")
    assert resp.status_code == 401


def test_delete_conversation_returns_no_content(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    convo = _FakeConversationService()
    _override(app, conversations=convo)

    resp = client.delete(f"/api/v1/conversations/{CONV_ID}")

    assert resp.status_code == 204
    assert convo.deleted[0]["conversation_id"] == CONV_ID


def test_delete_unowned_conversation_is_404(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login(client)
    _override(app, conversations=_FakeConversationService(deleted_owned=False))

    resp = client.delete(f"/api/v1/conversations/{CONV_ID}")

    assert resp.status_code == 404
