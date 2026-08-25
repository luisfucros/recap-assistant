"""Chat routes: talk to the assistant and read past conversations.

Three surfaces over one turn:

* ``POST /chat`` — run a turn to completion and return the answer.
* ``POST /chat/stream`` — the same turn as **Server-Sent Events**: tool steps,
  then answer tokens, then a terminal ``done`` (or a lone ``blocked``).
* ``WS /chat/ws`` — the same event vocabulary over a WebSocket, many turns per
  connection (one inbound JSON message per turn).
* ``GET /conversations`` and ``GET /conversations/{id}/messages`` — the history.
* ``DELETE /conversations/{id}`` — delete a thread, its messages, and its
  checkpointed agent state.

Handlers stay thin: the agent run is delegated to
:class:`~api.services.agent_service.AgentService` and the transcript to
:class:`~api.services.conversation_service.ConversationService`. Both the
conversation (via a user-scoped repository) and the agent's per-turn
:class:`~api.agent.context.ToolContext` carry the owner from the access-token
cookie, never the request body, so a caller can only ever touch their own threads.
The conversation id doubles as the LangGraph checkpointer ``thread_id``, so a
follow-up turn on the same id resumes prior context.
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from api.agent.context import ToolContext, build_tool_context
from api.agent.events import BlockedEvent, DoneEvent, InterruptEvent, ToolCallEvent, ToolResultEvent
from api.deps import (
    AgentServiceDep,
    AuthServiceDep,
    ChatRateLimit,
    ConversationRepositoryDep,
    ConversationServiceDep,
    CurrentUser,
    DbSession,
    MessageRepositoryDep,
    ResourcesDep,
    SettingsDep,
    ToolContextDep,
    UserRepositoryDep,
    resolve_user_from_token,
)
from api.resources import Resources
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationPage,
    ConversationPublic,
    MessagePage,
    MessagePublic,
    ResumeRequest,
    ToolStepPublic,
)
from api.security import ACCESS_COOKIE
from api.services.agent_service import AgentService, AgentTurn, ToolStep
from api.services.conversation_service import ConversationService
from api.services.multimodal_service import MediaPart
from shared.core.config import Settings
from shared.core.enums import Language
from shared.core.errors import AppError, AuthenticationError, InvalidInputError
from shared.models.conversation import Conversation
from shared.models.user import User
from shared.repositories import (
    ConversationRepository,
    MessageRepository,
)

router = APIRouter(tags=["chat"])

# WebSocket close code for a missing/invalid session cookie — 4401 sits in the
# RFC 6455 private-use range (4000-4999); there's no reserved code for "auth".
_WS_UNAUTHORIZED = 4401

# Human-readable names for the answer language, rendered into the generate and
# guardrail_in prompts (clearer to the model than the ISO code). One entry per
# supported language.
_LANGUAGE_NAMES: dict[Language, str] = {
    Language.EN: "English",
    Language.ES: "Spanish",
    Language.DE: "German",
    Language.FR: "French",
    Language.IT: "Italian",
}


def _display_name(user: CurrentUser) -> str:
    """The name the assistant addresses the reader by (falls back gracefully)."""
    return user.display_name or "there"


def _answer_language(user: CurrentUser) -> str:
    """The language the assistant answers in (the reader's ``preferred_language``)."""
    return _LANGUAGE_NAMES.get(user.preferred_language, "English")


def _tool_calls_json(steps: list[ToolStep]) -> dict | None:
    """Serialize a turn's tool steps for the persisted assistant message (or None)."""
    if not steps:
        return None
    return {"steps": [{"name": s.name, "args": s.args, "result": s.result} for s in steps]}


def _media_parts(payload: ChatRequest, settings: Settings) -> list[MediaPart]:
    """Decode the request's attachments to :class:`MediaPart`, enforcing the byte cap.

    Structural checks (base64 validity, mime allowlist, kind/mime match) already
    ran in the schema; here we apply the configured per-attachment size limit —
    which depends on ``Settings`` and so can't live in the schema — and hand the
    agent decoded bytes.
    """
    parts: list[MediaPart] = []
    for part in payload.parts:
        if len(part.data) > settings.chat_media_max_bytes:
            raise InvalidInputError(
                f"attachment exceeds the {settings.chat_media_max_bytes}-byte limit",
                code="ATTACHMENT_TOO_LARGE",
            )
        parts.append(MediaPart(kind=part.kind, data=bytes(part.data), mime_type=part.mime_type))
    return parts


def _user_transcript(message: str, parts: list[MediaPart]) -> str:
    """The user's turn as stored in the visible transcript.

    The derived transcript/description lives inside the agent's reasoning, not the
    curated history, so here we keep the typed text and note any attachments (e.g.
    ``"[1 audio attachment]"``) so a re-opened thread shows something happened.
    """
    if not parts:
        return message
    counts: dict[str, int] = {}
    for part in parts:
        counts[part.kind] = counts.get(part.kind, 0) + 1
    tags = ", ".join(
        f"{count} {kind} attachment{'s' if count > 1 else ''}" for kind, count in counts.items()
    )
    return f"{message}\n\n[{tags}]".strip() if message.strip() else f"[{tags}]"


async def _resolve_conversation(
    *,
    conversation_service: ConversationService,
    conversations: ConversationRepository,
    session: AsyncSession,
    requested_id: uuid.UUID | None,
) -> Conversation:
    """Return the requested conversation (404 if not the caller's) or create one.

    Resolving up front — before any streaming begins — means an unauthorized or
    unknown id fails as a clean HTTP error rather than mid-stream.
    """
    if requested_id is not None:
        return await conversations.get_or_404(requested_id)
    return await conversation_service.create(conversations=conversations, session=session)


@router.post("/chat", response_model=ChatResponse, summary="Send a message and get the answer")
async def chat(
    payload: ChatRequest,
    user: CurrentUser,
    agent_service: AgentServiceDep,
    conversation_service: ConversationServiceDep,
    tool_context: ToolContextDep,
    conversations: ConversationRepositoryDep,
    messages: MessageRepositoryDep,
    session: DbSession,
    settings: SettingsDep,
    _rl: ChatRateLimit,
) -> ChatResponse:
    """Run one chat turn to completion (non-streaming).

    Continues the thread in ``conversation_id`` or starts a new one (returned in
    the response). Accepts typed text and/or audio/image attachments (FR-19). The
    turn is persisted to the transcript; a guardrail refusal comes back with
    ``blocked=true`` and the polite reason as ``answer``. Rate-limited per user
    (429 ``RATE_LIMITED``) — each turn triggers a real LLM call.
    """
    media_parts = _media_parts(payload, settings)
    conversation = await _resolve_conversation(
        conversation_service=conversation_service,
        conversations=conversations,
        session=session,
        requested_id=payload.conversation_id,
    )
    turn = await agent_service.run(
        tool_context=tool_context,
        display_name=_display_name(user),
        message=payload.message,
        conversation_id=str(conversation.id),
        answer_language=_answer_language(user),
        media_parts=media_parts,
    )
    if turn.interrupted:
        # Paused on a gated tool call: the user's question is real and worth
        # showing now, even though there's no answer yet — the assistant reply
        # is persisted once POST /chat/{conversation_id}/resume completes it.
        await conversation_service.record_user_message(
            conversations=conversations,
            messages=messages,
            session=session,
            conversation_id=conversation.id,
            user_text=_user_transcript(payload.message, media_parts),
        )
        return _interrupted_response(conversation.id, turn)
    await conversation_service.record_turn(
        conversations=conversations,
        messages=messages,
        session=session,
        conversation_id=conversation.id,
        user_text=_user_transcript(payload.message, media_parts),
        assistant_text=turn.answer,
        tool_calls=_tool_calls_json(turn.tool_steps),
    )
    return ChatResponse(
        conversation_id=conversation.id,
        answer=turn.answer,
        blocked=turn.blocked,
        tool_steps=[
            ToolStepPublic(name=s.name, args=s.args, result=s.result) for s in turn.tool_steps
        ],
        trace_id=turn.trace_id,
    )


def _interrupted_response(conversation_id: uuid.UUID, turn: AgentTurn) -> ChatResponse:
    """Shape a paused turn as a :class:`ChatResponse` (no answer yet, HITL)."""
    return ChatResponse(
        conversation_id=conversation_id,
        answer="",
        blocked=False,
        tool_steps=[],
        trace_id=turn.trace_id,
        interrupt=turn.interrupt or {},
    )


def _sse(event_type: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _run_turn(
    *,
    agent_service: AgentService,
    conversation_service: ConversationService,
    conversations: ConversationRepository,
    messages: MessageRepository,
    session: AsyncSession,
    tool_context: ToolContext,
    display_name: str,
    answer_language: str,
    conversation: Conversation,
    message: str,
    media_parts: list[MediaPart],
) -> AsyncIterator[dict]:
    """Yield the turn's frames as plain dicts, then persist the transcript once complete.

    Transport-agnostic: ``/chat/stream`` wraps each frame as an SSE event and
    ``/chat/ws`` sends it straight as a JSON message, so both surfaces relay the
    same event vocabulary. Emits a leading ``conversation`` frame (so the client
    learns the thread id), the ordered agent events, and — after the stream ends
    — writes the user message and the assistant reply. The answer persisted is
    the terminal ``done`` (sanitized) text, or the ``blocked`` reason. A lone
    ``interrupt`` frame means a gated tool call paused the turn for approval
    (HITL): nothing is persisted yet — the caller resumes via
    ``POST /chat/{conversation_id}/resume``, which persists once the turn
    actually completes.
    """
    yield {"type": "conversation", "conversation_id": str(conversation.id)}

    calls: dict[str, tuple[str, dict]] = {}
    steps: list[ToolStep] = []
    answer = ""
    interrupted = False
    async for event in agent_service.stream(
        tool_context=tool_context,
        display_name=display_name,
        message=message,
        conversation_id=str(conversation.id),
        answer_language=answer_language,
        media_parts=media_parts,
    ):
        yield event.as_dict()
        if isinstance(event, ToolCallEvent):
            calls[event.id] = (event.name, event.args)
        elif isinstance(event, ToolResultEvent):
            name, args = calls.get(event.id, (event.name, {}))
            steps.append(ToolStep(name=name, args=args, result=event.content))
        elif isinstance(event, DoneEvent):
            answer = event.answer
        elif isinstance(event, BlockedEvent):
            answer = event.reason
        elif isinstance(event, InterruptEvent):
            interrupted = True

    if interrupted:
        # Same reasoning as the non-streaming /chat handler: the question is
        # real, the answer isn't yet — persist just the user's side now.
        await conversation_service.record_user_message(
            conversations=conversations,
            messages=messages,
            session=session,
            conversation_id=conversation.id,
            user_text=_user_transcript(message, media_parts),
        )
        return

    await conversation_service.record_turn(
        conversations=conversations,
        messages=messages,
        session=session,
        conversation_id=conversation.id,
        user_text=_user_transcript(message, media_parts),
        assistant_text=answer,
        tool_calls=_tool_calls_json(steps),
    )


async def _stream_turn(
    *,
    agent_service: AgentService,
    conversation_service: ConversationService,
    conversations: ConversationRepository,
    messages: MessageRepository,
    session: AsyncSession,
    tool_context: ToolContext,
    display_name: str,
    answer_language: str,
    conversation: Conversation,
    message: str,
    media_parts: list[MediaPart],
) -> AsyncIterator[str]:
    """SSE adapter over :func:`_run_turn` — one ``event:``/``data:`` frame per dict."""
    async for frame in _run_turn(
        agent_service=agent_service,
        conversation_service=conversation_service,
        conversations=conversations,
        messages=messages,
        session=session,
        tool_context=tool_context,
        display_name=display_name,
        answer_language=answer_language,
        conversation=conversation,
        message=message,
        media_parts=media_parts,
    ):
        yield _sse(frame["type"], frame)


@router.post("/chat/stream", summary="Send a message and stream the answer (SSE)")
async def chat_stream(
    payload: ChatRequest,
    user: CurrentUser,
    agent_service: AgentServiceDep,
    conversation_service: ConversationServiceDep,
    tool_context: ToolContextDep,
    conversations: ConversationRepositoryDep,
    messages: MessageRepositoryDep,
    session: DbSession,
    settings: SettingsDep,
    _rl: ChatRateLimit,
) -> StreamingResponse:
    """Stream one chat turn as Server-Sent Events.

    Accepts typed text and/or audio/image attachments (FR-19). Frame order: a
    ``conversation`` frame (the thread id), then any ``tool_call``/``tool_result``
    pairs, then answer ``token`` frames, then a terminal ``done`` — or a single
    ``blocked`` frame if a guardrail stops the turn. The turn is persisted after
    the stream completes. Rate-limited per user (429 ``RATE_LIMITED``).
    """
    media_parts = _media_parts(payload, settings)
    conversation = await _resolve_conversation(
        conversation_service=conversation_service,
        conversations=conversations,
        session=session,
        requested_id=payload.conversation_id,
    )
    stream = _stream_turn(
        agent_service=agent_service,
        conversation_service=conversation_service,
        conversations=conversations,
        messages=messages,
        session=session,
        tool_context=tool_context,
        display_name=_display_name(user),
        answer_language=_answer_language(user),
        conversation=conversation,
        message=payload.message,
        media_parts=media_parts,
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        # Disable client/proxy buffering so frames reach the browser as they're
        # produced (X-Accel-Buffering turns off nginx's response buffering).
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/chat/{conversation_id}/resume",
    response_model=ChatResponse,
    summary="Resume a turn paused for a gated tool call's approval",
)
async def chat_resume(
    conversation_id: uuid.UUID,
    payload: ResumeRequest,
    user: CurrentUser,
    agent_service: AgentServiceDep,
    conversation_service: ConversationServiceDep,
    tool_context: ToolContextDep,
    conversations: ConversationRepositoryDep,
    messages: MessageRepositoryDep,
    session: DbSession,
    _rl: ChatRateLimit,
) -> ChatResponse:
    """Continue a turn paused on a gated tool call with the user's decision (HITL).

    ``decision`` is ``approve`` (run the call unchanged), ``deny`` (skip it —
    the model sees a denial instead of a result), or ``edit`` (run it with
    ``args`` in place of the model's). Resuming a conversation with nothing
    paused (already resolved, or never interrupted) fails with ``409
    NO_PENDING_INTERRUPT`` rather than silently running an empty turn. A turn
    that pauses again (another gated call further down the same turn) comes
    back with ``interrupt`` set once more and nothing persisted yet; otherwise
    the assistant's reply is appended to the transcript alongside the user
    message the original, paused call already recorded. Rate-limited per user
    (429 ``RATE_LIMITED``) — resuming can run a real tool call (e.g. web search).
    """
    conversation = await conversations.get_or_404(conversation_id)
    turn = await agent_service.resume(
        tool_context=tool_context,
        display_name=_display_name(user),
        conversation_id=str(conversation.id),
        decision=payload.model_dump(exclude_none=True),
        answer_language=_answer_language(user),
    )
    if turn.interrupted:
        return _interrupted_response(conversation.id, turn)
    await conversation_service.record_assistant_reply(
        conversations=conversations,
        messages=messages,
        session=session,
        conversation_id=conversation.id,
        assistant_text=turn.answer,
        tool_calls=_tool_calls_json(turn.tool_steps),
    )
    return ChatResponse(
        conversation_id=conversation.id,
        answer=turn.answer,
        blocked=turn.blocked,
        tool_steps=[
            ToolStepPublic(name=s.name, args=s.args, result=s.result) for s in turn.tool_steps
        ],
        trace_id=turn.trace_id,
    )


@router.websocket("/chat/ws")
async def chat_ws(
    websocket: WebSocket,
    auth: AuthServiceDep,
    users: UserRepositoryDep,
    resources: ResourcesDep,
) -> None:
    """Chat over a WebSocket: many turns on one connection, each streamed as JSON.

    Auth mirrors the other chat routes (the access-token cookie), but is
    resolved *inside* this function rather than as a ``CurrentUser`` parameter:
    raising from a dependency would make FastAPI try to render the 401 as an
    HTTP response over the WebSocket ASGI scope, which raises a ``RuntimeError``
    instead of closing cleanly (see ``api.deps.get_current_user``'s docstring).
    An unauthenticated connection is closed during the handshake instead. Only
    ``auth``/``users``/``resources`` are declared params — all cheap and
    side-effect-free to resolve — deliberately *not* the agent/conversation
    services: those are read from ``resources`` below, after auth succeeds, so
    an anonymous or bad-cookie probe never pays for building the agent's LLM
    wiring. Each inbound text frame is a :class:`ChatRequest` JSON payload; each
    turn's outbound frames share the SSE vocabulary (``conversation``,
    ``tool_call``, ``tool_result``, ``token``, ``done``, or a lone ``blocked``),
    every frame carrying a ``type`` field, so the two transports speak one event
    model. A turn-level error (bad payload, unknown conversation) sends a
    ``{"type": "error", ...}`` frame and keeps the connection open for the next
    turn; only an auth failure or a client disconnect ends the connection.
    """
    token = websocket.cookies.get(ACCESS_COOKIE)
    try:
        user = await resolve_user_from_token(token, auth, users)
    except AuthenticationError:
        await websocket.close(code=_WS_UNAUTHORIZED)
        return

    await websocket.accept()
    settings = resources.settings
    agent_service = resources.agent_service
    conversation_service = resources.conversation_service

    # WS bypasses RequestIDMiddleware (websocket scope); bind one id for the
    # whole connection instead, so its turns' logs are still correlatable.
    with logger.contextualize(request_id=str(uuid.uuid4())):
        await _chat_ws_loop(
            websocket, user, settings, agent_service, conversation_service, resources
        )


async def _chat_ws_loop(
    websocket: WebSocket,
    user: User,
    settings: Settings,
    agent_service: AgentService,
    conversation_service: ConversationService,
    resources: Resources,
) -> None:
    """The per-connection receive/dispatch loop backing :func:`chat_ws`."""
    try:
        while True:
            try:
                raw = await websocket.receive_json()
            except ValueError:
                await websocket.send_json(
                    {"type": "error", "detail": "Invalid JSON payload.", "code": "BAD_REQUEST"}
                )
                continue
            try:
                payload = ChatRequest.model_validate(raw)
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": "Request validation failed.",
                        "code": "VALIDATION_ERROR",
                        # `include_context=False`: a raised-ValueError entry's
                        # `ctx.error` holds the raw exception object, which isn't
                        # JSON-serializable.
                        "errors": exc.errors(include_url=False, include_context=False),
                    }
                )
                continue

            try:
                await resources.rate_limit_service.enforce(
                    key=f"ratelimit:chat:{user.id}",
                    limit=settings.rate_limit_chat_max,
                    window_seconds=settings.rate_limit_chat_window_seconds,
                )
                async with resources.sessionmaker() as session:
                    conversations = ConversationRepository(session, user.id)
                    messages = MessageRepository(session, user.id)
                    media_parts = _media_parts(payload, settings)
                    conversation = await _resolve_conversation(
                        conversation_service=conversation_service,
                        conversations=conversations,
                        session=session,
                        requested_id=payload.conversation_id,
                    )
                    tool_context = build_tool_context(
                        session=session,
                        user=user,
                        progress_service=resources.progress_service,
                        retrieval_service=resources.retrieval_service,
                        summarizer=resources.chat_model(tier="cheap"),
                        prompts=resources.prompts,
                        memory_service=resources.memory_service,
                        recommendation_service=resources.recommendation_service,
                        web_search=lambda: resources.web_search,
                        usage_service=resources.usage_service,
                    )
                    async for frame in _run_turn(
                        agent_service=agent_service,
                        conversation_service=conversation_service,
                        conversations=conversations,
                        messages=messages,
                        session=session,
                        tool_context=tool_context,
                        display_name=_display_name(user),
                        answer_language=_answer_language(user),
                        conversation=conversation,
                        message=payload.message,
                        media_parts=media_parts,
                    ):
                        await websocket.send_json(frame)
            except WebSocketDisconnect:
                raise
            except AppError as exc:
                await websocket.send_json(
                    {"type": "error", "detail": exc.message, "code": exc.code}
                )
            except Exception as exc:
                # Last-resort boundary (mirrors api.errors._handle_unexpected): an
                # unanticipated failure ends this turn, not the whole connection.
                logger.opt(exception=exc).error("Unhandled error on /chat/ws turn")
                await websocket.send_json(
                    {"type": "error", "detail": "Internal server error.", "code": "INTERNAL_ERROR"}
                )
    except WebSocketDisconnect:
        return


@router.get("/conversations", response_model=ConversationPage, summary="List your conversations")
async def list_conversations(
    conversation_service: ConversationServiceDep,
    conversations: ConversationRepositoryDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 10,
) -> ConversationPage:
    """Return the caller's conversations, most-recently-active first (paginated)."""
    items, total = await conversation_service.list_conversations(
        conversations=conversations, limit=page_size, offset=(page - 1) * page_size
    )
    return ConversationPage(
        items=[ConversationPublic.model_validate(c) for c in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessagePage,
    summary="Get a conversation's messages",
)
async def list_messages(
    conversation_id: uuid.UUID,
    conversation_service: ConversationServiceDep,
    conversations: ConversationRepositoryDep,
    messages: MessageRepositoryDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 100,
) -> MessagePage:
    """Return a conversation's messages in chronological order (paginated).

    A 404 means the conversation isn't the caller's or doesn't exist — the two
    are deliberately indistinguishable so existence never leaks.
    """
    items, total = await conversation_service.list_messages(
        conversations=conversations,
        messages=messages,
        conversation_id=conversation_id,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    return MessagePage(
        items=[MessagePublic.model_validate(m) for m in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation and its history",
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user: CurrentUser,
    conversation_service: ConversationServiceDep,
    conversations: ConversationRepositoryDep,
    session: DbSession,
) -> None:
    """Delete a conversation, its messages, and its agent checkpoint state.

    Idempotent from the client's view for a given id only while it exists; a
    second delete returns 404. A 404 means the conversation isn't the caller's
    or doesn't exist — the two are deliberately indistinguishable.
    """
    await conversation_service.delete(
        conversations=conversations, session=session, conversation_id=conversation_id
    )
