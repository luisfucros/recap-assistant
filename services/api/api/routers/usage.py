"""Usage routes: per-user token spend and tool-call counts (NFR-13).

``GET /usage`` mirrors ``/analytics``: the caller's own usage, scoped by the
access-token cookie. ``GET /usage/{user_id}`` additionally serves the operator
story NFR-13 describes ("token spend per user") — a caller may look up their
own usage this way too, but looking up *another* user's requires ``is_admin``.
"""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Query

from api.deps import CurrentUser, DbSession, UsageEventRepositoryDep, UsageServiceDep
from api.schemas import UsageSummary
from shared.core.errors import AuthorizationError
from shared.repositories import UsageEventRepository

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("", response_model=UsageSummary, summary="Get your token-spend and tool-call usage")
async def get_usage(
    user: CurrentUser,
    usage_service: UsageServiceDep,
    usage: UsageEventRepositoryDep,
    window_days: int = Query(30, ge=1, le=365, description="Trailing window in days."),
) -> UsageSummary:
    """Return the caller's LLM token spend and tool-call counts.

    Computed over the trailing ``window_days`` from the append-only usage-event
    trail, and served from a short-lived per-user cache.
    """
    return await usage_service.get_usage(
        user_id=user.id, usage=usage, today=datetime.now(UTC).date(), window_days=window_days
    )


@router.get(
    "/{user_id}",
    response_model=UsageSummary,
    summary="Get a user's token-spend and tool-call usage (self, or any user if admin)",
)
async def get_user_usage(
    user_id: uuid.UUID,
    caller: CurrentUser,
    usage_service: UsageServiceDep,
    session: DbSession,
    window_days: int = Query(30, ge=1, le=365, description="Trailing window in days."),
) -> UsageSummary:
    """Return ``user_id``'s usage — the operator story behind NFR-13.

    Raises 403 ``FORBIDDEN`` unless the caller is looking up their own usage
    or is an administrator; per-user spend is exactly the kind of data a
    non-admin caller must never be able to read for anyone but themselves.
    """
    if caller.id != user_id and not caller.is_admin:
        raise AuthorizationError("You can only view your own usage.")
    usage = UsageEventRepository(session, user_id)
    return await usage_service.get_usage(
        user_id=user_id, usage=usage, today=datetime.now(UTC).date(), window_days=window_days
    )
