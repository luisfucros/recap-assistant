"""Current-user routes.

Read and update the authenticated user's own profile. The user is always
resolved from the access-token cookie via ``get_current_user`` — never from a
path/body id — so a user can only ever read or change their own record.
"""

from fastapi import APIRouter

from api.deps import CurrentUser, DbSession
from api.schemas import UpdateMeRequest, UserPublic

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserPublic, summary="Get the authenticated user")
async def read_me(user: CurrentUser) -> UserPublic:
    """Return the profile of the user identified by the access-token cookie."""
    return UserPublic.model_validate(user)


@router.patch("/me", response_model=UserPublic, summary="Update the authenticated user")
async def update_me(payload: UpdateMeRequest, user: CurrentUser, session: DbSession) -> UserPublic:
    """Update the caller's own profile (display name, preferred language, spoiler-safe).

    Only fields present in the request are changed. ``get_current_user`` and this
    handler share the request's DB session, so mutating the loaded user and
    committing persists the change.
    """
    changes = payload.model_dump(exclude_unset=True)
    if "display_name" in changes:
        user.display_name = changes["display_name"]
    if "preferred_language" in changes:
        user.preferred_language = changes["preferred_language"]
    if "spoiler_safe" in changes:
        user.spoiler_safe = changes["spoiler_safe"]
    await session.commit()
    return UserPublic.model_validate(user)
