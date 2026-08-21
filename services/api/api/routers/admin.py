"""Admin-only routes: directly provision user accounts (FR unspecified — an
operational need, not a reader-facing feature).

Unlike public self-registration, this lets an already-authenticated admin
create any account — including another admin — without that person going
through the sign-up flow themselves.
"""

from fastapi import APIRouter, status

from api.deps import AdminUser, AuthServiceDep, DbSession, UserRepositoryDep
from api.schemas import AdminCreateUserRequest, UserPublic
from api.services.auth_service import DuplicateUserError
from shared.core.errors import ConflictError

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/users",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user account (admin only)",
)
async def create_user(
    payload: AdminCreateUserRequest,
    _admin: AdminUser,
    auth: AuthServiceDep,
    users: UserRepositoryDep,
    session: DbSession,
) -> UserPublic:
    """Create a regular or admin account directly, bypassing self-registration.

    Raises 409 ``USER_ALREADY_EXISTS`` if the email is taken, 403 for a caller
    that isn't an admin, 401 if unauthenticated.
    """
    try:
        user = await auth.register(
            users,
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
            is_admin=payload.is_admin,
        )
    except DuplicateUserError as exc:
        raise ConflictError(
            "An account with this email already exists.", code="USER_ALREADY_EXISTS"
        ) from exc
    await session.commit()
    return UserPublic.model_validate(user)
