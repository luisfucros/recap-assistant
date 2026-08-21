"""Authentication routes: register, login, refresh, logout.

Handlers stay thin — they validate input (Pydantic), delegate the real work to
:class:`~api.services.auth_service.AuthService`, translate its domain errors to
HTTP, and manage the httpOnly auth cookies. Tokens are set as cookies (never
returned in the body) so the browser never exposes them to JavaScript.
"""

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import RedirectResponse

from api.deps import (
    AuthRateLimit,
    AuthServiceDep,
    CurrentUser,
    DbSession,
    GoogleOAuthDep,
    SettingsDep,
    UserRepositoryDep,
)
from api.schemas import LoginRequest, RegisterRequest, UserPublic
from api.security import REFRESH_COOKIE, clear_auth_cookies, set_auth_cookies
from api.services.auth_service import (
    DuplicateUserError,
    InvalidCredentialsError,
    InvalidTokenError,
)
from shared.core.errors import AuthenticationError, ConflictError

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new email/password account",
)
async def register(
    payload: RegisterRequest,
    auth: AuthServiceDep,
    users: UserRepositoryDep,
    session: DbSession,
    _rl: AuthRateLimit,
) -> UserPublic:
    """Create an account. Returns the new user; the caller then logs in.

    Raises 409 ``USER_ALREADY_EXISTS`` if the email is taken. Rate-limited per
    client IP (429 ``RATE_LIMITED``).
    """
    try:
        user = await auth.register(
            users,
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
        )
    except DuplicateUserError as exc:
        raise ConflictError(
            "An account with this email already exists.", code="USER_ALREADY_EXISTS"
        ) from exc
    await session.commit()
    return UserPublic.model_validate(user)


@router.post("/login", response_model=UserPublic, summary="Log in and set auth cookies")
async def login(
    payload: LoginRequest,
    response: Response,
    auth: AuthServiceDep,
    users: UserRepositoryDep,
    settings: SettingsDep,
    _rl: AuthRateLimit,
) -> UserPublic:
    """Verify credentials, set httpOnly access/refresh cookies, return the user.

    Raises 401 ``INVALID_CREDENTIALS`` on a bad email/password (same error for
    both, so it never reveals whether an email is registered). Rate-limited per
    client IP (429 ``RATE_LIMITED``) — the primary brute-force guard.
    """
    try:
        user = await auth.authenticate(users, email=payload.email, password=payload.password)
    except InvalidCredentialsError as exc:
        raise AuthenticationError("Invalid email or password.", code="INVALID_CREDENTIALS") from exc
    set_auth_cookies(response, auth.issue_tokens(user.id), settings)
    return UserPublic.model_validate(user)


@router.post("/refresh", status_code=status.HTTP_204_NO_CONTENT, summary="Rotate auth cookies")
async def refresh(
    request: Request,
    response: Response,
    auth: AuthServiceDep,
    users: UserRepositoryDep,
    settings: SettingsDep,
    _rl: AuthRateLimit,
) -> None:
    """Exchange a valid refresh cookie for a fresh access + refresh pair.

    Raises 401 ``INVALID_TOKEN`` when the refresh cookie is missing, expired,
    forged, or for a user that no longer exists. Rate-limited per client IP
    (429 ``RATE_LIMITED``).
    """
    token = request.cookies.get(REFRESH_COOKIE)
    if not token:
        raise AuthenticationError("Not authenticated.", code="INVALID_TOKEN")
    try:
        tokens = await auth.refresh(users, token)
    except InvalidTokenError as exc:
        raise AuthenticationError("Invalid or expired token.", code="INVALID_TOKEN") from exc
    set_auth_cookies(response, tokens, settings)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Clear auth cookies")
async def logout(response: Response, settings: SettingsDep, _user: CurrentUser) -> None:
    """Clear the auth cookies. Requires a valid session so it can't be abused as
    an unauthenticated probe."""
    clear_auth_cookies(response, settings)


@router.get("/google/login", summary="Start Google sign-in")
async def google_login(request: Request, google: GoogleOAuthDep) -> RedirectResponse:
    """Redirect the browser to Google's consent screen (404 if OAuth is off)."""
    return await google.authorize_redirect(request)


@router.get("/google/callback", summary="Complete Google sign-in")
async def google_callback(
    request: Request,
    google: GoogleOAuthDep,
    auth: AuthServiceDep,
    users: UserRepositoryDep,
    session: DbSession,
    settings: SettingsDep,
) -> RedirectResponse:
    """Handle Google's redirect: verify claims, create/link the user, set cookies.

    On success the browser is redirected to the SPA with the auth cookies set;
    the ``user_id`` in those cookies comes from the verified ``id_token``, never
    from the request.
    """
    claims = await google.fetch_claims(request)
    user = await auth.authenticate_google(
        users, google_sub=claims.sub, email=claims.email, display_name=claims.name
    )
    await session.commit()
    response = RedirectResponse(url=settings.frontend_url)
    set_auth_cookies(response, auth.issue_tokens(user.id), settings)
    return response
