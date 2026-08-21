"""Auth cookie handling.

Tokens are delivered to the browser as **httpOnly cookies** (never readable by
JavaScript, so an XSS bug can't exfiltrate them) rather than in a response body
for the SPA to store. The access cookie is sent on every request; the refresh
cookie is path-scoped to the auth endpoints so it only travels where it's needed
(login/refresh/logout), shrinking its exposure.

Cookie flags come from settings so the same code is secure in production
(``Secure`` + configurable ``SameSite``) and still works over plain HTTP locally.
"""

from fastapi import Response

from api.services.auth_service import TokenPair
from shared.core.config import Settings

ACCESS_COOKIE = "recap_access_token"
REFRESH_COOKIE = "recap_refresh_token"


def _refresh_cookie_path(settings: Settings) -> str:
    """Path the refresh cookie is scoped to (the auth routes only)."""
    return f"{settings.api_v1_prefix}/auth"


def set_auth_cookies(response: Response, tokens: TokenPair, settings: Settings) -> None:
    """Attach the access + refresh tokens as httpOnly cookies on ``response``.

    Cookie ``max_age`` mirrors each token's TTL so the browser drops the cookie
    at roughly the same time the token stops verifying.
    """
    common = {
        "httponly": True,
        "secure": settings.cookie_secure,
        "samesite": settings.cookie_samesite,
    }
    response.set_cookie(
        ACCESS_COOKIE,
        tokens.access_token,
        max_age=settings.access_token_ttl_minutes * 60,
        path="/",
        **common,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        tokens.refresh_token,
        max_age=settings.refresh_token_ttl_days * 24 * 60 * 60,
        path=_refresh_cookie_path(settings),
        **common,
    )


def clear_auth_cookies(response: Response, settings: Settings) -> None:
    """Delete both auth cookies (logout). Paths must match how they were set."""
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path=_refresh_cookie_path(settings))
