"""Google OAuth (OpenID Connect) client wrapper.

Thin adapter over Authlib's Starlette OAuth client so the router depends on a
small, mockable surface (``authorize_redirect`` + ``fetch_claims``) instead of
Authlib internals. Authlib handles the OIDC discovery, the authorization
redirect, the code→token exchange, and the ``id_token`` signature/nonce
validation; we only pull the verified claims we need.

The CSRF ``state`` and OIDC ``nonce`` are carried in the signed session cookie
(Starlette ``SessionMiddleware``), so the flow is safe without any server-side
state of our own.
"""

from dataclasses import dataclass

from authlib.integrations.starlette_client import OAuth
from fastapi import Request
from starlette.responses import RedirectResponse

from shared.core.config import Settings
from shared.core.errors import AppError

# Google's OIDC discovery document — Authlib reads endpoints + JWKS from here.
_GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"


class OAuthNotConfiguredError(AppError):
    """Google OAuth was requested but its client credentials are not configured."""

    status_code = 404
    code = "OAUTH_NOT_CONFIGURED"
    message = "Google sign-in is not enabled."


class OAuthError(AppError):
    """The OAuth exchange failed (bad state, denied consent, invalid token, …)."""

    status_code = 401
    code = "OAUTH_FAILED"
    message = "Google sign-in failed."


@dataclass(frozen=True, slots=True)
class GoogleClaims:
    """The verified identity claims we consume from Google."""

    sub: str
    email: str
    name: str | None = None


class GoogleOAuthClient:
    """Adapter exposing just the two operations the auth router needs."""

    def __init__(self, settings: Settings) -> None:
        if not settings.google_oauth_configured:
            raise OAuthNotConfiguredError()
        oauth = OAuth()
        oauth.register(
            name="google",
            server_metadata_url=_GOOGLE_METADATA_URL,
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret.get_secret_value(),
            client_kwargs={"scope": "openid email profile"},
        )
        self._google = oauth.google
        self._redirect_uri = settings.google_redirect_uri

    async def authorize_redirect(self, request: Request) -> RedirectResponse:
        """Begin the flow: redirect the browser to Google's consent screen."""
        return await self._google.authorize_redirect(request, self._redirect_uri)

    async def fetch_claims(self, request: Request) -> GoogleClaims:
        """Complete the callback: exchange the code and return verified claims.

        Raises:
            OAuthError: the state/nonce check or token exchange failed, or the
                id token lacked a subject/email.
        """
        try:
            token = await self._google.authorize_access_token(request)
        except Exception as exc:  # authlib raises a variety of OAuth/JOSE errors
            raise OAuthError() from exc
        info = token.get("userinfo") or {}
        sub, email = info.get("sub"), info.get("email")
        if not sub or not email:
            raise OAuthError("Google did not return an email.")
        return GoogleClaims(sub=sub, email=email, name=info.get("name"))
