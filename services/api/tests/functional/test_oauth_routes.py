"""Functional tests for the Google OAuth routes.

Google itself is mocked at the ``GoogleOAuthClient`` boundary (no network, no
real token exchange); the DB is the in-memory ``user_repo`` fixture. What's
exercised for real: routing, the create/link user logic, cookie-setting on the
post-login redirect, and the not-configured / failure error paths.
"""

from collections.abc import Callable, Iterator

import pytest
from api.deps import get_google_oauth
from api.oauth import GoogleClaims, OAuthError
from api.security import ACCESS_COOKIE
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

pytestmark = pytest.mark.functional


class _FakeGoogleOAuth:
    """Stands in for the real Authlib-backed client at the router boundary."""

    def __init__(self, *, claims: GoogleClaims | None = None, fail: bool = False) -> None:
        self._claims = claims
        self._fail = fail

    async def authorize_redirect(self, request: Request) -> RedirectResponse:
        return RedirectResponse(url="https://accounts.google.com/o/oauth2/v2/auth?fake=1")

    async def fetch_claims(self, request: Request) -> GoogleClaims:
        if self._fail or self._claims is None:
            raise OAuthError()
        return self._claims


@pytest.fixture
def use_google(app: FastAPI) -> Iterator[Callable[[_FakeGoogleOAuth], None]]:
    """Install a fake Google client for a test; remove it afterwards."""

    def _install(fake: _FakeGoogleOAuth) -> None:
        app.dependency_overrides[get_google_oauth] = lambda: fake

    try:
        yield _install
    finally:
        app.dependency_overrides.pop(get_google_oauth, None)


def test_login_redirects_to_google(
    client: TestClient, use_google: Callable[[_FakeGoogleOAuth], None]
):
    use_google(_FakeGoogleOAuth())
    resp = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "accounts.google.com" in resp.headers["location"]


def test_callback_creates_user_sets_cookies_and_redirects(
    client: TestClient,
    user_repo: FakeUserRepository,
    use_google: Callable[[_FakeGoogleOAuth], None],
):
    use_google(
        _FakeGoogleOAuth(claims=GoogleClaims(sub="g-42", email="ada@example.com", name="Ada"))
    )

    resp = client.get("/api/v1/auth/google/callback?code=abc&state=xyz", follow_redirects=False)

    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "http://localhost:5173"  # the SPA
    assert ACCESS_COOKIE in resp.cookies
    # The user was provisioned and the access cookie now reaches a protected route.
    assert await_user(user_repo, "g-42")
    assert client.get("/api/v1/users/me").json()["email"] == "ada@example.com"


def test_callback_failure_is_401(
    client: TestClient,
    user_repo: FakeUserRepository,
    use_google: Callable[[_FakeGoogleOAuth], None],
):
    use_google(_FakeGoogleOAuth(fail=True))
    resp = client.get("/api/v1/auth/google/callback?code=bad", follow_redirects=False)
    assert resp.status_code == 401
    assert resp.json()["code"] == "OAUTH_FAILED"


def test_login_404_when_oauth_not_configured(client: TestClient, user_repo: FakeUserRepository):
    # No override → the real (unconfigured) client is built → clean 404.
    resp = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert resp.status_code == 404
    assert resp.json()["code"] == "OAUTH_NOT_CONFIGURED"


def await_user(repo: FakeUserRepository, google_sub: str) -> bool:
    """Sync helper: was a user with this Google id provisioned?"""
    return google_sub in repo._by_google_sub
