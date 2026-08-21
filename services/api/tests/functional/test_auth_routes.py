"""Functional tests for the auth flow over real HTTP.

The DB is the only boundary mocked (the shared in-memory ``user_repo`` fixture
in ``conftest.py``), so these exercise the full request/response cycle — routing,
Pydantic validation, cookie setting, error bodies, and the ``get_current_user``
dependency — without a Postgres container. The crypto (hashing, JWT) is real.
"""

import pytest
from api.deps import get_rate_limit_service
from api.security import ACCESS_COOKIE
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

from shared.core.errors import RateLimitExceededError

pytestmark = pytest.mark.functional


class _AlwaysRateLimited:
    """A rate limiter that always reports the caller as over the limit."""

    async def enforce(self, **_: object) -> None:
        raise RateLimitExceededError()


def _register(client: TestClient, email: str, password: str = "hunter2!") -> None:
    resp = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text


def test_register_then_full_session_lifecycle(client: TestClient, user_repo: FakeUserRepository):
    # Register — returns the user, never the password/hash.
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": "ada@example.com", "password": "hunter2!", "display_name": "Ada"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "ada@example.com"
    assert body["display_name"] == "Ada"
    assert "hashed_password" not in body and "password" not in body

    # Login — sets the httpOnly access cookie.
    resp = client.post(
        "/api/v1/auth/login", json={"email": "ada@example.com", "password": "hunter2!"}
    )
    assert resp.status_code == 200
    assert ACCESS_COOKIE in client.cookies

    # The access cookie reaches a protected endpoint.
    me = client.get("/api/v1/users/me")
    assert me.status_code == 200
    assert me.json()["email"] == "ada@example.com"

    # Refresh rotates the cookies; the new access cookie still works.
    assert client.post("/api/v1/auth/refresh").status_code == 204
    assert client.get("/api/v1/users/me").status_code == 200

    # Logout clears cookies → protected endpoint now rejects.
    assert client.post("/api/v1/auth/logout").status_code == 204
    assert client.get("/api/v1/users/me").status_code == 401


def test_register_duplicate_email_conflicts(client: TestClient, user_repo: FakeUserRepository):
    _register(client, "dup@example.com")
    resp = client.post(
        "/api/v1/auth/register", json={"email": "dup@example.com", "password": "hunter2!"}
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "USER_ALREADY_EXISTS"


def test_login_wrong_password_is_401(client: TestClient, user_repo: FakeUserRepository):
    _register(client, "ada@example.com")
    resp = client.post(
        "/api/v1/auth/login", json={"email": "ada@example.com", "password": "wrong-one"}
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_CREDENTIALS"


def test_login_unknown_email_is_401(client: TestClient, user_repo: FakeUserRepository):
    resp = client.post(
        "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "hunter2!"}
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_CREDENTIALS"


def test_me_requires_authentication(client: TestClient, user_repo: FakeUserRepository):
    resp = client.get("/api/v1/users/me")
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHENTICATED"


def test_refresh_without_cookie_is_401(client: TestClient, user_repo: FakeUserRepository):
    resp = client.post("/api/v1/auth/refresh")
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_TOKEN"


def test_register_rejects_malformed_email(client: TestClient, user_repo: FakeUserRepository):
    resp = client.post(
        "/api/v1/auth/register", json={"email": "not-an-email", "password": "hunter2!"}
    )
    assert resp.status_code == 422


def test_me_reflects_the_cookie_user_not_another(client: TestClient, user_repo: FakeUserRepository):
    # Two users exist; /me always returns the one the current cookie belongs to.
    _register(client, "a@example.com")
    _register(client, "b@example.com")

    client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "hunter2!"})
    assert client.get("/api/v1/users/me").json()["email"] == "a@example.com"

    client.post("/api/v1/auth/login", json={"email": "b@example.com", "password": "hunter2!"})
    assert client.get("/api/v1/users/me").json()["email"] == "b@example.com"


def test_login_returns_429_when_rate_limited(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
):
    # Register while unthrottled, then throttle just the next call (login).
    _register(client, "throttled@example.com")
    app.dependency_overrides[get_rate_limit_service] = lambda: _AlwaysRateLimited()
    try:
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "throttled@example.com", "password": "hunter2!"},
        )
    finally:
        app.dependency_overrides.pop(get_rate_limit_service, None)

    assert resp.status_code == 429
    assert resp.json()["code"] == "RATE_LIMITED"


def test_register_returns_429_when_rate_limited(
    app: FastAPI, client: TestClient, user_repo: FakeUserRepository
):
    app.dependency_overrides[get_rate_limit_service] = lambda: _AlwaysRateLimited()
    try:
        resp = client.post(
            "/api/v1/auth/register",
            json={"email": "new@example.com", "password": "hunter2!"},
        )
    finally:
        app.dependency_overrides.pop(get_rate_limit_service, None)

    assert resp.status_code == 429
    assert resp.json()["code"] == "RATE_LIMITED"
