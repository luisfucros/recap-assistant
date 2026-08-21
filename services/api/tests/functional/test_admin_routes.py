"""Functional tests for the admin-only user-creation route.

Same DB-boundary-only mocking as ``test_auth_routes.py`` (an in-memory
``FakeUserRepository`` + a no-op session) — the crypto (hashing) is real.
"""

import pytest
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

pytestmark = pytest.mark.functional


def _login_as_admin(
    client: TestClient, user_repo: FakeUserRepository, email: str = "admin@example.com"
) -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text
    user_repo._by_email[email].is_admin = True


def _login_as_reader(client: TestClient, email: str = "reader@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text


def test_admin_can_create_a_regular_user(client: TestClient, user_repo: FakeUserRepository) -> None:
    _login_as_admin(client, user_repo)

    resp = client.post(
        "/api/v1/admin/users",
        json={"email": "newbie@example.com", "password": "hunter2!", "display_name": "Newbie"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["email"] == "newbie@example.com"
    assert body["is_admin"] is False
    assert "hashed_password" not in body and "password" not in body
    assert user_repo._by_email["newbie@example.com"].is_admin is False


def test_admin_can_create_another_admin(client: TestClient, user_repo: FakeUserRepository) -> None:
    _login_as_admin(client, user_repo)

    resp = client.post(
        "/api/v1/admin/users",
        json={"email": "second-admin@example.com", "password": "hunter2!", "is_admin": True},
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["is_admin"] is True
    assert user_repo._by_email["second-admin@example.com"].is_admin is True


def test_create_user_rejects_duplicate_email(
    client: TestClient, user_repo: FakeUserRepository
) -> None:
    _login_as_admin(client, user_repo)
    client.post("/api/v1/admin/users", json={"email": "dupe@example.com", "password": "hunter2!"})

    resp = client.post(
        "/api/v1/admin/users", json={"email": "dupe@example.com", "password": "hunter2!"}
    )

    assert resp.status_code == 409
    assert resp.json()["code"] == "USER_ALREADY_EXISTS"


def test_create_user_requires_admin(client: TestClient, user_repo: FakeUserRepository) -> None:
    _login_as_reader(client)

    resp = client.post(
        "/api/v1/admin/users", json={"email": "blocked@example.com", "password": "hunter2!"}
    )

    assert resp.status_code == 403


def test_create_user_requires_authentication(
    client: TestClient, user_repo: FakeUserRepository
) -> None:
    resp = client.post(
        "/api/v1/admin/users", json={"email": "blocked@example.com", "password": "hunter2!"}
    )

    assert resp.status_code == 401
