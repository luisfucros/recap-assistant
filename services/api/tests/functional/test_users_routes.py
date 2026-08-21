"""Functional tests for the current-user profile routes (`/users/me`)."""

import pytest
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

pytestmark = pytest.mark.functional


def _login(client: TestClient, email: str = "ada@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200


def test_me_defaults_to_english(client: TestClient, user_repo: FakeUserRepository):
    _login(client)
    body = client.get("/api/v1/users/me").json()
    assert body["preferred_language"] == "en"


def test_patch_updates_language_and_display_name(client: TestClient, user_repo: FakeUserRepository):
    _login(client)

    resp = client.patch(
        "/api/v1/users/me", json={"preferred_language": "es", "display_name": "Ada L."}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["preferred_language"] == "es"
    assert body["display_name"] == "Ada L."

    # Persisted: a fresh read reflects the change.
    assert client.get("/api/v1/users/me").json()["preferred_language"] == "es"


def test_patch_is_partial(client: TestClient, user_repo: FakeUserRepository):
    _login(client)
    client.patch("/api/v1/users/me", json={"display_name": "Only Name"})
    body = client.get("/api/v1/users/me").json()
    assert body["display_name"] == "Only Name"
    assert body["preferred_language"] == "en"  # untouched


def test_me_defaults_to_spoiler_safe_on(client: TestClient, user_repo: FakeUserRepository):
    _login(client)
    body = client.get("/api/v1/users/me").json()
    assert body["spoiler_safe"] is True  # SPOILER_SAFE_DEFAULT


def test_patch_toggles_spoiler_safe(client: TestClient, user_repo: FakeUserRepository):
    _login(client)
    resp = client.patch("/api/v1/users/me", json={"spoiler_safe": False})
    assert resp.status_code == 200
    assert resp.json()["spoiler_safe"] is False
    # Persisted, and unrelated fields are untouched.
    body = client.get("/api/v1/users/me").json()
    assert body["spoiler_safe"] is False
    assert body["preferred_language"] == "en"


def test_patch_rejects_unsupported_language(client: TestClient, user_repo: FakeUserRepository):
    _login(client)
    resp = client.patch("/api/v1/users/me", json={"preferred_language": "pt"})
    assert resp.status_code == 422


def test_patch_requires_authentication(client: TestClient, user_repo: FakeUserRepository):
    resp = client.patch("/api/v1/users/me", json={"preferred_language": "de"})
    assert resp.status_code == 401
