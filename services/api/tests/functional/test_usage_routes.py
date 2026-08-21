"""Functional tests for the usage routes (NFR-13).

``UsageService`` is faked at the boundary — only the routes' HTTP contract
(auth, self-vs-admin authorization, request/response shape) is under test
here; the real aggregation is covered in ``test_usage_service.py`` (unit) and
``test_usage_service_pg.py`` (integration).
"""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from api.deps import get_usage_service
from api.schemas import UsageSummary
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

pytestmark = pytest.mark.functional


def _summary(**overrides: Any) -> UsageSummary:
    base = {
        "window_days": 30,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "tool_calls": 2,
        "tool_calls_by_tool": [{"tool": "retrieve_chunks", "count": 2}],
    }
    base.update(overrides)
    return UsageSummary(**base)


class _FakeUsageService:
    def __init__(self, summary: UsageSummary | None = None) -> None:
        self._summary = summary or _summary()
        self.calls: list[dict[str, Any]] = []

    async def get_usage(self, **kwargs: Any) -> UsageSummary:
        self.calls.append(kwargs)
        return self._summary


@pytest.fixture
def usage_service(app: FastAPI) -> Iterator[_FakeUsageService]:
    """Override the usage-aggregation boundary with an in-memory fake."""
    service = _FakeUsageService()
    app.dependency_overrides[get_usage_service] = lambda: service
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_usage_service, None)


def _login(client: TestClient, email: str = "reader@example.com") -> uuid.UUID:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text
    return uuid.UUID(client.get("/api/v1/users/me").json()["id"])


def _login_as_admin(
    client: TestClient, user_repo: FakeUserRepository, email: str = "admin@example.com"
) -> uuid.UUID:
    user_id = _login(client, email)
    user_repo._by_email[email].is_admin = True
    return user_id


# --- GET /usage ---------------------------------------------------------------- #


def test_get_usage_requires_authentication(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    resp = client.get("/api/v1/usage")
    assert resp.status_code == 401


def test_get_usage_returns_the_callers_summary(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    user_id = _login(client)

    resp = client.get("/api/v1/usage")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_tokens"] == 120
    assert body["tool_calls_by_tool"] == [{"tool": "retrieve_chunks", "count": 2}]
    assert usage_service.calls[0]["user_id"] == user_id


def test_get_usage_respects_window_days(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    _login(client)

    resp = client.get("/api/v1/usage", params={"window_days": 7})

    assert resp.status_code == 200, resp.text
    assert usage_service.calls[0]["window_days"] == 7


def test_get_usage_rejects_window_days_out_of_range(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    _login(client)

    resp = client.get("/api/v1/usage", params={"window_days": 0})

    assert resp.status_code == 422


# --- GET /usage/{user_id} ------------------------------------------------------- #


def test_get_user_usage_requires_authentication(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    resp = client.get(f"/api/v1/usage/{uuid.uuid4()}")
    assert resp.status_code == 401


def test_get_user_usage_self_is_allowed(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    user_id = _login(client)

    resp = client.get(f"/api/v1/usage/{user_id}")

    assert resp.status_code == 200, resp.text
    assert usage_service.calls[0]["user_id"] == user_id


def test_get_user_usage_another_user_is_forbidden(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    _login(client, "reader-a@example.com")
    other_user_id = uuid.uuid4()

    resp = client.get(f"/api/v1/usage/{other_user_id}")

    assert resp.status_code == 403
    assert usage_service.calls == []


def test_get_user_usage_admin_can_view_any_user(
    client: TestClient, user_repo: FakeUserRepository, usage_service: _FakeUsageService
) -> None:
    _login_as_admin(client, user_repo)
    other_user_id = uuid.uuid4()

    resp = client.get(f"/api/v1/usage/{other_user_id}")

    assert resp.status_code == 200, resp.text
    assert usage_service.calls[0]["user_id"] == other_user_id
