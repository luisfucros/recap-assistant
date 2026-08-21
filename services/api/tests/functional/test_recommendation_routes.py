"""Functional tests for the recommendations route (FR-5).

The recommendation service is faked at the boundary — only the route's HTTP
contract (auth, request/response shape) is under test here; ranking/
explanation logic is covered in `test_recommendation_service.py`.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from api.deps import (
    CurrentUser,
    get_document_repository,
    get_memory_repository,
    get_memory_service,
    get_progress_repository,
    get_progress_service,
    get_recommendation_service,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.functional.conftest import FakeUserRepository

pytestmark = pytest.mark.functional

# The route also injects these as pass-through arguments to the (faked)
# recommendation service; each pulls in real DB/embedder access otherwise, so
# they're stubbed out too rather than only the top-level service.
_PASSTHROUGH_DEPS = (
    get_document_repository,
    get_progress_repository,
    get_progress_service,
    get_memory_repository,
    get_memory_service,
)


class _FakeRecommendationService:
    def __init__(self, items: list[Any] | None = None) -> None:
        self._items = items or []
        self.calls: list[dict[str, Any]] = []

    async def recommend_from_library(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return list(self._items)


@pytest.fixture
def recommendation_service(app: FastAPI) -> Iterator[_FakeRecommendationService]:
    """Override the recommendation service boundary with an in-memory fake."""
    service = _FakeRecommendationService()

    def _service(_user: CurrentUser) -> _FakeRecommendationService:
        return service

    for dep in _PASSTHROUGH_DEPS:
        app.dependency_overrides[dep] = lambda: None
    app.dependency_overrides[get_recommendation_service] = _service
    try:
        yield service
    finally:
        for dep in (*_PASSTHROUGH_DEPS, get_recommendation_service):
            app.dependency_overrides.pop(dep, None)


def _login(client: TestClient, email: str = "reader@example.com") -> None:
    client.post("/api/v1/auth/register", json={"email": email, "password": "hunter2!"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "hunter2!"})
    assert resp.status_code == 200, resp.text


def _recommendation(**overrides: Any) -> SimpleNamespace:
    base = {
        "title": "The Iliad",
        "reason": "Because you completed The Odyssey",
        "document_id": None,
        "author": "Homer",
        "url": None,
        "score": 0.8,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_get_recommendations_returns_explainable_items(
    client: TestClient,
    user_repo: FakeUserRepository,
    recommendation_service: _FakeRecommendationService,
) -> None:
    _login(client)
    recommendation_service._items = [_recommendation()]

    resp = client.get("/api/v1/recommendations")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"] == [
        {
            "title": "The Iliad",
            "reason": "Because you completed The Odyssey",
            "document_id": None,
            "author": "Homer",
            "url": None,
            "score": 0.8,
        }
    ]


def test_get_recommendations_empty_list_when_nothing_to_recommend(
    client: TestClient,
    user_repo: FakeUserRepository,
    recommendation_service: _FakeRecommendationService,
) -> None:
    _login(client)

    resp = client.get("/api/v1/recommendations")

    assert resp.status_code == 200, resp.text
    assert resp.json()["items"] == []


def test_get_recommendations_respects_limit_query_param(
    client: TestClient,
    user_repo: FakeUserRepository,
    recommendation_service: _FakeRecommendationService,
) -> None:
    _login(client)

    resp = client.get("/api/v1/recommendations", params={"limit": 3})

    assert resp.status_code == 200, resp.text
    assert recommendation_service.calls[0]["limit"] == 3


def test_get_recommendations_rejects_limit_out_of_range(
    client: TestClient,
    user_repo: FakeUserRepository,
    recommendation_service: _FakeRecommendationService,
) -> None:
    _login(client)

    resp = client.get("/api/v1/recommendations", params={"limit": 50})

    assert resp.status_code == 422


def test_get_recommendations_requires_authentication(
    client: TestClient,
    user_repo: FakeUserRepository,
    recommendation_service: _FakeRecommendationService,
) -> None:
    resp = client.get("/api/v1/recommendations")
    assert resp.status_code == 401
