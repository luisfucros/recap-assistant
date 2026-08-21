"""Functional HTTP tests for the core app scaffolding (no external infra needed).

The shared ``app``/``client`` fixtures live in ``conftest.py`` (one app per
functional session — see the note there).
"""

import pytest
from api.resources import Resources
from fastapi.testclient import TestClient

pytestmark = pytest.mark.functional


def test_health_returns_ok(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_security_headers_applied(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "strict-transport-security" in response.headers


def test_cors_allows_the_configured_origin_with_credentials(client: TestClient):
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_rejects_an_unlisted_origin(client: TestClient):
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://evil.test",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_request_id_is_generated_and_echoed_when_absent(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.headers["x-request-id"]


def test_request_id_honors_an_incoming_header(client: TestClient):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "caller-supplied-id"})
    assert response.headers["x-request-id"] == "caller-supplied-id"


def test_unknown_route_uses_standard_error_body(client: TestClient):
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"detail", "code"}
    assert body["code"] == "NOT_FOUND"


def test_metrics_endpoint_exposes_prometheus_text(client: TestClient):
    # A prior request generates at least one HTTP metric sample.
    client.get("/api/v1/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # Custom RAG metric is registered even before its first observation.
    assert "recap_operation_seconds" in response.text


def test_lifespan_builds_resources(client: TestClient):
    # The lifespan ran on context enter and stashed the singletons on app.state.
    resources = client.app.state.resources
    assert isinstance(resources, Resources)
    assert resources.engine is not None
    assert resources.redis is not None
    assert resources.qdrant is not None
    assert resources.prompts is not None
    assert resources.tracer is not None
