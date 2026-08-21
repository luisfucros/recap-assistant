"""Unit tests for API error shaping and security headers."""

import pytest
from api.errors import APIError, error_body
from api.middleware import security_headers

pytestmark = pytest.mark.unit


def test_error_body_derives_code_from_status():
    body = error_body(404)
    assert body.code == "NOT_FOUND"
    assert body.detail == "Not Found"


def test_error_body_honors_explicit_detail_and_code():
    body = error_body(409, "dup", "DUPLICATE_DOCUMENT")
    assert body.detail == "dup"
    assert body.code == "DUPLICATE_DOCUMENT"


def test_error_body_unknown_status_falls_back():
    body = error_body(499)
    assert body.code == "ERROR"


def test_api_error_default_code_from_status():
    assert APIError(403).code == "FORBIDDEN"


def test_api_error_explicit_code_wins():
    assert APIError(400, "bad", code="MALFORMED_UPLOAD").code == "MALFORMED_UPLOAD"


def test_security_headers_present():
    headers = security_headers()
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "Strict-Transport-Security" in headers
