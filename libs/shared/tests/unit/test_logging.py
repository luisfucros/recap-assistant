"""Unit tests for PII-redacting structured logging."""

import json

import pytest
from loguru import logger

from shared.core.logging import configure_logging, redact_text

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "raw",
    [
        "user email bob@example.com signed in",
        "header Authorization: Bearer abc.def-ghi_123",
        "anthropic key sk-ant-0123456789abcdef",
        "openai key sk-proj-ABCDEFGH1234",
        "password=hunter2 in config",
        "api_key: 9f8e7d6c5b4a",
    ],
)
def test_redacts_sensitive_values(raw):
    assert "[REDACTED]" in redact_text(raw)


def test_redacts_jwt_completely():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4"
    assert jwt not in redact_text(f"session token {jwt}")


def test_preserves_ordinary_text():
    assert redact_text("chapter 3 summary generated") == "chapter 3 summary generated"


def test_configured_logger_emits_redacted_json(capsys):
    configure_logging("INFO")
    logger.bind(user="carol@example.com").info("login from {}", "alice@example.com")
    captured = capsys.readouterr().err
    line = next(item for item in captured.splitlines() if item.strip().startswith("{"))
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert "alice@example.com" not in payload["message"]
    assert "[REDACTED]" in payload["message"]
    # bound extras are redacted too
    assert "carol@example.com" not in json.dumps(payload)
