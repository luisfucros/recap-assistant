"""Unit tests for the application Settings model."""

import pytest
from pydantic import ValidationError

from shared.core.config import Settings

pytestmark = pytest.mark.unit


def test_defaults_are_local_and_safe():
    settings = Settings(_env_file=None)
    assert settings.environment == "local"
    assert settings.embed_batch_size == 64
    assert settings.qdrant_chunks_collection == "document_chunks"
    assert settings.qdrant_memory_collection == "long_term_memory"
    assert settings.tracing_enabled is False
    assert settings.is_production is False


def test_allowed_origins_parses_comma_separated_value():
    settings = Settings(_env_file=None, backend_cors_origins="http://a.test, http://b.test ,")
    assert settings.allowed_origins == ["http://a.test", "http://b.test"]


def test_tracing_enabled_only_when_all_langfuse_creds_present():
    settings = Settings(
        _env_file=None,
        langfuse_host="http://lf",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )
    assert settings.tracing_enabled is True


def test_production_requires_jwt_secret():
    with pytest.raises(ValidationError, match="jwt_secret"):
        Settings(
            _env_file=None,
            environment="prod",
            backend_cors_origins="https://app.example.com",
        )


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValidationError, match="wildcard"):
        Settings(
            _env_file=None,
            environment="prod",
            jwt_secret="super-secret",
            backend_cors_origins="*",
        )


def test_production_rejects_insecure_cookies():
    with pytest.raises(ValidationError, match="cookie_secure"):
        Settings(
            _env_file=None,
            environment="prod",
            jwt_secret="super-secret",
            backend_cors_origins="https://app.example.com",
            cookie_secure=False,
        )


def test_secrets_do_not_render_in_repr():
    settings = Settings(_env_file=None, jwt_secret="super-secret")
    assert "super-secret" not in repr(settings)
