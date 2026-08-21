"""Cross-cutting core: configuration (Pydantic Settings), structured logging, and error types."""

from shared.core.config import Settings, get_settings
from shared.core.errors import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    ErrorResponse,
    InvalidInputError,
    NotFoundError,
)
from shared.core.logging import configure_logging, redact_text

__all__ = [
    "AppError",
    "AuthenticationError",
    "AuthorizationError",
    "ConflictError",
    "ErrorResponse",
    "InvalidInputError",
    "NotFoundError",
    "Settings",
    "configure_logging",
    "get_settings",
    "redact_text",
]
