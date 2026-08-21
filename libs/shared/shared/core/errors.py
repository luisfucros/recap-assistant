"""Domain error types and the standard API error body.

These exceptions are framework-agnostic — they carry an HTTP status and a stable
machine-readable ``code`` but know nothing about FastAPI. The API service
translates them into ``{detail, code}`` JSON responses (see the service's
exception handlers); the ingestion service can raise/catch the same types
without importing a web framework. Raising a specific subclass is how a service
signals an outcome; the transport layer decides how to render it.
"""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard API error body: a human-readable detail and a stable machine code."""

    detail: str
    code: str


class AppError(Exception):
    """Base domain error carrying an HTTP status and a stable error code.

    Subclasses set class-level ``status_code``/``code``/``message`` defaults;
    any of them can be overridden per-instance via the constructor.
    """

    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or type(self).message
        self.code = code or type(self).code
        self.status_code = status_code or type(self).status_code
        super().__init__(self.message)


class NotFoundError(AppError):
    """A requested resource does not exist (or is not visible to the caller)."""

    status_code = 404
    code = "NOT_FOUND"
    message = "The requested resource was not found."


class ConflictError(AppError):
    """The request conflicts with existing state (e.g. a uniqueness violation)."""

    status_code = 409
    code = "CONFLICT"
    message = "The request conflicts with existing state."


class InvalidInputError(AppError):
    """The request input is semantically invalid beyond schema validation."""

    status_code = 422
    code = "INVALID_INPUT"
    message = "The request input is invalid."


class AuthenticationError(AppError):
    """The caller is not authenticated."""

    status_code = 401
    code = "UNAUTHENTICATED"
    message = "Authentication is required."


class AuthorizationError(AppError):
    """The caller is authenticated but not allowed to access the resource."""

    status_code = 403
    code = "FORBIDDEN"
    message = "You do not have access to this resource."


class RateLimitExceededError(AppError):
    """The caller exceeded the allowed request rate for this endpoint."""

    status_code = 429
    code = "RATE_LIMITED"
    message = "Too many requests. Please try again later."
