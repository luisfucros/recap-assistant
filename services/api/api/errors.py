"""FastAPI error handling that renders every error as ``{detail, code}``.

Errors are raised as FastAPI ``HTTPException``s — use ``APIError`` to attach a
stable machine ``code``. Everything funnels through a single handler on
``HTTPException`` that serializes the standard ``{detail, code}`` body, so it
also covers FastAPI's own built-in errors (404/405/...). The framework-agnostic
``AppError`` raised by the service/ingestion layer is converted to an
``APIError`` and flows through the same path; unhandled exceptions are logged
and returned as a generic 500 so internal details never leak.
"""

from http import HTTPStatus

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from shared.core.errors import AppError, ErrorResponse

# Default SNAKE_CASE codes for statuses raised without an explicit code.
_CODE_BY_STATUS: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
}


class APIError(HTTPException):
    """An ``HTTPException`` that also carries a stable machine-readable ``code``."""

    def __init__(
        self,
        status_code: int,
        detail: str | None = None,
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code or _CODE_BY_STATUS.get(status_code, "ERROR")


def _reason(status_code: int) -> str:
    """Human-readable default detail for a status code (e.g. 404 -> 'Not Found')."""
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


def error_body(status_code: int, detail: object = None, code: str | None = None) -> ErrorResponse:
    """Build the standard ``{detail, code}`` body, filling sensible defaults."""
    resolved_code = code or _CODE_BY_STATUS.get(status_code, "ERROR")
    resolved_detail = str(detail) if detail else _reason(status_code)
    return ErrorResponse(detail=resolved_detail, code=resolved_code)


async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """The one place an error becomes a response body — all handlers route here."""
    body = error_body(exc.status_code, exc.detail, getattr(exc, "code", None))
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(),
        headers=getattr(exc, "headers", None),
    )


async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return await _handle_http_exception(
        request, APIError(exc.status_code, exc.message, code=exc.code)
    )


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    api_error = APIError(422, "Request validation failed.", code="VALIDATION_ERROR")
    return await _handle_http_exception(request, api_error)


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.opt(exception=exc).error("Unhandled error on {} {}", request.method, request.url.path)
    api_error = APIError(500, "Internal server error.", code="INTERNAL_ERROR")
    return await _handle_http_exception(request, api_error)


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers so every error is rendered as a ``{detail, code}`` JSON body."""
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected)
