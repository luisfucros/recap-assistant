"""Structured, PII-redacting logging built on loguru.

Emits one JSON object per log line (easy to ship/parse) and scrubs sensitive
values — emails, JWTs, bearer tokens, API keys, and ``secret=...`` style
key/value pairs — from every rendered message. Redaction runs inside the sink
so it applies uniformly regardless of the call site, and stdlib logging
(uvicorn, celery, sqlalchemy, ...) is intercepted into the same sink so the
whole process logs one consistent, redacted JSON stream. ``diagnose`` is kept
off so loguru never expands local variables into tracebacks (which could leak
secrets/PII), satisfying the "never log tokens/PII/secrets" rule.
"""

import logging
import sys
from collections.abc import Iterable
from datetime import UTC
from re import Match, Pattern, compile
from traceback import format_exception
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

_REDACTED = "[REDACTED]"

# Ordered (pattern, replacement) rules applied to every rendered message.
_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    # JSON Web Tokens (header.payload.signature)
    (compile(r"eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"), _REDACTED),
    # Authorization: Bearer <token>
    (compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), f"Bearer {_REDACTED}"),
    # OpenAI/Anthropic-style API keys (sk-, sk-ant-, sk-proj-)
    (compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{8,}"), _REDACTED),
    # Email addresses
    (compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), _REDACTED),
)

# secret-bearing key/value pairs: keep the key, mask the value.
_KV_PATTERN = compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|authorization)\b(\s*[=:]\s*)(\S+)"
)


def redact_text(text: str) -> str:
    """Return ``text`` with emails, tokens, API keys, and secret values masked."""

    def _mask_value(match: Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}{_REDACTED}"

    redacted = _KV_PATTERN.sub(_mask_value, text)
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _serialize(record: "Record") -> str:
    """Build the redacted JSON payload for one loguru record."""
    import json

    payload: dict[str, Any] = {
        "timestamp": record["time"].astimezone(UTC).isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": redact_text(record["message"]),
    }
    for key, value in record["extra"].items():
        payload[key] = redact_text(value) if isinstance(value, str) else value
    exception = record["exception"]
    if exception is not None:
        trace = "".join(format_exception(exception.type, exception.value, exception.traceback))
        payload["exception"] = redact_text(trace)
    return json.dumps(payload, default=str)


def _json_sink(message: Any) -> None:
    """loguru sink: write the record as a single redacted JSON line to stderr."""
    sys.stderr.write(_serialize(message.record) + "\n")


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru so all output is unified."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str = "INFO", *, quiet_loggers: Iterable[str] = ()) -> None:
    """Install the JSON, PII-redacting loguru sink and intercept stdlib logging.

    Args:
        level: Root log level name (e.g. ``"INFO"``).
        quiet_loggers: Logger names to pin at ``WARNING`` (e.g. noisy access logs).
    """
    logger.remove()
    logger.add(_json_sink, level=level.upper(), backtrace=False, diagnose=False, enqueue=False)
    logging.basicConfig(handlers=[_InterceptHandler()], level=level.upper(), force=True)
    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
