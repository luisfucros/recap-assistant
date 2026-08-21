"""Optional LLM/RAG tracing via Langfuse.

Tracing is **optional and best-effort**, the deep LLM-trace layer on top of the
always-on metrics: with no Langfuse credentials, ``build_tracer`` returns a no-op
tracer and the app behaves identically; even when enabled, any tracing error is
swallowed so a Langfuse hiccup never breaks a user request. Crucially, errors
*from the traced code* still propagate — only tracing-internal failures are
suppressed.

Usage:
    tracer = build_tracer(settings)          # NoOpTracer if creds are absent
    with tracer.span("retrieval", query=q) as span:
        results = do_retrieval()
        span.update(output={"n": len(results)})
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, suppress
from typing import Any, Protocol, runtime_checkable

from shared.core.config import Settings


@runtime_checkable
class Span(Protocol):
    """A handle to the current trace span; ``update`` attaches input/output/metadata."""

    def update(self, **fields: Any) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """Creates spans and flushes buffered traces. Implementations never raise."""

    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Span]: ...

    def current_trace_id(self) -> str | None: ...

    def flush(self) -> None: ...


class _NoOpSpan:
    """A span that records nothing."""

    def update(self, **fields: Any) -> None:
        pass


class NoOpTracer:
    """Tracer used when Langfuse is not configured — every operation is a no-op."""

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[Span]:
        yield _NoOpSpan()

    def current_trace_id(self) -> str | None:
        # No backend, so no trace to correlate to.
        return None

    def flush(self) -> None:
        pass


class _LangfuseSpan:
    """Wraps a Langfuse observation; ``update`` failures are swallowed."""

    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def update(self, **fields: Any) -> None:
        # Best-effort: never let a tracing call break the caller.
        with suppress(Exception):
            self._observation.update(**fields)


class LangfuseTracer:
    """Tracer backed by a Langfuse client (spans nest via the OTel current context)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[Span]:
        # Enter the Langfuse span manually so tracing errors are contained here,
        # while any exception from the *caller's* block still propagates (and is
        # reported to the span). If entering fails, degrade to a no-op span.
        cm: AbstractContextManager[Any] | None = None
        observation: Any = None
        try:
            cm = self._client.start_as_current_observation(
                name=name, as_type="span", metadata=attributes or None
            )
            observation = cm.__enter__()
        except Exception:
            cm = None
            observation = None

        try:
            yield _LangfuseSpan(observation) if observation is not None else _NoOpSpan()
        except BaseException as exc:
            if cm is not None:
                with suppress(Exception):
                    cm.__exit__(type(exc), exc, exc.__traceback__)
                cm = None
            raise
        finally:
            if cm is not None:
                with suppress(Exception):
                    cm.__exit__(None, None, None)

    def current_trace_id(self) -> str | None:
        # Read the active trace id from the current span context, so a running
        # ``span(...)`` block can return it to the caller (e.g. surfaced on the
        # ``done`` event). Best-effort: any client/context error degrades to None.
        with suppress(Exception):
            return self._client.get_current_trace_id()
        return None

    def flush(self) -> None:
        with suppress(Exception):
            self._client.flush()


def build_tracer(settings: Settings) -> Tracer:
    """Return a Langfuse tracer when credentials are set, else a no-op tracer.

    A failure to construct the Langfuse client (bad config, import error) also
    degrades to the no-op tracer — tracing must never be a hard dependency.
    """
    if not settings.tracing_enabled:
        return NoOpTracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),  # type: ignore[union-attr]
            secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
            host=settings.langfuse_host,
        )
        return LangfuseTracer(client)
    except Exception:
        return NoOpTracer()
