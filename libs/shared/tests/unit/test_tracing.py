"""Unit tests for the optional Langfuse tracing scaffold (no network).

The load-bearing guarantees: no credentials ⇒ no-op tracer; tracing-internal
errors are swallowed; but exceptions from the traced block still propagate.
"""

import pytest

from shared.core.config import Settings
from shared.observability.tracing import (
    LangfuseTracer,
    NoOpTracer,
    build_tracer,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- Fakes standing in for the Langfuse client -----------------------------


class _FakeObs:
    def __init__(self) -> None:
        self.updates: dict[str, object] = {}

    def update(self, **fields: object) -> None:
        self.updates.update(fields)


class _RecordingCM:
    def __init__(self, obs: _FakeObs) -> None:
        self.obs = obs
        self.exit_args: tuple | None = None

    def __enter__(self) -> _FakeObs:
        return self.obs

    def __exit__(self, *args: object) -> bool:
        self.exit_args = args
        return False


class _RecordingClient:
    def __init__(self) -> None:
        self.obs = _FakeObs()
        self.cm = _RecordingCM(self.obs)

    def start_as_current_observation(self, **_: object) -> _RecordingCM:
        return self.cm

    def get_current_trace_id(self) -> str:
        return "trace-123"

    def flush(self) -> None:
        pass


class _BadClient:
    def start_as_current_observation(self, **_: object):
        raise RuntimeError("langfuse down")

    def get_current_trace_id(self) -> str:
        raise RuntimeError("langfuse down")

    def flush(self) -> None:
        raise RuntimeError("langfuse down")


# --- Tests -----------------------------------------------------------------


@pytest.mark.unit
def test_no_credentials_gives_noop_tracer() -> None:
    tracer = build_tracer(_settings())
    assert isinstance(tracer, NoOpTracer)
    with tracer.span("retrieval", query="x") as span:
        span.update(output="ignored")  # must not raise


@pytest.mark.unit
def test_credentials_give_langfuse_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid constructing a real client: build_tracer does `from langfuse import Langfuse`.
    monkeypatch.setattr("langfuse.Langfuse", lambda **_: object())
    tracer = build_tracer(
        _settings(
            langfuse_host="http://lf",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
    )
    assert isinstance(tracer, LangfuseTracer)


@pytest.mark.unit
def test_noop_tracer_has_no_trace_id() -> None:
    # With no backend there's no trace to correlate to, so the done event carries None.
    assert NoOpTracer().current_trace_id() is None


@pytest.mark.unit
def test_langfuse_tracer_returns_current_trace_id() -> None:
    tracer = LangfuseTracer(_RecordingClient())
    assert tracer.current_trace_id() == "trace-123"


@pytest.mark.unit
def test_langfuse_trace_id_errors_degrade_to_none() -> None:
    # A client failure must never break the caller — it degrades to no trace id.
    assert LangfuseTracer(_BadClient()).current_trace_id() is None


@pytest.mark.unit
def test_langfuse_span_updates_and_ends() -> None:
    client = _RecordingClient()
    tracer = LangfuseTracer(client)
    with tracer.span("retrieval", foo=1) as span:
        span.update(output={"n": 3})
    assert client.obs.updates == {"output": {"n": 3}}
    assert client.cm.exit_args == (None, None, None)  # clean exit


@pytest.mark.unit
def test_tracing_errors_are_swallowed() -> None:
    tracer = LangfuseTracer(_BadClient())
    # start fails → degrades to a no-op span; update + flush never raise.
    with tracer.span("embedding") as span:
        span.update(output="y")
    tracer.flush()


@pytest.mark.unit
def test_caller_exception_propagates() -> None:
    client = _RecordingClient()
    tracer = LangfuseTracer(client)
    with pytest.raises(ValueError), tracer.span("llm"):
        raise ValueError("boom")
    # The span was closed with the exception info (not swallowed).
    assert client.cm.exit_args is not None
    assert client.cm.exit_args[0] is ValueError
