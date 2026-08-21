"""Unit tests for the API startup warm-up helpers (no infra, no real model).

Covers the defensive contract: warm-up force-builds the declared heavy resources
and probes connections, and every step swallows failures (logged, retried lazily)
so a warm-up can never take the app down at boot.
"""

import asyncio
from types import SimpleNamespace

import pytest
from api.lifespan import _heavy_resource_names, _probe, _warm_up_heavy_constructions
from api.resources import Resources

pytestmark = pytest.mark.unit


def test_agent_service_is_warmed_up_at_startup() -> None:
    # The agent runner is force-built at boot (not on first /chat), so the chat
    # model wiring is ready up front and a bad LLM key surfaces in the startup log.
    assert "agent_service" in Resources.HEAVY_RESOURCES


def test_agent_service_is_declared_loop_bound() -> None:
    # It builds an AsyncPostgresSaver, which captures the running loop at
    # construction — so it must be warmed on the loop, not in a worker thread.
    assert "agent_service" in Resources.LOOP_BOUND_RESOURCES


class _ProviderResources:
    """A resources double carrying just a transcription-provider setting."""

    HEAVY_RESOURCES = ("embedder",)

    def __init__(self, provider: str) -> None:
        self.settings = SimpleNamespace(transcription_provider=provider)


def test_local_transcriber_is_warmed_only_for_huggingface() -> None:
    # The local HF transcriber loads its Whisper model at construction, so it's
    # warmed at startup only when selected; hosted transcription stays lazy.
    assert "transcriber" in _heavy_resource_names(_ProviderResources("huggingface"))
    assert "transcriber" not in _heavy_resource_names(_ProviderResources("openai"))


def test_heavy_resource_names_default_when_settings_absent() -> None:
    # A lightweight double without `settings` just gets the base heavy set.
    assert _heavy_resource_names(_FakeResources()) == _FakeResources.HEAVY_RESOURCES


class _FakeResources:
    """Records which heavy resources were built and can fail on demand."""

    HEAVY_RESOURCES = ("embedder",)

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.accessed: list[str] = []

    @property
    def embedder(self) -> str:
        self.accessed.append("embedder")
        if self._fail:
            raise RuntimeError("model load failed")
        return "warm"


async def test_warm_up_constructions_builds_declared_heavy_resources() -> None:
    fake = _FakeResources()
    await _warm_up_heavy_constructions(fake)
    assert fake.accessed == list(_FakeResources.HEAVY_RESOURCES)


async def test_warm_up_constructions_swallows_build_errors() -> None:
    fake = _FakeResources(fail=True)
    # Must not raise — a failed build is logged and retried on the first request.
    await _warm_up_heavy_constructions(fake)
    assert fake.accessed == ["embedder"]


class _LoopBoundResources:
    """A heavy resource whose constructor captures the running loop — like the
    ``AsyncPostgresSaver`` the agent service builds."""

    HEAVY_RESOURCES = ("saver",)
    LOOP_BOUND_RESOURCES = frozenset({"saver"})

    def __init__(self) -> None:
        self.built_on_loop = False

    @property
    def saver(self) -> str:
        asyncio.get_running_loop()  # raises RuntimeError if built in a worker thread
        self.built_on_loop = True
        return "saver"


async def test_loop_bound_resource_is_built_on_the_event_loop() -> None:
    # Regression: a loop-capturing constructor must be warmed on the event loop,
    # not via asyncio.to_thread (a worker thread has no running loop → it raised).
    fake = _LoopBoundResources()
    await _warm_up_heavy_constructions(fake)
    assert fake.built_on_loop is True


class _MisdeclaredResources(_LoopBoundResources):
    """The same loop-capturing resource, but NOT declared loop-bound — reproduces
    the original bug: warm-up offloads it to a thread, where it raises."""

    LOOP_BOUND_RESOURCES = frozenset()


async def test_loop_capturing_resource_off_loop_is_swallowed() -> None:
    fake = _MisdeclaredResources()
    # Off-loop it hits 'no running event loop'; warm-up swallows it (best-effort),
    # so boot survives — but it never finished building (proves it needs the loop).
    await _warm_up_heavy_constructions(fake)
    assert fake.built_on_loop is False


async def test_probe_awaits_success_without_raising() -> None:
    calls: list[str] = []

    async def ok() -> None:
        calls.append("ran")

    await _probe("postgres", ok())
    assert calls == ["ran"]


async def test_probe_swallows_connection_errors() -> None:
    async def boom() -> None:
        raise ConnectionError("infra down")

    # A probe against unreachable infra must not raise — the app still boots.
    await _probe("qdrant", boom())
