"""Unit tests for the ingestion worker lifecycle hooks (no broker, no real model).

Focuses on the warm-up hook's contract: when enabled it force-builds the heavy
resources at process start and must never raise — a failed build is logged and
retried lazily on the first task, so a missing key / missing extra can't crash
the worker — and when ``warm_up_on_start`` is off it does nothing.
"""

from types import SimpleNamespace

import ingestion.bootstrap as bootstrap
import pytest

pytestmark = pytest.mark.unit


class _FakeResources:
    """Stand-in resources that records attribute access and can fail on demand."""

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


@pytest.fixture
def _enable_warm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "get_settings", lambda: SimpleNamespace(warm_up_on_start=True))


def test_warm_up_force_builds_every_heavy_resource(
    _enable_warm_up: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeResources()
    monkeypatch.setattr(bootstrap, "get_ingestion_resources", lambda: fake)

    bootstrap._warm_heavy_resources()

    assert fake.accessed == list(_FakeResources.HEAVY_RESOURCES)


def test_warm_up_swallows_construction_errors(
    _enable_warm_up: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeResources(fail=True)
    monkeypatch.setattr(bootstrap, "get_ingestion_resources", lambda: fake)

    # Must not raise: a genuine config error surfaces on the first task instead.
    bootstrap._warm_heavy_resources()

    assert fake.accessed == ["embedder"]


def test_warm_up_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "get_settings", lambda: SimpleNamespace(warm_up_on_start=False))
    fake = _FakeResources()
    monkeypatch.setattr(bootstrap, "get_ingestion_resources", lambda: fake)

    bootstrap._warm_heavy_resources()

    assert fake.accessed == []
