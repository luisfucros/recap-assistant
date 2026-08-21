"""Unit tests for the spoiler-safe effective-setting resolver (FR-18)."""

import pytest

from shared.core.spoiler import resolve_spoiler_safe

pytestmark = pytest.mark.unit


def test_per_query_override_wins_over_everything() -> None:
    # Per-query False beats a per-document True and a user default True.
    assert resolve_spoiler_safe(per_query=False, per_document=True, user_default=True) is False
    assert resolve_spoiler_safe(per_query=True, per_document=False, user_default=False) is True


def test_per_document_override_used_when_no_per_query() -> None:
    assert resolve_spoiler_safe(per_query=None, per_document=False, user_default=True) is False
    assert resolve_spoiler_safe(per_query=None, per_document=True, user_default=False) is True


def test_falls_back_to_user_default() -> None:
    assert resolve_spoiler_safe(per_query=None, per_document=None, user_default=True) is True
    assert resolve_spoiler_safe(per_query=None, per_document=None, user_default=False) is False
