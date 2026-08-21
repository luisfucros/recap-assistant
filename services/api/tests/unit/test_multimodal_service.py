"""Unit tests for :class:`MultimodalNormalizer` (FR-19 Phase B).

The transcriber/describer/storage are boundary fakes — no network, no ML stack —
so these assert the normalizer's own behavior: each part is archived under the
owner's content-addressed key, audio routes to the transcriber and images to the
describer, order is preserved, and an archival failure degrades to "no key" while
the derived text still comes back (so the turn survives a storage hiccup).
"""

import uuid
from typing import Any

import pytest
from api.services.multimodal_service import MediaPart, MultimodalNormalizer

pytestmark = pytest.mark.unit

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _FakeStorage:
    """Records puts; can be told to fail (to prove best-effort archival)."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.puts: list[tuple[str, bytes, str]] = []

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        if self._fail:
            raise RuntimeError("storage down")
        self.puts.append((key, data, content_type))

    async def get(self, key: str) -> bytes:  # pragma: no cover - unused here
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


class _FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, audio: bytes, *, mime_type: str) -> str:
        self.calls.append((audio, mime_type))
        return "transcribed speech"


class _FakeDescriber:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    async def describe(self, image: bytes, *, mime_type: str) -> str:
        self.calls.append((image, mime_type))
        return "a described image"


def _normalizer(**overrides: Any) -> tuple[MultimodalNormalizer, dict]:
    parts = {
        "transcriber": _FakeTranscriber(),
        "image_describer": _FakeDescriber(),
        "storage": _FakeStorage(),
    }
    parts.update(overrides)
    return MultimodalNormalizer(**parts), parts


async def test_audio_is_transcribed_and_archived_under_the_user() -> None:
    normalizer, deps = _normalizer()
    out = await normalizer.normalize(
        [MediaPart(kind="audio", data=b"RIFF....", mime_type="audio/wav")], user_id=USER_ID
    )

    assert out[0].kind == "audio"
    assert out[0].text == "transcribed speech"
    # Archived content-addressed under the owner's chat-media prefix.
    key = out[0].object_key
    assert key is not None and key.startswith(f"{USER_ID}/chat-media/") and key.endswith(".wav")
    assert deps["storage"].puts and deps["storage"].puts[0][0] == key
    assert deps["transcriber"].calls[0][1] == "audio/wav"


async def test_image_routes_to_the_describer() -> None:
    normalizer, deps = _normalizer()
    out = await normalizer.normalize(
        [MediaPart(kind="image", data=b"\x89PNG", mime_type="image/png")], user_id=USER_ID
    )

    assert out[0].text == "a described image"
    assert out[0].object_key.endswith(".png")
    assert deps["image_describer"].calls  # image path hit
    assert not deps["transcriber"].calls  # audio path untouched


async def test_parts_preserve_order() -> None:
    normalizer, _ = _normalizer()
    out = await normalizer.normalize(
        [
            MediaPart(kind="image", data=b"a", mime_type="image/png"),
            MediaPart(kind="audio", data=b"b", mime_type="audio/wav"),
        ],
        user_id=USER_ID,
    )
    assert [p.kind for p in out] == ["image", "audio"]


async def test_archival_failure_still_returns_derived_text() -> None:
    # A storage hiccup must not fail the turn: the derived text is what matters,
    # the archived original is best-effort (object_key comes back None).
    normalizer, _ = _normalizer(storage=_FakeStorage(fail=True))
    out = await normalizer.normalize(
        [MediaPart(kind="audio", data=b"x", mime_type="audio/wav")], user_id=USER_ID
    )
    assert out[0].text == "transcribed speech"
    assert out[0].object_key is None


async def test_identical_bytes_hash_to_the_same_key() -> None:
    normalizer, _ = _normalizer()
    first = await normalizer.normalize(
        [MediaPart(kind="audio", data=b"same", mime_type="audio/wav")], user_id=USER_ID
    )
    second = await normalizer.normalize(
        [MediaPart(kind="audio", data=b"same", mime_type="audio/wav")], user_id=USER_ID
    )
    assert first[0].object_key == second[0].object_key
