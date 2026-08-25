"""Normalize a chat turn's non-text parts to text before the agent reasons (FR-19).

The agent is single-modality: audio and images are turned into text at the front
door so guardrails, retrieval, memory, and the answer model all see text only
(FR-19.2/FR-19.5). This service is that front door — for each media part it

1. **stores the original** in object storage (content-addressed under the owner's
   ``user_id``, so isolation holds and identical bytes dedupe), then
2. **derives text** — a transcript (audio) or a caption/description (image) —
   via the config-selected :class:`~shared.providers.base.Transcriber` /
   :class:`~shared.providers.base.ImageDescriber`.

Storing the original is best-effort: an object-storage hiccup must not fail the
user's turn, since the *derived text* is what the agent reasons over (FR-19.4);
the archival copy is a nice-to-have. Deriving the text is not best-effort — if a
provider genuinely fails, the turn fails honestly rather than silently answering
about content it never understood.

Selection is pure config (``TRANSCRIPTION_PROVIDER`` / ``VISION_PROVIDER``), so the
whole multimodal path runs fully hosted or fully local with no code change.
"""

import uuid
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from shared.ingestion_core.content_address import sha256_hexdigest
from shared.providers.base import ImageDescriber, StorageProvider, Transcriber

# Object-storage prefix for chat attachments, kept distinct from the ``sha256/``
# document namespace so archived turn media is never confused with a library
# document. Still namespaced by ``user_id`` first, upholding per-user isolation.
_MEDIA_PREFIX = "chat-media"

# A minimal mime→extension map for the formats the API accepts (see the schema
# allowlist). Only used to give the stored object a sensible extension; an
# unknown subtype falls back to the raw subtype, never an error.
_EXTENSIONS: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


@dataclass(slots=True)
class MediaPart:
    """One non-text attachment in a chat turn (already decoded to bytes)."""

    kind: Literal["audio", "image"]
    data: bytes
    mime_type: str


@dataclass(slots=True)
class NormalizedPart:
    """A media part after normalization: the derived text and where it was archived."""

    kind: str
    text: str
    # Storage key of the archived original, or ``None`` if the archival write
    # failed (the turn still proceeds on the derived text).
    object_key: str | None


def _media_object_key(user_id: uuid.UUID, content_sha256: str, mime_type: str) -> str:
    """Build the content-addressed storage key for a user's chat attachment."""
    ext = _EXTENSIONS.get(mime_type.lower(), mime_type.lower().split("/")[-1])
    return f"{user_id}/{_MEDIA_PREFIX}/{content_sha256}.{ext}"


class MultimodalNormalizer:
    """Turn a chat turn's audio/image parts into text, archiving the originals."""

    def __init__(
        self,
        *,
        transcriber: Transcriber,
        image_describer: ImageDescriber,
        storage: StorageProvider,
    ) -> None:
        self._transcriber = transcriber
        self._describer = image_describer
        self._storage = storage

    async def normalize(
        self, parts: list[MediaPart], *, user_id: uuid.UUID
    ) -> list[NormalizedPart]:
        """Normalize each part to text, archiving originals under ``user_id``.

        Parts are processed in order so the derived text preserves the order the
        user attached them. ``user_id`` comes from the authenticated context (never
        the request body), so archived media lands only under the caller's prefix.
        """
        return [await self._normalize_one(part, user_id=user_id) for part in parts]

    async def _normalize_one(self, part: MediaPart, *, user_id: uuid.UUID) -> NormalizedPart:
        object_key = await self._archive(part, user_id=user_id)
        text = await self._derive_text(part)
        return NormalizedPart(kind=part.kind, text=text, object_key=object_key)

    async def _archive(self, part: MediaPart, *, user_id: uuid.UUID) -> str | None:
        """Store the original bytes content-addressed; best-effort (never raises)."""
        key = _media_object_key(user_id, sha256_hexdigest(part.data), part.mime_type)
        try:
            await self._storage.put(key, part.data, part.mime_type)
            return key
        except Exception:
            # Losing the archival copy must not fail the turn — the derived text is
            # what the agent reasons over. Log and continue without a key.
            logger.warning("multimodal.archive: failed for {}; continuing on derived text", key)
            return None

    async def _derive_text(self, part: MediaPart) -> str:
        """Transcribe audio / describe an image to the text the agent reasons over.

        Logs only metadata (byte size, mime type, resulting text length) around
        the call — never the audio/image bytes or the derived text itself, which
        can carry the reader's own words or a description of personal content. A
        genuine provider failure propagates rather than being swallowed here (see
        the module docstring); the enclosing graph node's own error log surfaces it.
        """
        if part.kind == "audio":
            logger.info(
                "multimodal: transcription started ({} bytes, {})", len(part.data), part.mime_type
            )
            try:
                text = await self._transcriber.transcribe(part.data, mime_type=part.mime_type)
            except Exception:
                logger.opt(exception=True).error("multimodal: transcription failed")
                raise
            logger.info("multimodal: transcription succeeded ({} chars)", len(text))
            return text
        logger.info(
            "multimodal: image description started ({} bytes, {})", len(part.data), part.mime_type
        )
        try:
            text = await self._describer.describe(part.data, mime_type=part.mime_type)
        except Exception:
            logger.opt(exception=True).error("multimodal: image description failed")
            raise
        logger.info("multimodal: image description succeeded ({} chars)", len(text))
        return text
