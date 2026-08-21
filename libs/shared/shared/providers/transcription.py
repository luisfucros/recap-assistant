"""Speech-to-text providers (FR-19): hosted OpenAI Whisper, or local HuggingFace.

The agent's ``normalize_input`` step turns a voice note into text *before* any
downstream node sees it, so the whole pipeline stays single-modality. Both
providers satisfy the :class:`~shared.providers.base.Transcriber` protocol, so
selection is a config change (``TRANSCRIPTION_PROVIDER``).

There's no local counterpart to the hosted provider the way vision has Ollama:
OpenAI's own docs draw a hard line between plain speech-to-text — the dedicated
``audio.transcriptions`` endpoint used here (``whisper-1`` et al.) — and audio as
*chat* input (the separate ``gpt-4o-audio-preview`` family, for conversational
audio understanding, not transcription); and Ollama's OpenAI-compatible surface
covers chat completions and embeddings only, with no transcription endpoint to
point at. So the local path stays :class:`HuggingFaceTranscriber`, which imports
``transformers`` **lazily** (only when built), so this module — and the app —
import fine without the heavy ML stack; the offline model is needed only when
``TRANSCRIPTION_PROVIDER=huggingface`` is actually selected.
"""

import asyncio
from typing import TYPE_CHECKING

from shared.core.config import Settings
from shared.providers._config import require_secret
from shared.providers.base import Transcriber
from shared.providers.errors import ProviderConfigError

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _filename_for(mime_type: str) -> str:
    """A filename carrying the extension OpenAI's endpoint infers the audio
    format from — it doesn't trust the multipart content-type, only the name.

    Every mime type this app accepts (``_AUDIO_MIMES`` in the API's request
    schema) is ``audio/<subtype>`` or ``audio/x-<subtype>``, and the bare
    subtype is already one of the endpoint's supported extensions (wav, mp3,
    mp4, m4a, mpeg, webm, ogg, ...), so stripping the `x-` prefix is enough.
    """
    subtype = mime_type.split("/", 1)[-1].removeprefix("x-")
    return f"audio.{subtype}"


class OpenAIWhisperTranscriber:
    """Transcription via the hosted OpenAI Whisper API (the dedicated
    ``audio.transcriptions`` endpoint, not a chat-completions call)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-1",
        client: "AsyncOpenAI | None" = None,
    ) -> None:
        self._model = model
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)
        self._client = client

    async def transcribe(self, audio: bytes, *, mime_type: str) -> str:
        """Return the transcript of an audio clip (uploaded as multipart)."""
        response = await self._client.audio.transcriptions.create(
            model=self._model, file=(_filename_for(mime_type), audio, mime_type)
        )
        return response.text


class HuggingFaceTranscriber:
    """Transcription via a local HuggingFace Whisper model (offline, no API).

    The ``transformers`` ASR pipeline is built **once at construction** — mirroring
    the local embedder — so the heavy, one-time model load happens when the
    provider is warmed at startup rather than blocking the first voice-note turn;
    inference is then offloaded to a thread (it's blocking/CPU-bound). Requires the
    ``local`` extra (``transformers``/``torch``) and an audio decoder (ffmpeg) in
    the runtime image; ``pipe`` is injectable so tests exercise the class without
    the ML stack.
    """

    def __init__(self, *, model: str, pipe: object | None = None) -> None:
        if pipe is None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # the local ML stack isn't installed
                raise ProviderConfigError(
                    "local transcription requires the 'local' extra (transformers/torch)"
                ) from exc
            pipe = pipeline("automatic-speech-recognition", model=model)
        self._pipe = pipe

    async def transcribe(self, audio: bytes, *, mime_type: str) -> str:
        """Transcribe raw audio bytes with the local model (decoded via ffmpeg)."""
        result = await asyncio.to_thread(self._pipe, audio)
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result)


def build_transcriber(settings: Settings) -> Transcriber:
    """Build the configured transcriber (hosted OpenAI Whisper or local HF)."""
    if settings.transcription_provider == "openai":
        return OpenAIWhisperTranscriber(
            api_key=require_secret(
                settings.openai_api_key, "OPENAI_API_KEY", "openai transcription"
            ),
            model=settings.transcription_model,
        )
    if settings.transcription_provider == "huggingface":
        return HuggingFaceTranscriber(model=settings.transcription_model_local)
    raise ProviderConfigError(f"Unknown transcription provider {settings.transcription_provider!r}")
