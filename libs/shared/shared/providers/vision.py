"""Image-to-text provider (FR-19): an OpenAI-compatible vision chat API.

``normalize_input`` captions an uploaded image to text before the agent sees it,
keeping the pipeline single-modality (the description is reasoned over, never the
pixels). One provider, :class:`OpenAIVisionDescriber`, satisfies the
:class:`~shared.providers.base.ImageDescriber` protocol and serves both modes,
selected by ``VISION_PROVIDER``:

* **hosted** — OpenAI's vision API (needs ``OPENAI_API_KEY``);
* **local** — an Ollama vision model (e.g. ``llava``) over Ollama's
  OpenAI-compatible endpoint at ``OLLAMA_BASE_URL`` — no key, and **no heavy**
  ``transformers``/``torch`` **install** (unlike the local Whisper transcriber).

Both speak the identical chat-completions image payload via the official
``openai`` SDK client, so they are the same class pointed at a different
``base_url`` — mirroring how the LLM layer reuses an OpenAI-compatible client
for Ollama (``api/llm.py``'s ``ChatOpenAI(..., api_key="ollama", ...)``).
"""

import base64
from typing import TYPE_CHECKING

from shared.core.config import Settings
from shared.providers._config import require_secret, resolve_ollama_api_key
from shared.providers.base import ImageDescriber
from shared.providers.errors import ProviderConfigError

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_DESCRIBE_PROMPT = (
    "Describe this image in detail for someone who cannot see it. Note any text, "
    "diagrams, or figures relevant to a reading assistant."
)


class OpenAIVisionDescriber:
    """Image description via an OpenAI-compatible vision chat API.

    Works against hosted OpenAI or a local Ollama vision model — the only
    differences are the client's ``base_url``/``api_key``, so the same request
    shape serves both.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        client: "AsyncOpenAI | None" = None,
    ) -> None:
        self._model = model
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client

    async def describe(self, image: bytes, *, mime_type: str) -> str:
        """Return a textual description of an image (sent as a base64 data URI)."""
        data_uri = f"data:{mime_type};base64,{base64.b64encode(image).decode()}"
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _DESCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""


def build_image_describer(settings: Settings) -> ImageDescriber:
    """Build the configured image describer (hosted OpenAI vision or local Ollama)."""
    if settings.vision_provider == "openai":
        return OpenAIVisionDescriber(
            model=settings.vision_model,
            api_key=require_secret(settings.openai_api_key, "OPENAI_API_KEY", "openai vision"),
        )
    if settings.vision_provider == "ollama":
        # Local Ollama ignores the key (a placeholder is used, mirroring
        # `api/llm.py`'s `ChatOpenAI`); OLLAMA_API_KEY lets this same client
        # reach Ollama Cloud's hosted vision models instead. `vision_model_local`
        # is the pulled/cloud model id (e.g. `llava`).
        return OpenAIVisionDescriber(
            model=settings.vision_model_local,
            api_key=resolve_ollama_api_key(settings.ollama_api_key),
            base_url=settings.ollama_base_url,
        )
    raise ProviderConfigError(f"Unknown vision provider {settings.vision_provider!r}")
