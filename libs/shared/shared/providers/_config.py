"""Small helpers shared by the provider factories."""

from pydantic import SecretStr

from shared.providers.errors import ProviderConfigError


def require_secret(value: SecretStr | None, env_name: str, what: str) -> str:
    """Return the secret's value, or raise a clear config error if it is unset.

    Args:
        value: The optional secret from settings.
        env_name: The environment variable the user should set (for the message).
        what: Human phrase naming what needs it (e.g. "openai embeddings").
    """
    if value is None:
        raise ProviderConfigError(f"{what} requires {env_name} to be set.")
    return value.get_secret_value()


def resolve_ollama_api_key(value: SecretStr | None) -> str:
    """Return ``OLLAMA_API_KEY`` if set, else the local no-auth placeholder.

    A local Ollama install's OpenAI-compatible endpoint accepts any non-empty
    string as a key, so the LLM/vision/embedding providers all default to the
    conventional placeholder ``"ollama"``. Setting ``OLLAMA_API_KEY`` lets that
    same client target Ollama Cloud's hosted models instead (paired with
    pointing ``OLLAMA_BASE_URL`` at the cloud endpoint) — no other code changes.
    """
    return value.get_secret_value() if value is not None else "ollama"
