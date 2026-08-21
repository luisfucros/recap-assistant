"""Provider error types."""


class ProviderError(Exception):
    """Base class for provider construction or usage errors."""


class ProviderConfigError(ProviderError):
    """A provider was selected but its configuration is incomplete.

    Raised for a missing API key, an un-inferable embedding dimension, or an
    unavailable optional dependency (e.g. the local embedding extra). The
    message names the offending setting so the fix is obvious.
    """
