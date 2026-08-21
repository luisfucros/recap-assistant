"""Prompt registry error types."""


class PromptError(Exception):
    """Base class for prompt registry errors."""


class PromptNotFoundError(PromptError):
    """No prompt matches the requested name (and version)."""


class PromptRenderError(PromptError):
    """A prompt could not be rendered — a required variable was missing."""
