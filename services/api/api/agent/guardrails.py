"""Deterministic input/output guardrails for the agent.

Two ends of a turn are protected here, without an LLM, so the checks are cheap,
predictable, and unit-testable:

* **Input** — flag likely prompt-injection attempts and redact secrets/PII before
  a message is ever sent to a hosted LLM (the "no secrets/PII to external
  providers" rule). The topical-relevance and appropriateness judgments are
  LLM-based and live in the graph's ``guardrail_in`` node (using
  :class:`~api.agent.schemas.GuardrailDecision`); this service is the
  deterministic layer beneath them.
* **Output** — neutralize HTML/script constructs in the model's answer so a
  prompt-injection-driven payload can't turn into stored/reflected XSS, as
  defense-in-depth atop the SPA's text rendering.

Redaction reuses the shared log redactor (emails, JWTs, bearer tokens, API keys,
``secret=…`` pairs) so the project has one definition of "sensitive value".
"""

import html
import re
from dataclasses import dataclass

from loguru import logger

from shared.core.logging import redact_text

# Heuristic prompt-injection signals. Deliberately high-precision (a match is a
# strong signal), not exhaustive — the LLM guardrail catches the subtler cases.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\bignore\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?)"
    ),
    re.compile(r"(?i)\bdisregard\s+(?:all\s+|the\s+)?(?:previous|prior|above)\b"),
    re.compile(r"(?i)\bforget\s+(?:everything|all|your\s+instructions)\b"),
    re.compile(
        r"(?i)\b(?:reveal|show|print|repeat|leak)\b.{0,30}"
        r"(?:system\s+prompt|your\s+instructions|initial\s+prompt)"
    ),
    re.compile(r"(?i)\boverride\b.{0,20}(?:instructions|guardrails?|safety)"),
    re.compile(r"(?i)\byou\s+are\s+now\b.{0,40}(?:do\s+anything|unrestricted|dan\b)"),
    re.compile(r"(?i)\bpretend\s+(?:you\s+are|to\s+be)\b"),
)

# HTML tags to strip from model output (XSS defense-in-depth). Requires a letter
# (or '/') immediately after '<' so real tags (<script>, </b>, <img …>) match
# while prose with spaced angle brackets ("a < b and c > d") is left for escaping.
_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")
_DANGEROUS_URI = re.compile(r"(?i)(?:javascript|data|vbscript):")


@dataclass(slots=True)
class InputScreen:
    """The deterministic input-screen result for one user message."""

    injection_detected: bool
    redacted_text: str


class GuardrailService:
    """Deterministic screening of agent input and sanitization of agent output."""

    def screen_input(self, text: str) -> InputScreen:
        """Screen a user message: flag injection attempts and redact sensitive data.

        The redacted text is what may be forwarded to a hosted LLM; the
        ``injection_detected`` flag lets the graph short-circuit or harden the
        prompt before doing so.
        """
        injection = self.detect_injection(text)
        if injection:
            logger.info("guardrail.screen: injection detected")
        return InputScreen(
            injection_detected=injection,
            redacted_text=self.redact_for_llm(text),
        )

    @staticmethod
    def detect_injection(text: str) -> bool:
        """Return True if the text matches a known prompt-injection pattern."""
        return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)

    @staticmethod
    def redact_for_llm(text: str) -> str:
        """Redact secrets/PII (emails, tokens, API keys, ``secret=…``) from text.

        Applied before any hosted-LLM call so credentials or PII never leave the
        system in a prompt. Reuses the shared log-redaction rules.
        """
        return redact_text(text)

    @staticmethod
    def sanitize_output(text: str) -> str:
        """Neutralize HTML/script constructs in a model answer (XSS-safe).

        Strips HTML tags and defangs dangerous URI schemes (``javascript:`` etc.),
        then HTML-escapes any stray angle brackets. Reading answers are prose /
        markdown, so removing raw HTML costs nothing and closes the reflected-XSS
        vector even if a future surface renders the text as HTML.
        """
        without_tags = _TAG_PATTERN.sub("", text)
        defanged = _DANGEROUS_URI.sub("blocked:", without_tags)
        # Escape any remaining lone angle brackets so nothing re-parses as markup.
        return html.escape(defanged, quote=False)
