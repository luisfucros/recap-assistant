"""Language detection over the supported set, mapped to the shared ``Language``.

Detection is constrained to exactly the languages the product supports (English,
Spanish, German, French, Italian) by building the detector ``from_languages`` —
this both bounds memory (only those models load) and keeps results inside the
enum. A document whose language can't be determined (e.g. empty or symbol-only
text) falls back to a caller-supplied default; the detector is built once per
process and cached, since loading its models is not free.
"""

from functools import lru_cache

from shared.core.enums import Language


@lru_cache(maxsize=1)
def _detector():  # noqa: ANN202 — lingua's builder return type is internal
    """Build (once) a detector restricted to the supported languages."""
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(*_lingua_by_supported()).build()


@lru_cache(maxsize=1)
def _supported_by_lingua() -> dict[object, Language]:
    """Map each supported lingua ``Language`` to our ``Language`` enum member."""
    from lingua import Language as LinguaLanguage

    return {
        LinguaLanguage.ENGLISH: Language.EN,
        LinguaLanguage.SPANISH: Language.ES,
        LinguaLanguage.GERMAN: Language.DE,
        LinguaLanguage.FRENCH: Language.FR,
        LinguaLanguage.ITALIAN: Language.IT,
    }


def _lingua_by_supported() -> list[object]:
    return list(_supported_by_lingua().keys())


def detect_language(text: str, *, default: Language) -> Language:
    """Detect ``text``'s language, mapped to :class:`Language`.

    Args:
        text: Text to classify (typically the document's full text).
        default: Returned when the language can't be determined.

    Returns:
        The detected supported language, or ``default`` when detection yields
        nothing (empty/inconclusive input).
    """
    if not text.strip():
        return default
    detected = _detector().detect_language_of(text)
    if detected is None:
        return default
    return _supported_by_lingua().get(detected, default)
