"""Unit tests for language detection over the supported set."""

import pytest

from shared.core.enums import Language
from shared.ingestion_core.language import detect_language

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("This is a clear English sentence about reading many books.", Language.EN),
        ("Este es un texto claramente escrito en el idioma español.", Language.ES),
        ("Dies ist ein eindeutig auf Deutsch geschriebener Satz über Bücher.", Language.DE),
        ("Ceci est une phrase clairement écrite en langue française.", Language.FR),
        ("Questa è una frase chiaramente scritta nella lingua italiana.", Language.IT),
    ],
)
def test_detects_each_supported_language(text: str, expected: Language) -> None:
    assert detect_language(text, default=Language.EN) is expected


def test_empty_text_falls_back_to_default() -> None:
    assert detect_language("", default=Language.DE) is Language.DE
    assert detect_language("   \n  ", default=Language.ES) is Language.ES


def test_symbol_only_text_falls_back_to_default() -> None:
    # No linguistic content → detector is inconclusive → default.
    assert detect_language("1234 5678 !!! ??? ...", default=Language.IT) is Language.IT
