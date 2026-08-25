"""Unit tests for the prompt registry and rendering (no I/O beyond tmp files)."""

from pathlib import Path

import pytest

from shared.prompt import (
    Prompt,
    PromptNotFoundError,
    PromptRegistry,
    PromptRenderError,
    get_prompt_registry,
)

# --- Prompt.render ---------------------------------------------------------


@pytest.mark.unit
def test_render_substitutes_variables() -> None:
    prompt = Prompt(name="system", version="v1", template="Hello $name", input_variables=("name",))
    assert prompt.render(name="Ada") == "Hello Ada"


@pytest.mark.unit
def test_render_preserves_literal_braces() -> None:
    # string.Template (not str.format) — JSON braces in the body survive.
    prompt = Prompt(name="tool", version="v1", template='Return {"k": $v}', input_variables=("v",))
    assert prompt.render(v=1) == 'Return {"k": 1}'


@pytest.mark.unit
def test_render_missing_declared_variable_raises() -> None:
    prompt = Prompt(name="system", version="v1", template="Hi $name", input_variables=("name",))
    with pytest.raises(PromptRenderError, match="missing variables"):
        prompt.render()


@pytest.mark.unit
def test_render_missing_placeholder_raises() -> None:
    # Placeholder present in body but not declared/supplied.
    prompt = Prompt(name="system", version="v1", template="Hi $name")
    with pytest.raises(PromptRenderError):
        prompt.render()


# --- PromptRegistry --------------------------------------------------------


def _write_prompt(directory: Path, filename: str, body: str) -> None:
    (directory / filename).write_text(body)


@pytest.fixture
def registry(tmp_path: Path) -> PromptRegistry:
    _write_prompt(
        tmp_path,
        "system_v1.yaml",
        "name: system\nversion: v1\ninput_variables: [reader]\ntemplate: |\n  Companion for $reader.\n",
    )
    _write_prompt(
        tmp_path,
        "system_v2.yaml",
        "name: system\nversion: v2\ntemplate: Newer system prompt.\n",
    )
    return PromptRegistry.from_directory(tmp_path)


@pytest.mark.unit
def test_get_by_name_and_version(registry: PromptRegistry) -> None:
    prompt = registry.get("system", "v1")
    assert prompt.ref == "system@v1"
    assert "Companion for $reader." in prompt.template


@pytest.mark.unit
def test_get_latest_version_when_unspecified(registry: PromptRegistry) -> None:
    assert registry.get("system").version == "v2"


@pytest.mark.unit
def test_resolve_ref_string(registry: PromptRegistry) -> None:
    assert registry.resolve("system@v1").version == "v1"
    assert registry.resolve("system").version == "v2"


@pytest.mark.unit
def test_missing_prompt_raises(registry: PromptRegistry) -> None:
    with pytest.raises(PromptNotFoundError):
        registry.get("nope")
    with pytest.raises(PromptNotFoundError):
        registry.get("system", "v9")


@pytest.mark.unit
def test_missing_directory_is_empty_registry(tmp_path: Path) -> None:
    registry = PromptRegistry.from_directory(tmp_path / "does-not-exist")
    with pytest.raises(PromptNotFoundError):
        registry.get("system")


# --- Shipped default prompt store ------------------------------------------

# The agent nodes and tools resolve these by name@version; they must exist in the
# in-repo store and render with their declared variables (the "no inline prompts"
# rule means a missing registration is a hard failure, not a silent fallback).


@pytest.mark.unit
def test_default_store_ships_agent_prompts() -> None:
    registry = get_prompt_registry()
    for ref in (
        "guardrail_in@v1",
        "guardrail_in@v2",
        "guardrail_in@v3",
        "planner@v1",
        "generate@v1",
        "generate@v2",
        "generate@v3",
        "generate@v4",
        "summarize@v1",
        "compaction@v1",
        "spoiler_check@v1",
        "evaluation_judge@v1",
        "memory_classify@v1",
    ):
        assert registry.resolve(ref).ref == ref


@pytest.mark.unit
def test_summarize_prompt_renders_with_its_variables() -> None:
    prompt = get_prompt_registry().get("summarize", "v1")
    rendered = prompt.render(
        title="The Odyssey", page_start=10, page_end=24, passages="Odysseus sets sail."
    )
    assert "The Odyssey" in rendered
    assert "10" in rendered and "24" in rendered
    assert "Odysseus sets sail." in rendered


@pytest.mark.unit
def test_generate_prompt_renders_display_name_and_answer_language() -> None:
    rendered = (
        get_prompt_registry()
        .get("generate", "v1")
        .render(display_name="Ada", answer_language="Spanish")
    )
    assert "Ada" in rendered
    # The answer-language instruction (FR-16.4) is present so replies come back in
    # the reader's language while quotes/citations stay in the document's.
    assert "Spanish" in rendered


@pytest.mark.unit
def test_generate_v2_instructs_document_id_sourcing() -> None:
    # v2 carries the fix for a real bug where the model reused a document's
    # title (itself populated from a filing's own accession number) as a
    # document_id argument; kept resolvable even though v3 is now the latest.
    rendered = (
        get_prompt_registry()
        .get("generate", "v2")
        .render(display_name="Ada", answer_language="Spanish")
    )
    assert "Ada" in rendered
    assert "Spanish" in rendered
    assert "get_reading_progress" in rendered
    assert "document_id" in rendered


@pytest.mark.unit
def test_generate_v3_hides_tool_errors() -> None:
    # Kept resolvable even though v4 is now the latest — v3 is the tool-error
    # leak fix (a raw "not a valid document id" error, including the document
    # label and id, used to surface verbatim in the streamed answer).
    rendered = (
        get_prompt_registry()
        .get("generate", "v3")
        .render(display_name="Ada", answer_language="Spanish")
    )
    assert "Ada" in rendered
    assert "Spanish" in rendered
    assert "get_reading_progress" in rendered
    assert "document_id" in rendered
    assert "never quote that error text" in rendered


@pytest.mark.unit
def test_generate_v4_is_the_latest_and_stays_inside_tools() -> None:
    # v4 is what the agent graph actually resolves ("generate", "v4" pinned in
    # graph.py) — never suggest workarounds the tools cannot do (copy-paste
    # document text, recap an attached transcript/image as a book).
    assert get_prompt_registry().get("generate").version == "v4"
    rendered = (
        get_prompt_registry()
        .get("generate", "v4")
        .render(display_name="Ada", answer_language="Spanish")
    )
    assert "Ada" in rendered
    assert "Spanish" in rendered
    assert "never quote that error text" in rendered
    assert "copy and paste" in rendered
    assert "Attached audio transcript" in rendered
    assert "do not invent a workaround" in rendered


@pytest.mark.unit
def test_guardrail_in_v2_allows_small_talk() -> None:
    # Kept resolvable even though v3 is now the latest — v2 is the small-talk
    # widening (a bare greeting or a personal fact used to be refused).
    rendered = get_prompt_registry().get("guardrail_in", "v2").render(message="hi, I'm Ada")
    assert "greeting" in rendered
    assert "hi, I'm Ada" in rendered


@pytest.mark.unit
def test_guardrail_in_v3_is_the_latest_and_reasons_in_answer_language() -> None:
    # v3 is what the agent graph actually resolves ("guardrail_in", "v3" pinned
    # in graph.py) — the user-facing reason is written in the reader's language
    # so a blocked turn isn't stuck in English.
    assert get_prompt_registry().get("guardrail_in").version == "v3"
    rendered = (
        get_prompt_registry()
        .get("guardrail_in", "v3")
        .render(message="hi, I'm Ada", answer_language="Spanish")
    )
    assert "greeting" in rendered
    assert "hi, I'm Ada" in rendered
    assert "Spanish" in rendered
    assert "written in Spanish" in rendered


@pytest.mark.unit
def test_memory_classify_prompt_renders_with_its_variables() -> None:
    rendered = get_prompt_registry().get("memory_classify", "v1").render(message="I'm 34")
    assert "I'm 34" in rendered


@pytest.mark.unit
def test_compaction_prompt_renders_with_its_variables() -> None:
    rendered = get_prompt_registry().get("compaction", "v1").render(transcript="Reader: hi")
    assert "Reader: hi" in rendered


@pytest.mark.unit
def test_spoiler_check_prompt_renders_with_its_variables() -> None:
    rendered = (
        get_prompt_registry()
        .get("spoiler_check", "v1")
        .render(title="The Odyssey", current_page=50, answer="Odysseus reaches home.")
    )
    assert "The Odyssey" in rendered
    assert "50" in rendered
    assert "Odysseus reaches home." in rendered


@pytest.mark.unit
def test_evaluation_judge_prompt_renders_with_its_variables() -> None:
    rendered = (
        get_prompt_registry()
        .get("evaluation_judge", "v1")
        .render(
            query="What happened?",
            context="Odysseus reaches home.",
            reference_answer="He returns to Ithaca.",
            answer="Odysseus arrives home.",
        )
    )
    assert "What happened?" in rendered
    assert "Odysseus reaches home." in rendered
    assert "He returns to Ithaca." in rendered
    assert "Odysseus arrives home." in rendered
