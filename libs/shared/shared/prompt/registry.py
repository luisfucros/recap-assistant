"""Prompt registry — resolve prompts by ``name@version``.

The registry is the single source of prompts: no service or agent node holds an
inline prompt string. This scaffold loads prompts from an in-repo YAML store
(the always-available, offline fallback). A Langfuse-backed source can be layered
in front later without changing callers — resolution stays ``get(name, version)``.

YAML file format (one prompt per file, any filename under the prompts dir):

    name: system
    version: v1
    description: Base system prompt.
    input_variables: [display_name]
    template: |
      You are the reading companion for $display_name.
"""

from functools import lru_cache
from pathlib import Path

import yaml

from shared.prompt.errors import PromptNotFoundError
from shared.prompt.models import Prompt

# Default in-repo prompt store, shipped with the library.
_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptRegistry:
    """An immutable lookup of prompts, indexed by ``(name, version)``."""

    def __init__(self, prompts: list[Prompt]) -> None:
        self._by_key: dict[tuple[str, str], Prompt] = {(p.name, p.version): p for p in prompts}
        self._versions: dict[str, list[str]] = {}
        for prompt in prompts:
            self._versions.setdefault(prompt.name, []).append(prompt.version)

    @classmethod
    def from_directory(cls, path: Path) -> "PromptRegistry":
        """Load every ``*.yaml`` / ``*.yml`` prompt under ``path`` (non-recursive-safe).

        A missing directory yields an empty registry (valid — prompts arrive with
        the features that use them).
        """
        prompts: list[Prompt] = []
        if path.is_dir():
            for file in sorted([*path.glob("*.yaml"), *path.glob("*.yml")]):
                data = yaml.safe_load(file.read_text()) or {}
                prompts.append(
                    Prompt(
                        name=data["name"],
                        version=str(data["version"]),
                        template=data["template"],
                        input_variables=tuple(data.get("input_variables", [])),
                        description=data.get("description"),
                    )
                )
        return cls(prompts)

    def get(self, name: str, version: str | None = None) -> Prompt:
        """Return the prompt for ``name`` (and ``version``, or the latest).

        Raises:
            PromptNotFoundError: if no prompt matches.
        """
        if version is None:
            version = self._latest_version(name)
        try:
            return self._by_key[(name, version)]
        except KeyError:
            raise PromptNotFoundError(f"no prompt {name}@{version}") from None

    def resolve(self, ref: str) -> Prompt:
        """Resolve a ``"name@version"`` (or bare ``"name"`` → latest) reference."""
        name, _, version = ref.partition("@")
        return self.get(name, version or None)

    def _latest_version(self, name: str) -> str:
        versions = self._versions.get(name)
        if not versions:
            raise PromptNotFoundError(f"no prompt named {name!r}")
        # Lexicographic max — use zero-padded/sortable version labels (e.g. v01).
        return max(versions)


@lru_cache
def get_prompt_registry() -> PromptRegistry:
    """Return the process-wide prompt registry (loaded once from the default store)."""
    return PromptRegistry.from_directory(_DEFAULT_PROMPTS_DIR)
