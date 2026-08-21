"""The ``Prompt`` value object.

A prompt is addressed by ``name@version`` and rendered with typed variables.
Templates use ``string.Template`` (``$var`` / ``${var}``) rather than ``str.format``
so prompt bodies can contain literal braces (JSON examples, code) without escaping.
"""

from dataclasses import dataclass
from string import Template

from shared.prompt.errors import PromptRenderError


@dataclass(frozen=True, slots=True)
class Prompt:
    """A single, versioned prompt template.

    Attributes:
        name: Logical prompt name (e.g. "system", "summarizer").
        version: Version label (e.g. "v1"); ``name@version`` is the address.
        template: The ``string.Template`` body.
        input_variables: Variables the caller must supply; enforced by ``render``.
        description: Optional human note on the prompt's purpose.
    """

    name: str
    version: str
    template: str
    input_variables: tuple[str, ...] = ()
    description: str | None = None

    @property
    def ref(self) -> str:
        """The ``name@version`` address of this prompt."""
        return f"{self.name}@{self.version}"

    def render(self, /, **variables: object) -> str:
        """Render the template, requiring every declared input variable.

        Raises:
            PromptRenderError: if a declared/used variable is missing.
        """
        missing = [v for v in self.input_variables if v not in variables]
        if missing:
            raise PromptRenderError(f"prompt {self.ref} missing variables: {missing}")
        try:
            return Template(self.template).substitute(**variables)
        except KeyError as exc:  # a $placeholder in the body with no matching variable
            raise PromptRenderError(f"prompt {self.ref} missing variable: {exc}") from exc
