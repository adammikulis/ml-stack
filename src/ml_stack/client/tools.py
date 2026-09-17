"""One declaration per tool, and the things derived from it.

A tool is declared once: what it is called, what it does, what arguments it takes, what a
user says to reach it, which group it belongs to, and what it needs in order to run. The
schemas a model is offered and the documents a selector embeds are both read off that
declaration, so neither can drift from the other.

``category`` groups tools a selector may choose between. ``needs`` names the capabilities a
tool cannot run without. They are different questions and have different words.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NOTHING", "ToolSpec", "documents", "function_schemas", "in_category",
           "runnable", "spec_by_name"]

#: The JSON Schema of a tool that takes no arguments.
NOTHING: Mapping[str, Any] = {"type": "object", "properties": {}, "required": []}


@dataclass(frozen=True)
class ToolSpec:
    """Everything one tool declares about itself."""

    name: str
    description: str
    schema: Mapping[str, Any] = field(default_factory=lambda: dict(NOTHING))
    examples: tuple[str, ...] = ()
    category: str = ""
    needs: tuple[str, ...] = ()

    def function_schema(self) -> dict[str, Any]:
        """This tool as the OpenAI function schema a model is offered."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.schema),
            },
        }

    def documents(self) -> tuple[str, ...]:
        """The texts a selector compares a user's message against.

        The example utterances, because a message against messages is like-to-like. A tool
        that declares none falls back to its description, which is worse and measurable.
        """
        said = tuple(e for e in self.examples if str(e).strip())
        return said or (f"{self.name}: {self.description}",)

    def required(self) -> tuple[str, ...]:
        """The argument names this tool cannot run without."""
        return tuple(str(r) for r in (self.schema.get("required") or ()))


def function_schemas(specs: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """The OpenAI tool schemas for these declarations, in order."""
    return [spec.function_schema() for spec in specs]


def documents(specs: Iterable[ToolSpec]) -> dict[str, tuple[str, ...]]:
    """``{name: texts}`` for a :class:`~ml_stack.client.select.Selector`."""
    return {spec.name: spec.documents() for spec in specs}


def spec_by_name(specs: Iterable[ToolSpec]) -> dict[str, ToolSpec]:
    """These declarations keyed by tool name. Raises on a duplicate name."""
    out: dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in out:
            raise ValueError(f"two tools are declared as {spec.name!r}")
        out[spec.name] = spec
    return out


def in_category(specs: Iterable[ToolSpec], category: str) -> list[ToolSpec]:
    """The declarations in one category."""
    return [spec for spec in specs if spec.category == category]


def runnable(specs: Iterable[ToolSpec], granted: Sequence[str]) -> list[ToolSpec]:
    """The declarations every one of whose ``needs`` is in ``granted``."""
    held = set(granted)
    return [spec for spec in specs if held.issuperset(spec.needs)]
