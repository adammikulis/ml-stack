"""The extras a full install has, and which of them this one holds."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from ml_stack.log import say

__all__ = ["STANDARD", "Capability", "extras", "missing", "report", "standard"]


@dataclass(frozen=True, slots=True)
class Capability:
    """One extra: what it is called, what it lets this machine do, and the import that
    proves it is here."""

    extra: str
    name: str
    does: str
    module: str

    @property
    def fix(self) -> str:
        return f"pip install 'ml-stack[{self.extra}]'"

    def present(self) -> bool:
        """Whether the module behind this extra can be imported on this machine."""
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False


STANDARD: tuple[Capability, ...] = (
    Capability("store", "the graph store",
               "keeps a graph, and the bench's runs, in one file on disk", "ladybug"),
    Capability("hub", "model downloads",
               "fetches a model by name into the one cache on this machine",
               "huggingface_hub"),
    Capability("web", "reading the web",
               "searches the web and reads a page into text", "ddgs"),
    Capability("plot", "plots", "draws the panels ml-stack-serve fit --plot writes",
               "matplotlib"),
    Capability("graph", "graph maths", "builds a graph out of what it has read", "numpy"),
)


def extras() -> str:
    """The extras a full install asks pip for, as pip is given them."""
    return ",".join(c.extra for c in STANDARD)


def standard() -> list[tuple[Capability, bool]]:
    """Every extra a full install has, with whether this machine holds it."""
    return [(c, c.present()) for c in STANDARD]


def missing() -> list[Capability]:
    """The extras a full install has and this one does not."""
    return [c for c, here in standard() if not here]


def report() -> int:
    """Print what this install cannot do, and the line that fixes each. 0 when it can do
    everything."""
    gone = missing()
    for one in gone:
        say(f"  ! {one.name}: not installed -- {one.does}")
        say(f"    fix: {one.fix}")
    if not gone:
        say("  every part of a full install is here")
    return 1 if gone else 0


if __name__ == "__main__":
    raise SystemExit(report())
