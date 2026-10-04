"""The fleet page assembled from web components."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ml_stack.ui import Component, assemble, load

WEB = Path(__file__).parent / "web"
COMPONENTS_DIR = WEB / "components"
#: the page, in the order the elements wire themselves up
COMPONENTS = ("fleet-model", "fleet-nav", "sign-in", "startup-models", "first-run", "cluster-view",
              "chat-view", "wired-memory", "models-view", "settings-view", "fit-model", "fit-view",
              "fit-charts", "rates-view", "telemetry-view", "close-sheet")
#: the fit screen on its own, for a machine running no daemon
FIT_ONLY = ("fleet-model", "fit-model", "fit-view", "fit-charts", "rates-view",
            "telemetry-view")


def components(names: Sequence[str | Component] = COMPONENTS) -> list[Component]:
    """Return the named fleet components."""
    return [c if isinstance(c, Component) else load(COMPONENTS_DIR, [c])[0] for c in names]


def render(parts: Sequence[str | Component] = COMPONENTS) -> str:
    """The whole page, as one string."""
    shell = (WEB / "shell.html").read_text(encoding="utf-8")
    return assemble(shell, components(parts))
