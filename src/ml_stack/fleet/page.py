"""The fleet's page, assembled from the components under ``web/components``.

`COMPONENTS` is the whole interface. A caller hands `render` a shorter list to leave a
screen out -- ``ml-stack-serve fit --ui`` serves `FIT_ONLY`, which is the fit view and
nothing that needs a daemon.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ml_stack.ui import Component, assemble, load

WEB = Path(__file__).parent / "web"
COMPONENTS_DIR = WEB / "components"
#: the page, in the order the elements wire themselves up
COMPONENTS = ("fleet-model", "fleet-nav", "sign-in", "first-run", "cluster-view",
              "chat-view", "models-view", "settings-view", "fit-view", "close-sheet")
#: the fit view on its own, for a machine running no daemon
FIT_ONLY = ("fleet-model", "fit-view")


def components(names: Sequence[str | Component] = COMPONENTS) -> list[Component]:
    """The named components: a bare name is one of the page's own under ``web/components``;
    a `Component` is taken as given, wherever its file lives."""
    return [c if isinstance(c, Component) else load(COMPONENTS_DIR, [c])[0] for c in names]


def render(parts: Sequence[str | Component] = COMPONENTS) -> str:
    """The whole page, as one string."""
    shell = (WEB / "shell.html").read_text(encoding="utf-8")
    return assemble(shell, components(parts))
