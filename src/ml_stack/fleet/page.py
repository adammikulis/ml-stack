"""Fleet page components and their shared script modules."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ml_stack.ui import Component, assemble, load

WEB = Path(__file__).parent / "web"
COMPONENTS_DIR = WEB / "components"
#: the page, in the order the elements wire themselves up
COMPONENTS = ("fleet-model", "fleet-nav", "sign-in", "startup-models", "cluster-actions", "first-run", "fleet-benchmark", "cluster-view",
              "chat-stream", "chat-view", "coordinator-control", "board-view", "projects-view", "wired-memory", "models-library", "model-browser", "models-view", "settings-view", "fit-model", "fit-view",
              "fit-charts", "rates-view", "telemetry-view",
              "workspace-jobs", "history-view", "tasks-view", "data-view",
              "training-view", "tools-view", "benchmarks-view", "gym-scene-controls",
              "gym-drone-camera", "gym-scene", "gym-recordings",
              "gym-world-options", "gym-model-options", "gym-view", "close-sheet")
MODULES = {"task-publishing": frozenset({"tasks-view"}),
           "task-review-quality": frozenset({"tasks-view"}),
           "gym-drone-geometry": frozenset({"gym-scene"}),
           "setup-recovery": frozenset({"first-run", "cluster-actions"}),
           "chat-model-picker": frozenset({"chat-view"}),
           "chat-coding": frozenset({"chat-view"}), "workspace-model": frozenset({
    "workspace-jobs", "history-view", "tasks-view", "data-view", "training-view", "tools-view", "benchmarks-view",
    "gym-world-options", "gym-recordings", "gym-model-options", "gym-view",
})}
#: the fit screen on its own, for a machine running no daemon
FIT_ONLY = ("fleet-model", "fit-model", "fit-view", "fit-charts", "rates-view",
            "telemetry-view")


def components(names: Sequence[str | Component] = COMPONENTS) -> list[Component]:
    """Return page components with shared modules inserted before their consumers."""
    parts = [part if isinstance(part, Component) else load(COMPONENTS_DIR, [part])[0]
             for part in names]
    present = {part.name for part in parts}
    for module, consumers in MODULES.items():
        if module in present:
            continue
        index = next((index for index, part in enumerate(parts) if part.name in consumers), None)
        if index is not None:
            parts.insert(index, load(COMPONENTS_DIR, [module])[0])
    return parts



def render(parts: Sequence[str | Component] = COMPONENTS) -> str:
    """The whole page, as one string."""
    shell = (WEB / "shell.html").read_text(encoding="utf-8")
    if any((part.name if isinstance(part, Component) else part) == "models-view" for part in parts):
        shell = shell.replace("    <fit-view></fit-view>\n", "")
    return assemble(shell, components(parts))
