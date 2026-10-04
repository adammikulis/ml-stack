"""Which harness drives a local agent: ml-stack's own loop for chat-sized board agents, the Codex
harness for coding."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from typing import Any

__all__ = ["CODEX", "OWN", "launcher", "stub_command"]

OWN = "ml-stack-agent"
CODEX = "codex"
RUNNER = "ml_stack.workspace.localcoding"


def stub_command(model: str, project: str) -> str:
    """The one command a person runs to start the Codex harness on ``model`` by hand."""
    return f"ml-stack-codex --model {model}" + (f" --project {project}" if project else "")


def launcher() -> Callable[..., Any] | None:
    """``ml_stack.coding.launch_coding_agent`` when the harness branch has landed, else None.
    This is the seam: it is called as ``launch_coding_agent(model, role, project, harness='codex', name=, orders_from=)`` and returns the harness exit code."""
    found = sys.modules.get("ml_stack.coding")
    if found is None:
        spec = importlib.util.find_spec("ml_stack.coding")
        if spec is None or spec.loader is None:
            return None
        found = importlib.util.module_from_spec(spec)
        sys.modules["ml_stack.coding"] = found
        spec.loader.exec_module(found)
    return getattr(found, "launch_coding_agent", None)
