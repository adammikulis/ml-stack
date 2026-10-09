"""Which harness drives a local agent: ml-stack's loop for chat agents and Pi for coding."""

from __future__ import annotations

__all__ = ["CODEX", "OWN", "PI", "RUNNER"]

OWN = "ml-stack-agent"
CODEX = "codex"
PI = "pi"
RUNNER = "ml_stack.workspace.localcoding"
