"""Which harness drives a local agent: ml-stack's own loop for chat-sized board agents, the Codex
harness for coding."""

from __future__ import annotations

__all__ = ["CODEX", "OWN", "RUNNER"]

OWN = "ml-stack-agent"
CODEX = "codex"
RUNNER = "ml_stack.workspace.localcoding"
