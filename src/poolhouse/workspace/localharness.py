"""Which harness drives a local agent: poolhouse's loop for chat agents and Pi for coding."""

from __future__ import annotations

__all__ = ["CODEX", "OWN", "PI", "RUNNER"]

OWN = "poolhouse-agent"
CODEX = "codex"
PI = "pi"
RUNNER = "poolhouse.workspace.localcoding"
