"""Pytest plugin: every test starts without the variables that mark the shell as an agent's."""
from __future__ import annotations

import pytest

AGENT_VARIABLES = ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE",
                   "ML_STACK_WORKSPACE_TOKEN", "ML_STACK_WORKSPACE_DENYLIST", "ML_STACK_WORKSPACE_AGENT",
                   "ML_STACK_SESSION_HARNESS", "ML_STACK_SESSION_ID", "CLAUDE_CODE_SESSION_ATTENDED")


@pytest.fixture(autouse=True)
def _no_agent_markers(monkeypatch):
    """Remove the agent markers; a test that sets one itself sets it after this runs."""
    for name in AGENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
