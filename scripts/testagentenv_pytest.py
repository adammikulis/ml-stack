"""Pytest plugin: every test starts without the variables that mark the shell as an agent's."""
from __future__ import annotations

import pytest

AGENT_VARIABLES = ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CLAUDECODE", "POOLHOUSE_AGENT", "POOLHOUSE_NONINTERACTIVE",
                   "POOLHOUSE_WORKSPACE_TOKEN", "POOLHOUSE_WORKSPACE_DENYLIST", "POOLHOUSE_WORKSPACE_AGENT",
                   "POOLHOUSE_SESSION_HARNESS", "POOLHOUSE_SESSION_ID", "CLAUDE_CODE_SESSION_ATTENDED")


@pytest.fixture(autouse=True)
def _no_agent_markers(monkeypatch):
    """Remove the agent markers; a test that sets one itself sets it after this runs."""
    for name in AGENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
