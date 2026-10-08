"""Loads src/ml_stack/worktreerules.py by file, for the hook scripts that sit beside this one."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_MARKERS = ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE", "CODEX_THREAD_ID", "CODEX_SESSION_ID")


def guard_off() -> str:
    """"off" when MLSTACK_GUARD=off and no agent marker is set, "hard" when it is set under a running
    agent (only the hard rules apply), else an empty string."""
    if os.environ.get("MLSTACK_GUARD") != "off":
        return ""
    return "hard" if any(os.environ.get(name) for name in AGENT_MARKERS) else "off"


def load():
    """The worktreerules module of the checkout these hooks sit in, None when it lacks one."""
    where = HERE.parent.parent / "src" / "ml_stack" / "worktreerules.py"
    if not where.exists():
        return None
    spec = importlib.util.spec_from_file_location("_ml_stack_worktreerules", where)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
