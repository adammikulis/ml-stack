"""Loads src/ml_stack/worktreerules.py by file, for the hook scripts that sit beside this one."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def guard_off() -> bool:
    """Whether MLSTACK_GUARD=off turns the soft rules off; the hooks run only for an agent's calls, so the hard
    rules stay in force."""
    return os.environ.get("MLSTACK_GUARD") == "off"


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
