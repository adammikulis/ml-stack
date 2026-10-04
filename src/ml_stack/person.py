"""Whether the process is a person's: a terminal on stdin and stdout and no agent marker."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

__all__ = ["AGENT_MARKERS", "HumanRequired", "marked", "require_person"]

AGENT_MARKERS = ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE")
"""Environment variables whose presence means an agent started this process."""


class HumanRequired(PermissionError):
    """An action that only a person at a terminal may take was asked for by something else."""


def marked(env: Mapping[str, str] | None = None) -> str:
    """The first agent marker set in ``env`` (the process's when not given), else an empty string."""
    env = os.environ if env is None else env
    return next((name for name in AGENT_MARKERS if env.get(name)), "")


def require_person(action: str, terminal: tuple[bool, bool] | None = None,
                   env: Mapping[str, str] | None = None) -> None:
    """Raise `HumanRequired` unless stdin and stdout are terminals and no agent marker is
    set. Every way of minting a grant goes through this first."""
    tty_in, tty_out = terminal or (sys.stdin.isatty(), sys.stdout.isatty())
    found = marked(env)
    if found:
        raise HumanRequired(f"{action} is for a person; this process was started by an agent "
                            f"({found} is set)")
    if not (tty_in and tty_out):
        raise HumanRequired(f"{action} needs a terminal on stdin and stdout")
