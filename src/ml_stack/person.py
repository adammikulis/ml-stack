"""Whether the process is a person's: a terminal on stdin and stdout and no agent marker."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from collections.abc import Mapping

__all__ = ["AGENT_MARKERS", "HumanRequired", "is_terminal", "marked", "require_person"]

_MSVCRT = importlib.import_module("msvcrt") if sys.platform == "win32" else None

AGENT_MARKERS = ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE")
"""Environment variables whose presence means an agent started this process."""


class HumanRequired(PermissionError):
    """An action that only a person at a terminal may take was asked for by something else."""


def marked(env: Mapping[str, str] | None = None) -> str:
    """The first agent marker set in ``env`` (the process's when not given), else an empty string."""
    env = os.environ if env is None else env
    return next((name for name in AGENT_MARKERS if env.get(name)), "")



def is_terminal(stream) -> bool:
    """Whether the stream is attached to an interactive terminal."""
    if not stream.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:
        descriptor = stream.fileno()
        handle = _MSVCRT.get_osfhandle(descriptor)
        mode = ctypes.c_ulong()
        return bool(ctypes.windll.kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))
    except (AttributeError, OSError, ValueError):
        return False

def require_person(action: str, terminal: tuple[bool, bool] | None = None,
                   env: Mapping[str, str] | None = None) -> None:
    """Raise `HumanRequired` unless stdin and stdout are terminals and no agent marker is
    set. Every way of minting a grant goes through this first."""
    tty_in, tty_out = terminal or (is_terminal(sys.stdin), is_terminal(sys.stdout))
    found = marked(env)
    if found:
        raise HumanRequired(f"{action} is for a person; this process was started by an agent "
                            f"({found} is set)")
    if not (tty_in and tty_out):
        raise HumanRequired(f"{action} needs a terminal on stdin and stdout")
