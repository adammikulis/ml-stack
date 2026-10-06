"""Whether the process is a person's: a terminal on stdin and stdout and no agent marker."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from collections.abc import Mapping

import psutil

__all__ = ["AGENT_MARKERS", "WSL_FORWARDED", "HumanRequired", "from_wsl", "is_terminal", "marked",
           "require_person", "require_unmarked"]

_MSVCRT = importlib.import_module("msvcrt") if sys.platform == "win32" else None

AGENT_MARKERS = ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE")
"""Environment variables whose presence means an agent started this process."""
WSL_FORWARDED = "ML_STACK_WSL_FORWARDED"
"""Set by ``scripts/ml-stack-workspace`` when it hands the agent markers to a Windows program through WSLENV."""
WSL_HOSTS = frozenset({"wsl.exe", "wslhost.exe", "wslrelay.exe"})


class HumanRequired(PermissionError):
    """An action that only a person at a terminal may take was asked for by something else."""


def from_wsl() -> bool:
    """Whether this is a Windows process started from a WSL shell."""
    if sys.platform != "win32":
        return False
    try:
        return any(parent.name().lower() in WSL_HOSTS for parent in psutil.Process().parents())
    except psutil.Error:
        return False


def marked(env: Mapping[str, str] | None = None) -> str:
    """Why this process counts as an agent's: the first agent marker set in ``env`` (the process's
    when not given), or, for the process's own environment, a start from WSL that did not forward
    the markers; else an empty string."""
    own = env is None
    env = os.environ if env is None else env
    found = next((name for name in AGENT_MARKERS if env.get(name)), "")
    if found:
        return f"{found} is set"
    if own and not env.get(WSL_FORWARDED) and from_wsl():
        return "it was started from WSL without scripts/ml-stack-workspace, which forwards the agent markers"
    return ""


def require_unmarked(action: str, env: Mapping[str, str] | None = None) -> None:
    """Raise `HumanRequired` when `marked` finds an agent; no terminal is needed."""
    found = marked(env)
    if found:
        raise HumanRequired(f"{action} is for a person; this process was started by an agent "
                            f"({found})")


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
    require_unmarked(action, env)
    if not (tty_in and tty_out):
        raise HumanRequired(f"{action} needs a terminal on stdin and stdout")
