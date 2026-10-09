"""Changing the wiring limit: the one privileged call, the boot-time daemon that keeps it, and
who may ask. Every path here is for a person; an agent's process is refused first."""

from __future__ import annotations

import platform
import plistlib
import shlex
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from poolhouse import authority, home
from poolhouse.files import read_json, write_json
from poolhouse.person import AGENT_MARKERS, HumanRequired, is_terminal
from poolhouse.serve.wired import KEY, MIB, MIN_MB, SYSCTL, Hooks, max_mb

__all__ = ["AGENT_MARKERS", "DAEMON", "LABEL", "Applied", "State", "argv_for", "checked", "original_mb",
           "reset", "script_for", "set_limit", "state"]

LABEL = "stack.ml.wired-limit"
DAEMON = Path(f"/Library/LaunchDaemons/{LABEL}.plist")
LAUNCHCTL = "/bin/launchctl"
_BUSY = threading.Lock()


@dataclass(frozen=True, slots=True)
class Applied:
    """What a change did: the command, whether it ran, and the limit before and after."""

    argv: tuple[str, ...]
    ok: bool
    message: str
    before_mb: int
    after_mb: int


@dataclass(frozen=True, slots=True)
class State:
    """Whether the limit is kept across restarts, and whether the live value agrees."""

    kept: bool
    kept_mb: int
    live_mb: int
    original_mb: int | None
    drift: bool


def _record() -> Path:
    return home.state("wired-limit.json")


def original_mb() -> int | None:
    """The limit this machine reported before the first change, or None when none was made."""
    held = read_json(_record(), {})
    value = held.get("original_mb") if isinstance(held, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _remember(before: int) -> None:
    if original_mb() is None:
        write_json(_record(), {"schema": 1, "original_mb": int(before)})


def checked(value: object, total: int, *, lowest: int = MIN_MB) -> int:
    """``value`` as a whole number of MB within ``[lowest, installed - 8 GiB]``, else
    `ValueError`. A bool, a float, a string or anything else that is not an int is refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("the limit must be a whole number of MB")
    if value < lowest or value > max_mb(total):
        raise ValueError(f"the limit must be between {lowest} and {max_mb(total)} MB")
    return value


def state(hooks: Hooks | None = None) -> State:
    """Whether the boot-time daemon is installed and what it sets, read without root."""
    hooks = hooks or Hooks()
    live, kept_mb = hooks.live_mb(), 0
    try:
        args = plistlib.loads((hooks.daemon or DAEMON).read_bytes()).get("ProgramArguments", [])
        kept_mb = int(str(args[-1]).split("=")[-1])
    except (OSError, ValueError, IndexError, plistlib.InvalidFileException):
        kept_mb = 0
    return State(bool(kept_mb), kept_mb, live, original_mb(), bool(kept_mb) and live != kept_mb)


def _plist(mb: int) -> str:
    body = plistlib.dumps({"Label": LABEL, "ProgramArguments": [SYSCTL, "-w", f"{KEY}={int(mb)}"],
                           "RunAtLoad": True}).decode()
    return body.replace("\n", "")


def script_for(mb: int | None, keep: bool | None, daemon: Path = DAEMON) -> str:
    """The shell script run with administrator rights. ``mb`` is an int and the only part that
    is not fixed text; ``keep`` True installs the daemon, False removes it, None leaves it."""
    steps: list[str] = []
    place = shlex.quote(str(daemon))
    if keep:
        steps += [f"printf '%s\\n' {shlex.quote(_plist(int(mb or 0)))} > {place}",
                  f"chown root:wheel {place}", f"chmod 644 {place}",
                  f"{LAUNCHCTL} bootout system/{LABEL} 2>/dev/null; true",
                  f"{LAUNCHCTL} bootstrap system {place}"]
    elif keep is False:
        steps += [f"{LAUNCHCTL} bootout system/{LABEL} 2>/dev/null; true", f"rm -f {place}"]
    if mb is not None:
        steps.append(f"{SYSCTL} -w {KEY}={int(mb)}")
    return " ; ".join(steps)


def argv_for(script: str, via: str) -> list[str]:
    """The command that runs ``script``: sudo at a terminal, macOS's own dialog otherwise."""
    if via == "sudo":
        return ["sudo", "/bin/sh", "-c", script]
    if via == "osascript":
        quoted = script.replace("\\", "\\\\").replace('"', '\\"')
        return ["osascript", "-e", f'do shell script "{quoted}" with administrator privileges']
    raise ValueError(f"unknown way to ask for administrator rights: {via!r}")


def _authorise(action: str, via: str, hooks: Hooks) -> None:
    """Pass the ``serve.wired-limit`` gate; the administrator prompt itself is macOS's dialog or sudo."""
    sudo = via == "sudo"
    authority.require("serve.wired-limit", action, hooks.terminal if sudo else (True, True), hooks.env)
    terminal = hooks.terminal or (is_terminal(sys.stdin), is_terminal(sys.stdout))
    if sudo and not all(terminal):
        raise HumanRequired(f"{action} through sudo needs a terminal on stdin and stdout")


def require_person(via: str) -> None:
    """Refuse, as `set_limit` does, a process an agent started; the command asks before it looks at the platform."""
    _authorise("change the wiring limit", via, Hooks())


def _run(argv: Sequence[str], capture: bool) -> tuple[int, str, str]:
    try:
        done = subprocess.run(list(argv), capture_output=capture, text=True, timeout=300,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", type(exc).__name__
    return done.returncode, done.stdout or "", done.stderr or ""


def _change(script: str, via: str, hooks: Hooks) -> Applied:
    if (hooks.system or platform.system()) != "Darwin":
        raise ValueError("the wiring limit is a macOS setting")
    if not _BUSY.acquire(blocking=False):
        raise ValueError("an administrator prompt is already open; answer it first")
    try:
        before = hooks.live_mb()
        _remember(before)
        argv = argv_for(script, via)
        code, _, err = (hooks.runner or _run)(argv, via != "sudo")
        after = hooks.live_mb()
    finally:
        _BUSY.release()
    if code != 0:
        why = "cancelled" if "-128" in err else (err.strip().splitlines() or ["failed"])[-1]
        return Applied(tuple(argv), False, why[:200], before, after)
    return Applied(tuple(argv), True, "set", before, after)


def set_limit(mb: object, *, keep: bool | None = None, via: str,
              hooks: Hooks | None = None) -> Applied:
    """Set the limit to ``mb`` MB (None leaves it) and install (``keep`` True), remove (False)
    or leave (None) the boot-time daemon, in one administrator prompt.

    ``via`` is ``sudo`` (a terminal) or ``osascript`` (macOS's dialog). Raises `HumanRequired`
    for a process an agent started or ``sudo`` with no terminal, `ValueError` for a bad ``mb``.
    """
    hooks = hooks or Hooks()
    _authorise("change the wiring limit", via, hooks)
    value = None if mb is None else checked(mb, hooks.total_bytes())
    if value is None and keep is not False:
        raise ValueError("nothing to change")
    if keep and value is None:
        raise ValueError("keeping a limit needs the limit")
    return _change(script_for(value, keep, hooks.daemon or DAEMON), via, hooks)


def reset(*, via: str, hooks: Hooks | None = None) -> Applied:
    """Put the limit back to what it was before the first change (0, the default share, when
    none was recorded) and remove the boot-time daemon."""
    hooks = hooks or Hooks()
    _authorise("reset the wiring limit", via, hooks)
    target = original_mb() or 0
    if target and target > hooks.total_bytes() // MIB:
        raise ValueError("the recorded original limit is larger than installed memory")
    return _change(script_for(target, False, hooks.daemon or DAEMON), via, hooks)
