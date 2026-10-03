"""Platform process, signal, file, and desktop operations."""

from __future__ import annotations

import contextlib
import os
import platform as _platform
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "detached_kwargs",
    "is_windows",
    "on_quit",
    "private_file",
    "process_group_kwargs",
    "quit_signals",
    "stop_gently",
    "stop_pid",
]

# subprocess only defines these on Windows; the values are Win32's own and do not change.
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CTRL_BREAK_EVENT = getattr(signal, "CTRL_BREAK_EVENT", 1)


def is_windows() -> bool:
    """Read when asked, never cached, so a test can say otherwise."""
    return _platform.system() == "Windows"


def applescript_quote(text: str) -> str:
    """``text`` safe inside an AppleScript string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


# -- starting and stopping a child -----------------------------------------------------
def start_process(argv, **kwargs):
    """Start a child in its own platform process group."""
    return subprocess.Popen(argv, **process_group_kwargs(), **kwargs)


def process_group_kwargs() -> dict[str, Any]:
    """The ``Popen`` keywords that put a child in a process group of its own.

    On POSIX that is a new session, so a signal aimed at the child does not also reach the
    daemon and a signal aimed at the daemon's terminal does not also kill the job. On
    Windows ``start_new_session`` is silently ignored -- the child would share the
    daemon's console group and a Ctrl+Break meant for one job would hit every job -- and
    ``CREATE_NEW_PROCESS_GROUP`` is what gives it a group ``CTRL_BREAK_EVENT`` can be
    aimed at.
    """
    if is_windows():
        return {"creationflags": CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def detached_kwargs() -> dict[str, Any]:
    """The ``Popen`` keywords for a child that outlives the terminal that started it.

    `process_group_kwargs` plus ``DETACHED_PROCESS`` on Windows, which takes the child off
    the console altogether so closing the window does not end it. A child started this way
    has no console to receive ``CTRL_BREAK_EVENT``, so `stop_gently` falls through to
    ``TerminateProcess`` for it.
    """
    if is_windows():
        return {"creationflags": CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS}
    return {"start_new_session": True}


def stop_gently(proc: Any) -> str:
    """Ask ``proc`` to stop in a way a checkpointing loop can catch. Returns what was sent.

    POSIX: ``SIGTERM``, which ``signal.signal(SIGTERM, ...)`` in the job catches. Windows
    has no SIGTERM to send -- ``send_signal(SIGTERM)`` there is ``TerminateProcess``, which
    nothing can catch -- so a ``CTRL_BREAK_EVENT`` is sent to the job's process group,
    which reaches it as ``SIGBREAK`` (``signal.signal(signal.SIGBREAK, ...)``). That needs
    the job to have been started with ``process_group_kwargs()`` and to share a console
    with the daemon; a daemon running with no console (a Scheduled Task with no window)
    gets ``OSError`` from the attempt and falls back to ``terminate()``, and this says so
    in what it returns so the log can record which one the job actually received.
    """
    if not is_windows():
        proc.send_signal(signal.SIGTERM)
        return "SIGTERM"
    try:
        proc.send_signal(CTRL_BREAK_EVENT)
        return "CTRL_BREAK_EVENT"
    except OSError:
        proc.terminate()
        return "TerminateProcess"


def stop_pid(pid: int) -> str:
    """`stop_gently` for a process there is no ``Popen`` for -- a bench the daemon adopted
    after ``ml-stack-bench --detach`` started it. Returns what was sent.

    POSIX: ``SIGTERM`` by pid, which the bench turns into an exit that releases its model
    and its lock. Windows: ``os.kill`` with ``CTRL_BREAK_EVENT`` reaches the process group
    ``pid`` heads, the way `stop_gently` does through a handle; a group with no console
    -- a bench started ``DETACHED_PROCESS`` -- refuses it with ``OSError``, and then
    ``os.kill`` with any other signal is ``TerminateProcess``, which nothing can catch, so
    what was actually sent is said. A pid that is gone raises ``OSError`` either way.
    """
    if not is_windows():
        os.kill(pid, signal.SIGTERM)
        return "SIGTERM"
    try:
        os.kill(pid, CTRL_BREAK_EVENT)
        return "CTRL_BREAK_EVENT"
    except OSError:
        os.kill(pid, signal.SIGTERM)
        return "TerminateProcess"


# -- shutting the daemon itself down ----------------------------------------------------
def quit_signals() -> list[int]:
    """Every signal that means 'shut down, cleanly' on this platform.

    ``SIGTERM`` everywhere -- launchd, systemd and a person's ``kill`` all send it. On
    Windows a console closing, ``Ctrl+Break``, and a stop from the Task Scheduler's own
    ``schtasks /End`` (which is a ``TerminateProcess`` and cannot be caught -- see the
    README) arrive as ``SIGBREAK`` when they arrive at all, so it is hooked as well.
    """
    out = [signal.SIGTERM]
    if is_windows() and hasattr(signal, "SIGBREAK"):
        out.append(signal.SIGBREAK)
    return out


def on_quit(callback: Callable[[int, Any], None]) -> list[int]:
    """Install ``callback`` for every quit signal. Returns the ones actually hooked.

    Only the main thread may set a signal handler; called from anywhere else this hooks
    nothing rather than raising, since a daemon started from a worker thread (the tests do
    that) still has to serve.
    """
    if threading.current_thread() is not threading.main_thread():
        return []
    hooked: list[int] = []
    for signum in quit_signals():
        with contextlib.suppress(ValueError, OSError, AttributeError):
            signal.signal(signum, callback)
            hooked.append(signum)
    return hooked


# -- a file for this user only ----------------------------------------------------------
def private_file(path: Path | str) -> None:
    """Make ``path`` readable and writable by its owner and nobody else.

    ``chmod(0o600)`` on POSIX. On Windows ``chmod`` knows only the read-only attribute, so
    ``0o600`` changes nothing about who may read the file; the equivalent is to strip the
    inherited ACL and grant the owner alone, which is what ``icacls`` does. Best effort:
    a file under the user's own profile is already unreadable to other accounts, and a
    refused ``icacls`` must not stop a key from being written.
    """
    p = Path(path)
    if not is_windows():
        p.chmod(0o600)
        return
    owner = os.environ.get("USERNAME", "")
    if not owner:
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            ["icacls", str(p), "/inheritance:r", "/grant:r", f"{owner}:F"],
            capture_output=True, check=False, timeout=30)


def open_path(path: Path | str) -> str:
    """Open ``path`` with whatever this desktop opens files with -- ``open`` on macOS,
    the shell association on Windows, ``xdg-open`` elsewhere -- and return the command
    used, or the reason it could not."""
    import shutil
    import subprocess

    where = str(path)
    if is_windows():
        try:
            os.startfile(where)  # type: ignore[attr-defined]  # noqa: S606 - the user's own file
            return "startfile"
        except OSError as exc:
            return f"could not open {where}: {exc}"
    tool = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(tool) is None:
        return f"could not open {where}: no {tool} on this machine"
    subprocess.Popen([tool, where], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tool
