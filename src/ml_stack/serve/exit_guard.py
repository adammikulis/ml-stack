"""Servers this process started stop when it does: at exit, on SIGTERM, SIGINT or SIGHUP, and
through a watchdog process when the host is killed outright."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ml_stack.platform import process_group_kwargs
from ml_stack.serve.process import kill_process_tree, started_at

logger = logging.getLogger(__name__)

TOLERANCE_S = 2.0
WATCHDOG = Path(__file__).with_name("_watchdog.py")
SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGBREAK")
                if hasattr(signal, name))

_lock = threading.RLock()
_guarded: dict[int, float] = {}
_watchdogs: dict[int, Any] = {}
_previous: dict[int, Any] = {}
_owner = os.getpid()
_exit_hooked = False


def protect(pid: int) -> None:
    """Stop ``pid``'s process tree when this process exits or is killed, until `release`.

    The first call registers the exit hook and the signal handlers; importing this module
    registers nothing.
    """
    created = started_at(pid)
    if created is None:
        return
    with _lock:
        _guarded[pid] = created
        _hook_exit()
        _hook_signals()
        _watch(pid, created)


def release(pid: int | None) -> None:
    """Stop guarding ``pid``: it was stopped, or it is meant to outlive this process."""
    with _lock:
        _guarded.pop(pid, None)
        watchdog = _watchdogs.pop(pid, None)
    if watchdog is not None:
        watchdog.terminate()
        try:
            watchdog.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            watchdog.kill()


def guarded() -> dict[int, float]:
    """The pids being guarded, with their start times."""
    with _lock:
        return dict(_guarded)


def stop_guarded(*, grace_s: float = 3.0) -> list[int]:
    """Stop every guarded server that is still the process that was guarded."""
    if os.getpid() != _owner:
        return []
    stopped: list[int] = []
    for pid, created in guarded().items():
        now = started_at(pid)
        if now is not None and abs(now - created) <= TOLERANCE_S:
            stopped += kill_process_tree(pid, grace_s=grace_s)
        release(pid)
    return stopped


def _hook_exit() -> None:
    global _exit_hooked
    if not _exit_hooked:
        atexit.register(stop_guarded)
        os.register_at_fork(after_in_child=_forget_all)
        _exit_hooked = True


def _forget_all() -> None:
    global _owner
    _guarded.clear()
    _watchdogs.clear()
    _owner = os.getpid()


def _hook_signals() -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    for number in SIGNALS:
        if number in _previous:
            continue
        before = signal.getsignal(number)
        if before == signal.SIG_IGN:
            continue
        _previous[number] = before
        signal.signal(number, _on_signal)


def _on_signal(number: int, frame: Any) -> None:
    before = _previous.get(number)
    if callable(before):
        before(number, frame)
        return
    stop_guarded()
    signal.signal(number, signal.SIG_DFL)
    os.kill(os.getpid(), number)


def _watch(pid: int, created: float) -> None:
    me = os.getpid()
    argv = [sys.executable, str(WATCHDOG), str(me), str(started_at(me)), str(pid), str(created)]
    try:
        _watchdogs[pid] = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, **process_group_kwargs())
    except OSError as exc:
        logger.warning("no watchdog for server pid %s; it will outlive a killed host: %s",
                       pid, exc)
