"""Existence and termination for processes this machine started."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pid_exists(pid: int | None) -> bool:
    """Whether ``pid`` names a process that is still doing something."""
    if not pid or pid <= 0:
        return False
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def kill_pid(pid: int, *, grace_s: float = 1.0) -> None:
    """Terminate ``pid`` gracefully, escalating to kill after ``grace_s``."""
    if not pid_exists(pid):
        return
    import psutil

    try:
        proc = psutil.Process(pid)
    except Exception as exc:
        logger.debug("kill_pid(%s) lookup failed: %s", pid, exc)
        return
    try:
        proc.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except psutil.TimeoutExpired:
        pass
    except psutil.NoSuchProcess:
        return
    try:
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def kill_process_tree(pid: int, *, grace_s: float = 5.0) -> list[int]:
    """Terminate ``pid`` and every descendant, returning the pids acted on."""
    if not pid_exists(pid):
        return []
    import psutil

    try:
        parent = psutil.Process(pid)
        victims = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return []

    for proc in victims:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _gone, alive = psutil.wait_procs(victims, timeout=grace_s)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return [proc.pid for proc in victims]
