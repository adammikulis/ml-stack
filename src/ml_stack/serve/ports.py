"""Is this port free, and if not, is the thing holding it ours?"""

from __future__ import annotations

import logging
import socket

from ml_stack.serve.process import kill_pid, pid_exists

logger = logging.getLogger(__name__)

SERVER_BINARIES = ("llama-server", "llama_server", "llama-server.exe")
"""Executable names of the model servers this library starts."""


DEFAULT_HOST = "127.0.0.1"
"""The interface servers are bound to. Loopback: a local model server on 0.0.0.0 is an"""


def port_is_free(port: int, host: str = DEFAULT_HOST) -> bool:
    """Whether ``port`` can be bound on ``host``. A pure socket check, no state."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def free_port(host: str = DEFAULT_HOST) -> int:
    """Ask the OS for an unused port. For tests and ephemeral servers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def server_pids_on_port(port: int) -> list[int]:
    """Pids of *our* model-server binaries listening on ``port``."""
    try:
        import psutil
    except ImportError:
        return []

    pids: list[int] = []
    for process in psutil.process_iter(["pid", "name"]):
        name = process.info.get("name") or ""
        if not any(binary in name for binary in SERVER_BINARIES):
            continue
        try:
            connections = process.net_connections(kind="inet")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for conn in connections:
            if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                pids.append(process.info["pid"])
                break
    return pids


def reclaim_port(port: int, *, recorded_pids: list[int] | None = None) -> bool:
    """Free ``port`` when it is held by a model server *this* manager recorded. Returns
    whether it was.

    Only a recorded pid, never whatever llama-server happens to hold the port. The day
    this was learned (2026-09-03): a test started a fake server on the default port with
    the real backend, the port was busy, and the old "unrecorded server" branch killed the
    Flash-Next that had been reading textbooks for twelve hours -- mid-request, from
    a process that had never leased it. A server another process started is that
    process's to stop; here it is reported as held by something that is not ours, and the
    caller refuses to kill it (the same rule the bash guard enforces on shells:
    `pkill llama-server` is not a way to free a port).
    """
    reclaimed = False
    for pid in recorded_pids or []:
        if pid_exists(pid):
            logger.info("reclaiming port %s from our own stale server (pid %s)", port, pid)
            kill_pid(pid)
            reclaimed = True
    if not reclaimed and server_pids_on_port(port):
        logger.info("port %s is held by a model server this manager did not start; "
                    "leaving it alone", port)
    return reclaimed
