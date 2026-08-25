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
    """Free ``port`` when it is held by a model server we started. Returns whether it was."""
    reclaimed = False

    for pid in recorded_pids or []:
        if pid_exists(pid):
            logger.info("reclaiming port %s from our own stale server (pid %s)", port, pid)
            kill_pid(pid)
            reclaimed = True

    if not reclaimed:
        for pid in server_pids_on_port(port):
            logger.info("reclaiming port %s from an unrecorded server (pid %s)", port, pid)
            kill_pid(pid)
            reclaimed = True

    return reclaimed
