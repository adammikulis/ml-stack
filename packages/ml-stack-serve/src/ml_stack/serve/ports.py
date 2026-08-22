"""Is this port free, and if not, is the thing holding it ours?"""

from __future__ import annotations

import logging
import socket

from ml_stack.serve.process import kill_pid, pid_exists

logger = logging.getLogger(__name__)

SERVER_BINARIES = ("llama-server", "llama_server", "llama-server.exe")
"""Executable names of the model servers this library starts.

Matching on the name is what makes reclaiming a port safe: a foreign service that happens
to hold the port is refused rather than killed.
"""


DEFAULT_HOST = "127.0.0.1"
"""The interface servers are bound to. Loopback: a local model server on 0.0.0.0 is an
unauthenticated inference endpoint on every network the machine is attached to."""


def port_is_free(port: int, host: str = DEFAULT_HOST) -> bool:
    """Whether ``port`` can be bound on ``host``. A pure socket check, no state.

    ``host`` must be the address the server will actually bind. Probing the wildcard
    address instead is a silent false negative: a listener bound to a specific interface
    does not always block a wildcard bind, so a port genuinely in use reads as free and
    the start fails later with a much less obvious error.

    ``SO_REUSEADDR`` is set to match how the server itself binds. It does not permit two
    live listeners on one address -- that would be ``SO_REUSEPORT`` -- but it does allow
    binding over a socket in TIME_WAIT. That distinction is the point: a socket just
    killed lingers in TIME_WAIT for seconds after its process is gone, and a probe
    without this option reports that as "still in use", indistinguishable from a live
    foreign listener. A lease that had already stopped the old server would then see a
    busy port with no live process left to reclaim, and refuse a start that would in fact
    have succeeded immediately.
    """
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
    """Pids of *our* model-server binaries listening on ``port``.

    Processes are matched by executable name *before* their sockets are read: a
    machine-wide connection scan needs root on macOS and raises on the first process this
    user does not own, so filtering first is what keeps this usable unprivileged.
    """
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
    """Free ``port`` when it is held by a model server we started. Returns whether it was.

    Ownership has two proofs and either suffices. The state file is the cheaper one, but
    it is cleared on shutdown and never written for a server that died before its first
    save -- so a recorded pid cannot be the only evidence, or a port held by an
    unrecorded orphan stays blocked forever and only a human with ``kill`` can clear it.
    The second proof is the process itself: a model-server binary listening on a port
    this machine assigns to a model server is ours by construction.
    """
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
