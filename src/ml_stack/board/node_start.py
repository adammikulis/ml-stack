"""Find the node binary and start the node when its socket is dead (single-flight).

This is the interim launcher: `ml_stack.node_launch` replaces it, keeping `binary` and `ensure`.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

from ml_stack import platform
from ml_stack.lock import only_one

__all__ = ["BIN_ENV", "NodeUnavailable", "answers", "binary", "ensure", "socket_path"]

BIN_ENV = "ML_STACK_NODE_BIN"
NAME = "poolside-node"
WAIT_S = 15.0


class NodeUnavailable(OSError):
    """There is no node binary to start, or the node did not answer after being started."""


def socket_path(state: Path) -> Path:
    """Where the node of ``state`` listens."""
    return state / "node.sock"


def binary() -> Path:
    """The node binary: the one named by $ML_STACK_NODE_BIN, one shipped in the package, or this
    checkout's cargo build (release before debug)."""
    named = os.environ.get(BIN_ENV)
    here = Path(__file__).resolve().parent
    candidates = [Path(named).expanduser()] if named else [
        here.parent / "bin" / NAME, *(here.parents[2] / "app" / "target" / kind / NAME for kind in ("release", "debug"))]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise NodeUnavailable(f"no {NAME} binary ({', '.join(str(p) for p in candidates)}); "
                          f"run `cargo build -p {NAME}` in app/ or set ${BIN_ENV}")


def answers(state: Path) -> bool:
    """Whether a node listens on the socket of ``state``."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(2.0)
    try:
        probe.connect(str(socket_path(state)))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def ensure(state: Path) -> None:
    """Make sure a node runs on ``state``, starting one when nothing answers. Callers queue on a
    lock file, so the second finds the first's node instead of starting another."""
    if answers(state):
        return
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with only_one(state / "start.lock", timeout=WAIT_S, announce=lambda *_: None):
        if answers(state):
            return
        platform.start_process([str(binary()), "run", "--state", str(state)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            if answers(state):
                return
            time.sleep(0.02)
    raise NodeUnavailable("the node did not answer after being started")
