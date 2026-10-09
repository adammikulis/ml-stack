"""Ask a node on its local socket (a named pipe on Windows) whether it is alive: one framed `hello`, nothing else it can be refused for."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ml_stack import win32
from ml_stack.platform import is_windows

SOCKET = "node.sock"
API_VERSION = 1
TIMEOUT_S = 2.0
MAX_FRAME = 1 << 20


def key_of(text: str) -> str:
    """The 32 hex characters that name a state directory's pipe and stop event: the SHA-256 of its absolute path with
    backslashes, lower case, no `\\\\?\\` prefix and no trailing separator. `poolside-node`'s `sys::key_of` computes the same."""
    text = text.replace("/", "\\")
    text = text[4:] if text.startswith("\\\\?\\") else text
    return hashlib.sha256(text.rstrip("\\").lower().encode("utf-8")).hexdigest()[:32]


def state_key(state: Path) -> str:
    """`key_of` the absolute path of a state directory."""
    return key_of(os.path.normpath(Path(state).absolute()))


def pipe_name(state: Path) -> str:
    """The named pipe the node of a state directory serves on Windows."""
    return f"\\\\.\\pipe\\poolside-node-{state_key(state)}"


def node_stop_event(state: Path) -> str:
    """The named event whose signal makes the node of a state directory exit cleanly on Windows."""
    return f"Local\\poolside-node-stop-{state_key(state)}"


def socket_path(state: Path) -> Path:
    """Where the node of a state directory listens: a socket file, or on Windows a pipe name."""
    return Path(pipe_name(state)) if is_windows() else state / SOCKET


def _read(recv: Callable[[int], bytes], count: int) -> bytes:
    data = b""
    while len(data) < count:
        block = recv(count - len(data))
        if not block:
            raise ConnectionError("the node closed the connection")
        data += block
    return data


def _exchange_socket(state: Path, frame: bytes, timeout: float) -> bytes:
    family = getattr(socket, "AF_UNIX", None)
    if family is None:  # a Python without Unix sockets: no node can answer here
        raise OSError("this Python has no Unix sockets")
    with socket.socket(family, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        stream.connect(str(socket_path(state)))
        stream.sendall(frame)
        return _reply(stream.recv)


def _reply(recv: Callable[[int], bytes]) -> bytes:
    (size,) = struct.unpack(">I", _read(recv, 4))
    if size > MAX_FRAME:
        raise ValueError("the node sent an oversized frame")
    return _read(recv, size)


def _talk_pipe(state: Path, frame: bytes, out: dict, timeout: float) -> None:
    name, until = pipe_name(state), time.monotonic() + timeout
    try:
        while True:
            try:
                pipe = win32.open_pipe(name)
                break
            except OSError as exc:
                if getattr(exc, "winerror", 0) != win32.ERROR_PIPE_BUSY or time.monotonic() > until:
                    raise
                time.sleep(0.02)
        with pipe:
            pipe.write(frame)
            out["reply"] = _reply(pipe.read)
    except (OSError, ValueError) as exc:
        out["error"] = exc


def _exchange_pipe(state: Path, frame: bytes, timeout: float) -> bytes:
    """A pipe read has no timeout, so the exchange runs in a thread that is given ``timeout`` and then abandoned."""
    out: dict = {}
    worker = threading.Thread(target=_talk_pipe, args=(state, frame, out, timeout), daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError("the node did not answer in time")
    if "error" in out:
        raise out["error"]
    return out["reply"]


def exchange(state: Path, body: bytes, timeout: float = TIMEOUT_S) -> bytes:
    """One framed request body to the node of ``state`` and the body of its reply."""
    return (_exchange_pipe if is_windows() else _exchange_socket)(state, struct.pack(">I", len(body)) + body, timeout)


def call(state: Path, method: str, params: dict | None = None, *, board: str = "", token: str = "") -> dict:
    """One request to the node of ``state``; the result, or OSError/ValueError when it is dead or refuses."""
    request = {"v": API_VERSION, "id": 1, "method": method, "params": params or {}}
    request.update({key: value for key, value in (("board", board), ("token", token)) if value})
    body = json.dumps(request).encode()
    reply = json.loads(exchange(state, body))
    if reply.get("ok") is not True:
        raise ValueError(str((reply.get("error") or {}).get("message", "refused")))
    return reply["result"]


def node_health(state: Path) -> dict | None:
    """The node's `hello` (node, version, pid, fingerprint) with the socket and the round trip, or None when it does not answer."""
    began = time.monotonic()
    try:
        said = call(state, "hello")
    except (OSError, ValueError):
        return None
    return {**said, "socket": str(socket_path(state)), "latency_ms": round((time.monotonic() - began) * 1000, 1)}
