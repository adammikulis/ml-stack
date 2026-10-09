"""Ask a node on its local socket whether it is alive: one framed `hello`, nothing else it can be refused for."""

from __future__ import annotations

import json
import socket
import struct
import time
from pathlib import Path

SOCKET = "node.sock"
API_VERSION = 1
TIMEOUT_S = 2.0
MAX_FRAME = 1 << 20


def socket_path(state: Path) -> Path:
    """Where the node of a state directory listens."""
    return state / SOCKET


def read_exactly(stream: socket.socket, count: int) -> bytes:
    """Exactly ``count`` bytes from ``stream``; ConnectionError when the node closes first."""
    data = b""
    while len(data) < count:
        block = stream.recv(count - len(data))
        if not block:
            raise ConnectionError("the node closed the connection")
        data += block
    return data


def call(state: Path, method: str, params: dict | None = None, *, board: str = "", token: str = "") -> dict:
    """One request to the node of ``state``; the result, or OSError/ValueError when it is dead or refuses."""
    request = {"v": API_VERSION, "id": 1, "method": method, "params": params or {}}
    request.update({key: value for key, value in (("board", board), ("token", token)) if value})
    body = json.dumps(request).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(TIMEOUT_S)
        stream.connect(str(socket_path(state)))
        stream.sendall(struct.pack(">I", len(body)) + body)
        (size,) = struct.unpack(">I", read_exactly(stream, 4))
        if size > MAX_FRAME:
            raise ValueError("the node sent an oversized frame")
        reply = json.loads(read_exactly(stream, size))
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
