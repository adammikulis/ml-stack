"""The node's local API: length-prefixed JSON over a Unix socket (docs/node.md, "Local API")."""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any

from ml_stack import node_launch
from ml_stack.node_health import API_VERSION, MAX_FRAME, socket_path

__all__ = ["Client", "Conflict", "Denied", "Invalid", "NodeError", "Quota"]



class NodeError(Exception):
    """The node refused a request; ``code`` is its word for why."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Denied(NodeError):
    """The caller may not do that."""


class Invalid(NodeError):
    """The request is malformed or names something that is not there."""


class Quota(NodeError):
    """Something past what the node keeps."""


class Conflict(Denied):
    """Another session holds what a claim asked for."""


def _error(reply: dict[str, Any]) -> NodeError:
    code, message = reply["error"]["code"], str(reply["error"]["message"])
    kind = {"denied": Denied, "invalid": Invalid, "quota": Quota}.get(code, NodeError)
    return kind(code, message)


class Client:
    """One node, reached on its socket; the node is started first when the socket is dead."""

    def __init__(self, state: Path | None = None) -> None:
        self.state = state or node_launch.default_state()
        self.requests = 0

    def call(self, method: str, board: str = "", token: str = "", **params: Any) -> Any:
        """Send one request and return its result; a refusal raises `NodeError`."""
        self.requests += 1
        request: dict[str, Any] = {"v": API_VERSION, "id": self.requests, "method": method, "params": params}
        if board:
            request["board"] = board
        if token:
            request["token"] = token
        reply = self._exchange(request)
        if reply.get("ok") is not True:
            raise _error(reply)
        return reply["result"]

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        node_launch.ensure_node(self.state)
        body = json.dumps(request).encode()
        if len(body) > MAX_FRAME:
            raise Quota("quota", "the request is larger than a frame may be")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(660.0)
            stream.connect(str(socket_path(self.state)))
            stream.sendall(struct.pack(">I", len(body)) + body)
            size = struct.unpack(">I", _take(stream, 4))[0]
            if size > MAX_FRAME:
                raise NodeError("invalid", "the node sent a frame larger than a frame may be")
            return json.loads(_take(stream, size))


def _take(stream: socket.socket, count: int) -> bytes:
    out = b""
    while len(out) < count:
        chunk = stream.recv(count - len(out))
        if not chunk:
            raise OSError("the node closed the connection")
        out += chunk
    return out
