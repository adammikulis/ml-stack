"""Pinned private Unix admission endpoints."""
from __future__ import annotations

import json
import os
import socket
import stat
import time
from pathlib import Path


def private_directory() -> Path:
    from poolhouse.activity.source_snapshot import private_namespace
    return private_namespace(dict(os.environ), "poolhouse-admission-")


def socket_identity(path: str) -> tuple[int, int, int]:
    node = Path(path)
    parent = node.parent.lstat()
    info = node.lstat()
    if (node.parent.resolve() != node.parent or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid() or parent.st_mode & 0o077
            or not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise PermissionError("test confinement: endpoint is not an owned private socket")
    return info.st_dev, info.st_ino, info.st_uid


def verify_socket(path: str, expected) -> None:
    if socket_identity(path) != tuple(expected):
        raise PermissionError("test confinement: endpoint identity changed")


PATIENCE = 120.0
"""Seconds a caller waits for a busy admission server before it gives up."""
ATTEMPT = 5.0
"""Seconds one connect may take; a full accept queue shows up as a connect that times out."""


def open_connection(endpoint: str, identity: str | None, attempt: float) -> socket.socket:
    if endpoint.startswith("unix:"):
        path = endpoint[5:]
        if identity is None:
            raise PermissionError("test confinement: endpoint identity is missing")
        value = json.loads(identity)
        if not isinstance(value, list) or len(value) != 3 or any(type(part) is not int or part < 0 for part in value):
            raise PermissionError("test confinement: invalid endpoint identity")
        verify_socket(path, value)
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            stream.settimeout(attempt)
            stream.connect(path)
            verify_socket(path, value)
            return stream
        except BaseException:
            stream.close()
            raise
    host, port = endpoint.rsplit(":", 1)
    return socket.create_connection((host, int(port)), timeout=attempt)


class AdmissionUnreachable(TimeoutError):
    """The admission server accepted no connection within the patience; it is busy or gone."""


def connection(endpoint: str, identity: str | None = None, *, patience: float | None = None,
               attempt: float = ATTEMPT) -> socket.socket:
    """Connect to an admission endpoint, waiting for a busy server up to ``patience`` seconds.

    Only a connect that times out is retried (a full accept queue); a refusal means nothing is
    listening and a failed identity check is a refusal of the endpoint, and both fail at once.
    ``DEV_TEST_CONNECT_S`` overrides the default patience."""
    if patience is None:
        patience = float(os.environ.get("DEV_TEST_CONNECT_S", PATIENCE))
    deadline = time.monotonic() + patience
    delay = 0.1
    while True:
        try:
            return open_connection(endpoint, identity, max(0.05, min(attempt, deadline - time.monotonic())))
        except TimeoutError as exc:
            if time.monotonic() + delay >= deadline:
                raise AdmissionUnreachable(f"the test admission endpoint {endpoint} accepted no connection in "
                                   f"{patience:g} s: the queue of runs ahead is too long") from exc
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
