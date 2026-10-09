"""A guard that refuses every socket connection that leaves this machine."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["EgressRefused", "local_only"]


class EgressRefused(OSError):
    """A connection to an address that is not on this machine."""


def _local(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:
        return True
    host = str(address[0])
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


@contextmanager
def local_only() -> Iterator[None]:
    """Inside the block, connecting to anything but a loopback address or a unix socket
    raises `EgressRefused` before a packet is sent."""
    own = {name: socket.socket.__dict__.get(name) for name in ("connect", "connect_ex")}
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def checked(self: socket.socket, address: Any) -> None:
        if not _local(address):
            raise EgressRefused(f"red-team traffic stays on this machine, not {address!r}")
        return connect(self, address)

    def checked_ex(self: socket.socket, address: Any) -> int:
        if not _local(address):
            raise EgressRefused(f"red-team traffic stays on this machine, not {address!r}")
        return connect_ex(self, address)

    socket.socket.connect, socket.socket.connect_ex = checked, checked_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        for name, kept in own.items():
            if kept is None:
                delattr(socket.socket, name)
            else:
                setattr(socket.socket, name, kept)
