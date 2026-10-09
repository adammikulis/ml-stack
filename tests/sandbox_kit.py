"""Shared pieces for the sandbox tests: a fixture that skips without a real Seatbelt, a policy
for running this interpreter, and a loopback listener."""

from __future__ import annotations

import os
import shutil
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from poolhouse import sandbox
from poolhouse.sandbox import AllowUnsandboxed, Limits, Net, Policy
from poolhouse.sandbox.policies import SYSTEM_EXEC
from poolhouse.sandbox.seatbelt import Seatbelt


@pytest.fixture
def seatbelt() -> Seatbelt:
    """The real Seatbelt backend, or a skip where ``sandbox-exec`` does not exist."""
    backend = Seatbelt()
    state = backend.available()
    if not state.ok:
        pytest.skip(state.reason)
    return backend


def no_sandbox_here() -> AllowUnsandboxed | None:
    """None where a sandbox exists; elsewhere the named opt-out, so a test of something else
    can still start its server."""
    if sandbox.backend().available().ok:
        return None
    return AllowUnsandboxed("this test host has no sandbox and the test is not about one")


def require_native_sandbox() -> None:
    """Skip native success tests when the host cannot run a confined process."""
    state = sandbox.backend().available()
    if not state.ok:
        pytest.skip(state.reason)
    executable = os.path.realpath(shutil.which("true") or "/bin/true")
    held = Policy("native-probe", exec=(executable,), env={},
                  limits=Limits(wall_seconds=3, output_bytes=4096))
    result = sandbox.run([executable], held, diagnose="never")
    error = result.stderr.lower()
    namespace_denied = "namespace" in error and any(
        message in error for message in ("operation not permitted", "permission denied",
                                        "not allow non-privileged user namespaces"))
    if not result.ok and namespace_denied:
        pytest.skip(f"native sandbox namespace unavailable: {result.stderr.strip()[:300]}")
    assert result.ok, f"native sandbox probe failed: {result.stderr}"


def runtime_reads() -> list[str]:
    """What this interpreter needs read access to: its prefixes and Homebrew's libraries."""
    found = {os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)}
    if Path("/opt/homebrew").is_dir():
        found.add("/opt/homebrew")
    return sorted(found)


def policy(*, read: list[Path | str] = (), write: list[Path | str] = (), net: Net | None = None,  # noqa: PLR0913
           env: dict[str, str] | None = None, limits: Limits | None = None,
           exec_: list[str] = (), python: bool = False,
           name: str = "test") -> Policy:
    """A policy that reads ``read``, writes ``write`` and runs the system tools; with
    ``python`` it also runs this interpreter."""
    runs = [*SYSTEM_EXEC, *exec_]
    reads = [str(p) for p in read]
    if python:
        runs.append(os.path.realpath(sys.executable))
        reads += runtime_reads()
    return Policy(name, read=tuple(reads), write=tuple(str(p) for p in write), exec=tuple(runs),
                  net=net or Net.deny(), env={"PATH": "/usr/bin:/bin", **(env or {})},
                  limits=limits or Limits(wall_seconds=30))


class Listener:
    """A loopback server on an ephemeral port that answers ``pong`` to every connection."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.accepted += 1
            conn.sendall(b"pong")
            conn.close()

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def listener() -> Iterator[Listener]:
    server = Listener()
    yield server
    server.close()


CONNECT = """
import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(("127.0.0.1", {port}))
    print("connected", s.recv(4).decode())
except OSError as e:
    print("refused", e.errno)
"""
