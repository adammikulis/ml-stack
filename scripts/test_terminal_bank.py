"""Supervisor-owned bounded PTY descriptors for admitted tests."""
from __future__ import annotations

import array
import json
import os
import secrets
import select
import socket
import threading
import time
from pathlib import Path

from test_kernel_endpoint import private_directory, socket_identity, verify_socket


class TerminalBank:
    """Hold fixed PTY device identities and bounded per-test terminal leases."""

    def __init__(self, count: int, active):
        import pty
        self.token = secrets.token_hex(24)
        self.active = active
        self.guard = threading.Lock()
        self.pairs = [pty.openpty() for _ in range(count)]
        self.paths = tuple(os.ttyname(slave) for _, slave in self.pairs)
        self.identities = tuple(os.fstat(slave).st_rdev for _, slave in self.pairs)
        self.leases: dict[str, tuple[int, float, int]] = {}
        self.directory = private_directory()
        self.endpoint = str(self.directory / "terminal.sock")
        self.used: set[int] = set()
        self.owners: dict[str, str] = {}
        self.transferred: set[str] = set()
        self.stopped = threading.Event()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.endpoint)
        Path(self.endpoint).chmod(0o600)
        self.identity = socket_identity(self.endpoint)
        self.listener.listen(count)
        self.listener.settimeout(.2)
        self.thread = threading.Thread(target=self.transfer)
        self.thread.start()

    def operate(self, request: dict) -> dict:
        with self.guard:
            if request.get("terminal_token") != self.token:
                raise PermissionError("test confinement: invalid terminal bank token")
            operation = request.get("terminal_operation")
            if operation == "allocate":
                used = self.used
                index = next((index for index in range(len(self.pairs)) if index not in used), None)
                if index is None:
                    raise RuntimeError("test confinement: reserved terminal bank exhausted")
                while select.select([self.pairs[index][0]], [], [], 0)[0]:
                    os.read(self.pairs[index][0], 4096)
                self.used.add(index)
                identifier = secrets.token_hex(24)
                self.leases[identifier] = (index, time.monotonic() + 90, 0)
                self.owners[identifier] = request["parent"]
                return {"terminal": identifier, "path": self.paths[index]}
            identifier = request.get("terminal")
            if identifier not in self.leases:
                raise PermissionError("test confinement: unknown terminal lease")
            if self.owners[identifier] != request.get("parent"):
                raise PermissionError("test confinement: foreign terminal admission")
            index, deadline, total = self.leases[identifier]
            if operation == "release":
                del self.leases[identifier]
                del self.owners[identifier]
                return {"released": True}
            if time.monotonic() > deadline or total >= 262144:
                raise RuntimeError("test confinement: terminal lease budget exhausted")
            master = self.pairs[index][0]
            if operation == "read":
                value = os.read(master, 4096) if select.select([master], [], [], .2)[0] else b""
                response = {"data": value.hex()}
            elif operation == "write":
                value = bytes.fromhex(request.get("data", ""))
                if len(value) > 4096:
                    raise ValueError("test confinement: terminal input exceeds bound")
                if not select.select([], [master], [], .2)[1]:
                    raise TimeoutError("test confinement: terminal input timed out")
                response = {"written": os.write(master, value)}
            else:
                raise ValueError("test confinement: unknown terminal operation")
            self.leases[identifier] = (index, deadline, total + len(value))
            return response

    def transfer(self) -> None:
        accepted = 0
        while not self.stopped.is_set() and accepted < 65536:
            verify_socket(self.endpoint, self.identity)
            try:
                connection, _ = self.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                accepted += 1
                connection.settimeout(2)
                try:
                    with connection.makefile("rb") as stream:
                        raw = stream.readline(1025)
                    if len(raw) > 1024:
                        raise ValueError("test confinement: descriptor request exceeds bound")
                    request = json.loads(raw)
                    if not isinstance(request, dict) or set(request) != {"operation", "token", "parent", "terminal"}:
                        raise ValueError("test confinement: invalid descriptor fields")
                    if request["operation"] != "slave" or any(
                            not isinstance(request[key], str) or len(request[key]) != 48
                            or any(c not in "0123456789abcdef" for c in request[key])
                            for key in ("token", "parent", "terminal")):
                        raise ValueError("test confinement: invalid descriptor values")
                    with self.active(request["parent"]), self.guard:
                        if request.get("token") != self.token or request.get("operation") != "slave":
                            raise PermissionError("test confinement: invalid descriptor request")
                        identifier = request["terminal"]
                        if set(request) != {"operation", "token", "parent", "terminal"} or self.owners.get(identifier) != request.get("parent"):
                            raise PermissionError("test confinement: foreign descriptor admission")
                        if identifier in self.transferred:
                            raise PermissionError("test confinement: descriptor already transferred")
                        index, deadline, _ = self.leases[identifier]
                        if time.monotonic() > deadline:
                            raise TimeoutError("test confinement: terminal lease expired")
                        descriptor = array.array("i", [self.pairs[index][1]])
                        connection.sendmsg([b"s"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptor)])
                        self.transferred.add(identifier)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        self.listener.close()

    def close(self) -> None:
        self.stopped.set()
        self.listener.close()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("test confinement: terminal transfer did not stop before deadline")
        try:
            verify_socket(self.endpoint, self.identity)
        finally:
            for pair in self.pairs:
                for fd in pair:
                    os.close(fd)
            self.leases.clear()
            self.owners.clear()
