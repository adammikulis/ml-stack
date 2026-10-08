"""Authenticated host admission for pytest collection and test execution."""
from __future__ import annotations

import array
import contextlib
import json
import math
import os
import secrets
import select
import socket
import socketserver
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import testslots
from test_kernel_endpoint import connection, private_directory, socket_identity, verify_socket


def _read(stream) -> dict:
    value = stream.readline(65537)
    if not value or len(value) > 65536:
        raise ConnectionError("testslots: admission connection closed")
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("testslots: admission request must be an object")
    return result


def identifier(value) -> bool:
    return isinstance(value, str) and len(value) == 48 and all(c in "0123456789abcdef" for c in value)


def validate_release(value: dict) -> None:
    if set(value) != {"release"} or type(value["release"]) is not bool or value["release"] is not True:
        raise ValueError("testslots: invalid lease release")


def validate_request(value: dict) -> None:
    operation = value.get("operation")
    if not isinstance(operation, str):
        raise ValueError("testslots: invalid operation type")
    if not identifier(value.get("token")):
        raise ValueError("testslots: invalid token shape")
    if operation == "terminal":
        fields = {"operation", "token", "parent", "terminal_token", "terminal_operation"}
        action = value.get("terminal_operation")
        if not isinstance(action, str):
            raise ValueError("testslots: invalid terminal operation type")
        if action in {"read", "write", "release"}:
            fields.add("terminal")
            if not identifier(value.get("terminal")):
                raise ValueError("testslots: invalid terminal identifier")
        if action == "write":
            fields.add("data")
            data = value.get("data")
            if not isinstance(data, str) or len(data) > 8192 or len(data) % 2 or any(c not in "0123456789abcdef" for c in data):
                raise ValueError("testslots: invalid bounded terminal data")
        if action not in {"allocate", "read", "write", "release"} or not identifier(value.get("terminal_token")):
            raise ValueError("testslots: invalid terminal operation")
        if not identifier(value.get("parent")):
            raise ValueError("testslots: terminal caller requires admission")
    else:
        fields = {"operation", "token", "parent", "label", "phase", "heavy"}
        if operation not in {"ready", "collected", "acquire", "ping"}:
            raise ValueError("testslots: invalid admission operation")
        if value.get("parent") is not None and not identifier(value.get("parent")):
            raise ValueError("testslots: invalid parent identifier")
        if not isinstance(value.get("label"), str) or len(value["label"]) > 256:
            raise ValueError("testslots: invalid bounded label")
        if value.get("phase") not in ("collection", "test") or type(value.get("heavy")) is not bool:
            raise ValueError("testslots: invalid admission settings")
    if set(value) != fields:
        raise ValueError("testslots: invalid admission fields")


def _write(stream, value: dict) -> None:
    stream.write(json.dumps(value).encode() + b"\n")
    stream.flush()


class Admission(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = False
    block_on_close = False

    def __init__(self, container: bool):
        self.terminals = None
        self.token = secrets.token_hex(24)
        self.ready = threading.Event()
        self.admitted = threading.Event()
        self.collected = threading.Event()
        self.stopped = threading.Event()
        self.active: set[str] = set()
        self.active_resources: dict[str, frozenset[str]] = {}
        self.fixture_resources = None
        self.guard = threading.Lock()
        self.connections: set[socket.socket] = set()
        self.running_threads: set[threading.Thread] = set()
        self.handlers = threading.BoundedSemaphore(128)
        self.request_count = 0
        self.wait_limit = float(os.environ.get("DEV_TEST_WAIT_S", "3600"))
        if not math.isfinite(self.wait_limit) or self.wait_limit <= 0:
            raise ValueError("testslots: finite positive wait required")
        address = self.address(container)
        super().__init__(address, Handler)
        host = "host.docker.internal" if container else "127.0.0.1"
        self.endpoint = "unix:" + self.server_address if self.address_family == socket.AF_UNIX else f"{host}:{self.server_address[1]}"
        self.identity = socket_identity(self.server_address) if self.address_family == socket.AF_UNIX else None
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def address(self, container):
        if self.address_family == socket.AF_UNIX:
            self.directory = private_directory()
            return str(self.directory / "admission.sock")
        return ("0.0.0.0" if container else "127.0.0.1", 0)  # noqa: S104 - authenticated Docker host endpoint

    def server_bind(self):
        super().server_bind()
        if self.address_family == socket.AF_UNIX:
            Path(self.server_address).chmod(0o600)

    def check_identity(self):
        if self.identity is not None:
            verify_socket(self.server_address, self.identity)

    def get_request(self):
        self.check_identity()
        stream, address = super().get_request()
        try:
            self.check_identity()
            stream.settimeout(5)
            with self.guard:
                if self.stopped.is_set():
                    raise ConnectionAbortedError("testslots: supervisor stopped")
                self.connections.add(stream)
            return stream, address
        except BaseException:
            stream.close()
            raise

    def shutdown_request(self, request):
        with self.guard:
            self.connections.discard(request)
        super().shutdown_request(request)

    def process_request(self, request, client_address):
        with self.guard:
            allowed = self.request_count < 65536 and self.handlers.acquire(blocking=False)
            if allowed:
                self.request_count += 1
        if not allowed:
            self.shutdown_request(request)
            return
        thread = threading.Thread(target=self.process_request_thread, args=(request, client_address),
                                  daemon=self.daemon_threads)
        with self.guard:
            self.running_threads.add(thread)
        try:
            thread.start()
        except BaseException:
            self.handlers.release()
            with self.guard:
                self.running_threads.discard(thread)
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request, client_address):
        current = threading.current_thread()
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.handlers.release()
            with self.guard:
                self.running_threads.discard(current)

    def finish(self) -> None:
        self.stopped.set()
        self.shutdown()
        with self.guard:
            connections = tuple(self.connections)
            threads = tuple(self.running_threads)
        for stream in connections:
            with contextlib.suppress(OSError):
                stream.shutdown(socket.SHUT_RDWR)
            stream.close()
        self.server_close()
        deadline = time.monotonic() + 10
        for thread in (*threads, self.thread):
            thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in (*threads, self.thread)):
            raise RuntimeError("testslots: admission handlers did not stop before deadline")
        self.check_identity()

    @contextlib.contextmanager
    def terminal_admission(self, identifier: str):
        with self.guard:
            if identifier not in self.active:
                raise PermissionError("testslots: inactive terminal admission")
            yield


class Handler(socketserver.StreamRequestHandler):
    def authenticated_request(self, server: Admission) -> dict:
        if server.stopped.is_set():
            raise ConnectionAbortedError("testslots: supervisor stopped")
        self.connection.settimeout(5)
        self.deadline = time.monotonic() + server.wait_limit
        request = _read(self.rfile)
        validate_request(request)
        if request["token"] != server.token:
            raise PermissionError("testslots: invalid admission token")
        return request

    def answered(self, server: Admission, request: dict) -> bool:
        operation = request.get("operation")
        if operation == "terminal":
            self.terminal(server, request)
            return True
        flags = {"ping": None, "ready": server.ready, "collected": server.collected}
        if operation not in flags:
            return False
        if flags[operation] is not None:
            flags[operation].set()
        _write(self.wfile, {operation: True})
        return True

    def handle(self):
        server = self.server
        if not isinstance(server, Admission):
            raise TypeError("testslots: invalid admission server")
        try:
            request = self.authenticated_request(server)
            if self.answered(server, request):
                return
            if request.get("operation") != "acquire":
                raise ValueError("testslots: invalid admission operation")
            with server.guard:
                if request.get("parent") in server.active:
                    raise RuntimeError("testslots: nested CPU run inside a live test lease")
            gate = server.admitted if request.get("phase") == "collection" else server.collected
            while not gate.wait(.01):
                self.check_connection()
            export = testslots.EXPORT_LEASE.set(False)
            cancel = testslots.CHECK_CANCELLED.set(self.check_connection)
            try:
                lane = testslots.heavy_lane(str(request.get("label"))) if request.get("heavy") else contextlib.nullcontext()
                with lane, testslots.lease(1, 1, label=str(request.get("label", "pytest")), say=lambda message: None):
                    identifier = secrets.token_hex(24)
                    with server.guard:
                        granted = server.fixture_resources(request["label"], request["phase"]) if server.fixture_resources is not None else frozenset()
                        server.active_resources[identifier] = granted
                        server.active.add(identifier)
                    try:
                        self.connection.settimeout(5)
                        _write(self.wfile, {"lease": identifier})
                        while not select.select([self.connection], [], [], .05)[0]:
                            self.check_connection()
                        self.connection.settimeout(5)
                        validate_release(_read(self.rfile))
                    finally:
                        with server.guard:
                            server.active.discard(identifier)
                            server.active_resources.pop(identifier, None)
            finally:
                testslots.CHECK_CANCELLED.reset(cancel)
                testslots.EXPORT_LEASE.reset(export)
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            with contextlib.suppress(OSError):
                _write(self.wfile, {"error": str(error)})

    def terminal(self, server: Admission, request: dict) -> None:
        if server.terminals is None:
            raise PermissionError("testslots: terminal bank unavailable")
        with server.terminal_admission(request["parent"]):
            response = server.terminals.operate(request)
        _write(self.wfile, response)

    def check_connection(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("testslots: admission caller wait expired")
        if self.server.stopped.is_set():
            raise ConnectionAbortedError("testslots: supervisor stopped")
        if select.select([self.connection], [], [], 0)[0] and not self.connection.recv(1, socket.MSG_PEEK):
            raise ConnectionAbortedError("testslots: worker disconnected")


@contextlib.contextmanager
def request(operation: str, *, label: str = "pytest", phase: str = "test", heavy: bool = False) -> Iterator[dict]:
    with connection(os.environ["DEV_TEST_PYTEST_ENDPOINT"], os.environ.get("DEV_TEST_PYTEST_IDENTITY")) as channel:
        channel.settimeout(float(os.environ.get("DEV_TEST_WAIT_S", "3600")))
        with channel.makefile("rwb") as stream:
            _write(stream, {"operation": operation, "token": os.environ["DEV_TEST_PYTEST_TOKEN"],
                            "parent": os.environ.get("DEV_TEST_REMOTE_LEASE"), "label": label, "phase": phase, "heavy": heavy})
            response = _read(stream)
            if "error" in response:
                raise RuntimeError(response["error"])
            previous = os.environ.get("DEV_TEST_REMOTE_LEASE")
            if "lease" in response:
                os.environ["DEV_TEST_REMOTE_LEASE"] = response["lease"]
            try:
                yield response
            finally:
                if previous is None:
                    os.environ.pop("DEV_TEST_REMOTE_LEASE", None)
                else:
                    os.environ["DEV_TEST_REMOTE_LEASE"] = previous
                if "lease" in response:
                    _write(stream, {"release": True})


def terminal_request(operation: str, **fields) -> dict:
    with connection(os.environ["DEV_TEST_PYTEST_ENDPOINT"], os.environ.get("DEV_TEST_PYTEST_IDENTITY")) as channel, channel.makefile("rwb") as stream:
        _write(stream, {"operation": "terminal", "token": os.environ["DEV_TEST_PYTEST_TOKEN"],
                        "parent": os.environ.get("DEV_TEST_REMOTE_LEASE"),
                        "terminal_token": os.environ["DEV_TEST_PTY_TOKEN"],
                        "terminal_operation": operation, **fields})
        response = _read(stream)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

def terminal_slave(identifier: str) -> int:
    verify_socket(os.environ["DEV_TEST_PTY_ENDPOINT"], json.loads(os.environ["DEV_TEST_PTY_IDENTITY"]))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(os.environ["DEV_TEST_PTY_ENDPOINT"])
        verify_socket(os.environ["DEV_TEST_PTY_ENDPOINT"], json.loads(os.environ["DEV_TEST_PTY_IDENTITY"]))
        connection.sendall(json.dumps({"operation": "slave", "token": os.environ["DEV_TEST_PTY_TOKEN"],
                                       "parent": os.environ.get("DEV_TEST_REMOTE_LEASE"),
                                       "terminal": identifier}).encode() + b"\n")
        message, ancillary, flags, _ = connection.recvmsg(1, socket.CMSG_SPACE(array.array("i").itemsize))
        descriptors = array.array("i")
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                descriptors.frombytes(data[:len(data) - len(data) % descriptors.itemsize])
        if message != b"s" or flags & socket.MSG_CTRUNC or len(descriptors) != 1:
            for fd in descriptors:
                os.close(fd)
            raise RuntimeError("test confinement: invalid reserved descriptor response")
        os.set_inheritable(descriptors[0], False)
        return descriptors[0]


class UnixAdmission(Admission):
    address_family = socket.AF_UNIX
    allow_reuse_address = False
