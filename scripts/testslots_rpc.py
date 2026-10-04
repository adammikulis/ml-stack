"""Authenticated host admission for pytest collection and test execution."""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import select
import socket
import socketserver
import threading
from collections.abc import Iterator

import testslots


def _read(stream) -> dict:
    value = stream.readline(65537)
    if not value or len(value) > 65536:
        raise ConnectionError("testslots: admission connection closed")
    return json.loads(value)


def _write(stream, value: dict) -> None:
    stream.write(json.dumps(value).encode() + b"\n")
    stream.flush()


class Admission(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = False

    def __init__(self, container: bool):
        self.token = secrets.token_hex(24)
        self.ready = threading.Event()
        self.admitted = threading.Event()
        self.collected = threading.Event()
        self.stopped = threading.Event()
        self.active: set[str] = set()
        self.guard = threading.Lock()
        super().__init__(("0.0.0.0" if container else "127.0.0.1", 0), Handler)  # noqa: S104 - authenticated Docker host endpoint
        host = "host.docker.internal" if container else "127.0.0.1"
        self.endpoint = f"{host}:{self.server_address[1]}"
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def finish(self) -> None:
        self.stopped.set()
        self.shutdown()
        self.server_close()
        self.thread.join()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        if not isinstance(server, Admission):
            raise TypeError("testslots: invalid admission server")
        self.connection.settimeout(5)
        try:
            request = _read(self.rfile)
            if request.get("token") != server.token:
                raise PermissionError("testslots: invalid admission token")
            if request.get("operation") == "ready":
                server.ready.set()
                _write(self.wfile, {"ready": True})
                return
            if request.get("operation") == "collected":
                server.collected.set()
                _write(self.wfile, {"collected": True})
                return
            if request.get("operation") != "acquire":
                raise ValueError("testslots: invalid admission operation")
            with server.guard:
                if request.get("parent") in server.active:
                    raise RuntimeError("testslots: nested CPU run inside a live test lease")
            self.connection.settimeout(None)
            gate = server.admitted if request.get("phase") == "collection" else server.collected
            while not gate.wait(.01):
                self.check_connection()
            export = testslots.EXPORT_LEASE.set(False)
            cancel = testslots.CHECK_CANCELLED.set(self.check_connection)
            try:
                lane = testslots.heavy_lane(str(request.get("label"))) if request.get("heavy") else contextlib.nullcontext()
                with lane, testslots.lease(1, 1, label=str(request.get("label", "pytest"))):
                    identifier = secrets.token_hex(24)
                    with server.guard:
                        server.active.add(identifier)
                    try:
                        _write(self.wfile, {"lease": identifier})
                        while not select.select([self.connection], [], [], .05)[0]:
                            self.check_connection()
                        self.connection.settimeout(5)
                        _read(self.rfile)
                    finally:
                        with server.guard:
                            server.active.discard(identifier)
            finally:
                testslots.CHECK_CANCELLED.reset(cancel)
                testslots.EXPORT_LEASE.reset(export)
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            with contextlib.suppress(OSError):
                _write(self.wfile, {"error": str(error)})

    def check_connection(self):
        if self.server.stopped.is_set():
            raise ConnectionAbortedError("testslots: supervisor stopped")
        if select.select([self.connection], [], [], 0)[0] and not self.connection.recv(1, socket.MSG_PEEK):
            raise ConnectionAbortedError("testslots: worker disconnected")


@contextlib.contextmanager
def request(operation: str, *, label: str = "pytest", phase: str = "test", heavy: bool = False) -> Iterator[dict]:
    host, port = os.environ["DEV_TEST_PYTEST_ENDPOINT"].rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=5) as connection:
        connection.settimeout(float(os.environ.get("DEV_TEST_WAIT_S", "3600")))
        with connection.makefile("rwb") as stream:
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
