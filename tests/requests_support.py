"""Fixtures for the request-inbox tests: an isolated home, a keyring that survives into child
processes (a JSON file), a provisioned master key and no agent markers."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend

from poolhouse import keystore
from poolhouse.http import Server
from poolhouse.inbox.route import MAX_BODY, RequestsApp
from poolhouse.person import AGENT_MARKERS
from poolhouse.requests import Ask, Origin

HERE = str(Path(__file__).resolve().parent)
SRC = str(Path(__file__).resolve().parent.parent / "src")
CANARY = "canary-request-subject-5d1f"


class FileRing(KeyringBackend):
    """The keyring interface over a JSON file named by ``$PCBE_TEST_RING``, so a child process sees it."""

    priority = 1  # type: ignore[assignment]

    def _path(self) -> Path:
        return Path(os.environ["POOLHOUSE_TEST_RING"])

    def _held(self) -> dict[str, str]:
        try:
            return json.loads(self._path().read_text())
        except (OSError, ValueError):
            return {}

    def get_password(self, service, username):
        return self._held().get(f"{service}/{username}")

    def set_password(self, service, username, password):
        held = self._held()
        held[f"{service}/{username}"] = password
        self._path().write_text(json.dumps(held))

    def delete_password(self, service, username):
        held = self._held()
        held.pop(f"{service}/{username}", None)
        self._path().write_text(json.dumps(held))


@pytest.fixture
def no_markers(monkeypatch):
    """A person's process: no agent marker in the environment."""
    for name in AGENT_MARKERS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def person_home(monkeypatch, tmp_path):
    """A person's process with a provisioned keystore under ``tmp_path``; returns the child environment."""
    for name in AGENT_MARKERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POOLHOUSE_TEST_RING", str(tmp_path / "ring.json"))
    before = keyring.get_keyring()
    keyring.set_keyring(FileRing())
    keystore.default().provision()
    yield {**os.environ, "PYTHONPATH": os.pathsep.join([SRC, HERE, "."]),
           "PYTHON_KEYRING_BACKEND": "requests_support.FileRing", "POOLHOUSE_HOME": os.environ["POOLHOUSE_HOME"],
           "POOLHOUSE_TEST_RING": str(tmp_path / "ring.json"), "POOLHOUSE_NOTIFY": "off"}
    keyring.set_keyring(before)


def ask(subject: str = "run_shell(command='ls')", *, kind: str = "tool_call", agent: str = "agent-a",
        project: str = "proj", choices=("allow-once", "deny"), **more) -> Ask:
    return Ask(kind, subject, "Asked because: reversible: runs a program.", tuple(choices),
               Origin(agent, project, "s1"), **more)


class Tty:
    """A stream that says it is a terminal."""

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None




class Handler(BaseHTTPRequestHandler):
    """``http.server`` in front of a `RequestsApp`, the way a shell's server mounts one."""

    app: RequestsApp

    def _serve(self) -> None:
        claimed = self.headers.get("content-length", "0").strip() or "0"
        if not claimed.isdigit() or int(claimed) > MAX_BODY:
            status, body, kind, extra = 413, b'{"error": "a body over the limit"}', "application/json", {}
        else:
            body_in = self.rfile.read(int(claimed)) if self.command == "POST" else b""
            reply = self.app.dispatch(self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body_in)
            status, body, kind, extra = reply.status, reply.body, reply.type, reply.headers
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _serve

    def log_message(self, *_args: object) -> None:
        return None


def make_server(app: RequestsApp, port: int = 0) -> Server:
    """A server on 127.0.0.1 answering for ``app``."""
    return Server(("127.0.0.1", port), type("RequestsHandler", (Handler,), {"app": app}))
