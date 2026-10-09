"""Saved conversations through the standalone daemon."""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.slow
def test_frozen_daemon_keeps_graph_conversations_after_restart(tmp_path):
    binary = os.environ.get("POOLHOUSE_FROZEN_BINARY", "")
    if not binary:
        pytest.skip("set POOLHOUSE_FROZEN_BINARY to the built standalone daemon")
    root = tmp_path / "daemon"
    (root / "chats").mkdir(parents=True)
    legacy = {"id": "saved", "title": "Saved before packaging", "model": "saved-model", "created": 1,
              "messages": [{"role": "user", "content": "Keep my conversation", "at": 1}]}
    (root / "chats" / "saved.json").write_text(json.dumps(legacy))
    (root / "settings.json").write_text(json.dumps({"setup_done": True}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    from poolhouse.workspace import Workspace, tokens

    workspace_root = tmp_path / "workspace"
    workspace = Workspace(workspace_root)
    owner = workspace.init("person")
    tokens.store(workspace_root, tokens.OWNER_FILE, owner)
    environment = {**os.environ, "POOLHOUSE_HOME": str(tmp_path / "home"),
                   "POOLHOUSE_WORKSPACE_HOME": str(workspace_root), "PYTHONPATH": "",
                   "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}
    with running(Path(binary), root, port, environment, tmp_path / "first.log"):
        readiness = call(port, "GET", "/ui/chat")
        assert readiness["runtime_ready"], readiness.get("runtime_error")
        assert call(port, "GET", "/ui/board/agents") == {"owner_id": "person", "agents": []}
        assert call(port, "GET", "/ui/conversations/saved")["messages"] == [
            {**legacy["messages"][0], "reasoning": "", "status": "complete"}]
        made = call(port, "POST", "/ui/conversations", {"model": "chosen-model", "title": "Packaged chat",
                    "settings": {"project": "/tmp/project", "temperature": .4}})
        assert made["settings"]["temperature"] == .4
        call(port, "POST", f"/ui/conversations/{made['id']}", {"title": "Renamed packaged chat"})
        call(port, "DELETE", "/ui/conversations/saved")
    with running(Path(binary), root, port, environment, tmp_path / "second.log"):
        chats = call(port, "GET", "/ui/conversations")["conversations"]
        assert [chat["id"] for chat in chats] == [made["id"]]
        saved = call(port, "GET", f"/ui/conversations/{made['id']}")
        assert saved["title"] == "Renamed packaged chat"
        assert saved["model"] == "chosen-model"
        assert saved["settings"]["project"] == "/tmp/project"
        assert saved["settings"]["temperature"] == .4
    assert (root / "chats" / "conversations.db").exists()
    assert json.loads((root / "chats" / "saved.json").read_text()) == legacy


COOKIES: dict[int, str] = {}
"""The session cookie of each running daemon, opened the way the app's window opens one."""


def call(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, None if body is None else json.dumps(body),
                           {"X-Poolhouse-UI": "1", "Content-Type": "application/json",
                            "Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin",
                            **({"Cookie": COOKIES[port]} if port in COOKIES else {}), **(headers or {})})
        response = connection.getresponse()
        raw = response.read()
        payload = json.loads(raw)
        assert response.status < 400, payload
        if response.getheader("Set-Cookie"):
            COOKIES[port] = response.getheader("Set-Cookie").split(";", 1)[0]
        return payload
    finally:
        connection.close()


def sign_in(port, root):
    """Open a session with a launch ticket from the daemon's recorded secret."""
    record = json.loads((root / "launch" / "secret.json").read_text())
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("POST", "/ui/launch/ticket", b"{}",
                           {"X-Poolhouse-UI": "1", "X-Poolhouse-Launch": record["secret"],
                            "Content-Type": "application/json"})
        ticket = json.loads(connection.getresponse().read())["ticket"]
    finally:
        connection.close()
    COOKIES.pop(port, None)
    call(port, "POST", "/ui/session", {"ticket": ticket})


@contextlib.contextmanager
def running(binary, root, port, environment, log_path):
    with log_path.open("w") as log:
        process = subprocess.Popen([str(binary), "--root", str(root), "--port", str(port), "--no-browser",
                                    "--no-announce", "--cluster-key", str(root / "cluster.key")],
                                   stdout=log, stderr=log, env=environment)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                try:
                    call(port, "GET", "/health")
                    sign_in(port, root)
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        pytest.fail(log_path.read_text())
                    time.sleep(.05)
            yield
        finally:
            COOKIES.pop(port, None)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
