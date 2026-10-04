"""Frozen daemon worker startup and native harness lifecycle with a fixture broker."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_packaging_conversations import call, running

pytestmark = pytest.mark.slow


def test_frozen_coding_worker_resumes_cancels_and_revokes_agent(tmp_path):
    binary = os.environ.get("ML_STACK_FROZEN_CODING_BINARY", "")
    if not binary:
        pytest.skip("set ML_STACK_FROZEN_CODING_BINARY to the fixture-broker standalone build")
    root = tmp_path / "daemon"
    root.mkdir()
    (root / "settings.json").write_text(json.dumps({"setup_done": True}))
    tools = tmp_path / "bin"
    tools.mkdir()
    codex = tools / "codex"
    source = Path(__file__).with_name("coding_kit.py").read_text().replace("from __future__ import annotations\n", "")
    codex.write_text(f"#!{sys.executable}\n" + source)
    codex.chmod(0o700)
    project = tmp_path / "project"
    project.mkdir()
    from ml_stack.workspace import Workspace, tokens

    workspace_root = tmp_path / "workspace"
    workspace = Workspace(workspace_root)
    tokens.store(workspace_root, tokens.OWNER_FILE, workspace.init("person"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                   "ML_STACK_HOME": str(tmp_path / "home"), "ML_STACK_WORKSPACE_HOME": str(workspace_root),
                   "PYTHONPATH": "", "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}
    blocked = subprocess.run([binary, "-m", "ml_stack.harnesshook", "pre", "--role", "read-only",
                              "--label", "fixture", "--root", str(project)],
                             input=json.dumps({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}),
                             capture_output=True, text=True, env=environment, timeout=10)
    assert blocked.returncode == 0, blocked.stderr
    assert json.loads(blocked.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
    with running(Path(binary), root, port, environment, tmp_path / "frozen.log"):
        made = call(port, "POST", "/ui/conversations", {"model": "fixture-model", "settings": {
            "mode": "coding", "project": str(project), "role": "read-only"}})
        cid = made["id"]
        for message in ("inspect project", "inspect sensors"):
            call(port, "POST", f"/ui/coding/{cid}/start", {"message": message})
            assert wait_state(port, cid, {"completed", "failed"})["state"] == "completed"
        saved = call(port, "GET", f"/ui/conversations/{cid}")
        assert len(saved["messages"]) == 4
        call(port, "POST", f"/ui/coding/{cid}/start", {"message": "wait until cancelled"})
        wait_state(port, cid, {"running"})
        call(port, "POST", f"/ui/coding/{cid}/cancel", {})
        assert wait_state(port, cid, {"cancelled", "failed"})["state"] == "cancelled"
        assert not workspace.registry.role_of(f"chat-{cid}")
        assert workspace.registry.role_of("person") == "human"


def wait_state(port, cid, states):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = call(port, "GET", f"/ui/coding/{cid}/status")
        if state["state"] in states:
            return state
        if state["state"] == "failed":
            pytest.fail(str(state))
        time.sleep(.02)
    pytest.fail("frozen coding worker did not reach the expected state")
