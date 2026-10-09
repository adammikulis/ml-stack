"""A real node process on a short temporary root, and the sessions and commands that talk to it.

Import `node_binary` and `workspace_node` into a test module to use them. The state directory is
under /tmp with a short name because a Unix socket path is limited to about 100 characters, which
a pytest `tmp_path` can exceed. The node binary is built once per session with
``cargo build -p poolside-node``.
"""

from __future__ import annotations

import fcntl
import os
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ml_stack.board import client as board_client, credentials, node_start, session as board_session

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
STRIPPED = ("CODEX_THREAD_ID", "CODEX_SESSION_ID", "CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE",
            "ML_STACK_WORKSPACE_TOKEN", "ML_STACK_WORKSPACE_DENYLIST", "ML_STACK_WORKSPACE_AGENT", "ML_STACK_BOARD")
BOARD = "demo"


def _cargo_env() -> tuple[str, dict[str, str]]:
    """The cargo binary and an environment where it finds the account's toolchain (HOME is moved in tests)."""
    account = Path(pwd.getpwuid(os.getuid()).pw_dir)
    found = shutil.which("cargo") or str(account / ".cargo" / "bin" / "cargo")
    return found, {**os.environ, "HOME": str(account)}


@pytest.fixture(scope="session")
def node_binary() -> Path:
    """The node binary, built once for the session (and once across parallel workers)."""
    named = os.environ.get(node_start.BIN_ENV)
    if named:
        return Path(named)
    cargo, env = _cargo_env()
    target = ROOT / "app" / "target"
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".test-build.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        done = subprocess.run([cargo, "build", "-p", "poolside-node"], cwd=ROOT / "app", env=env, capture_output=True,
                              text=True, check=False)
    if done.returncode:
        pytest.fail(f"cargo build -p poolside-node failed:\n{done.stderr[-2000:]}", pytrace=False)
    return target / "debug" / "poolside-node"


@dataclass
class Member:
    """A session registered on the node's board."""

    name: str
    token: str
    parent: str = ""
    session: str = ""

    def env(self) -> dict[str, str]:
        return {"ML_STACK_WORKSPACE_AGENT": self.name}


@dataclass
class WorkspaceNode:
    """A node running on ``state``, a client of it, and the board the tests use."""

    state: Path
    client: board_client.Client
    binary: Path
    board: str = BOARD
    members: dict[str, Member] = field(default_factory=dict)

    def member(self, native_id: str = "main", *, model: str = "claude-sonnet-5-5", harness: str = "claude-code",
               parent: Member | None = None) -> Member:
        """Register a native session (as a subagent of ``parent``) and keep its token where the CLI finds it."""
        made = board_session.register(self.client, self.board, board_session.Native(model, harness, native_id),
                                      parent.token if parent else "")
        found = Member(made.name, made.token, made.parent, native_id)
        self.members[native_id] = found
        return found

    def adopt(self, name: str, parent: str = "") -> Member:
        """The session ``name`` a hook or command registered, with the token it kept in its private file."""
        return Member(name, credentials.load(self.state, self.board, name), parent)

    def session(self, who: Member) -> board_session.Session:
        """The typed session of ``who`` on this node."""
        return board_session.Session(self.client, self.board, who.token, who.name)

    def env(self, who: Member | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
        """The environment of a command run for ``who``: this node, this board, no marker of another agent."""
        env = {k: v for k, v in os.environ.items() if k not in STRIPPED}
        env.update({"PYTHONPATH": SRC, "ML_STACK_NODE_DIR": str(self.state), "ML_STACK_NODE_BIN": str(self.binary),
                    "ML_STACK_BOARD": self.board, **(who.env() if who else {}), **(extra or {})})
        return env

    def cli(self, *argv: str, who: Member | None = None, env: dict[str, str] | None = None, stdin: str | None = None,
            timeout: float = 90, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        """Run ``ml-stack-workspace`` as a subprocess, as ``who``."""
        return subprocess.run([sys.executable, "-m", "ml_stack.workspace.cli", *argv], env=self.env(who, env),
                              capture_output=True, text=True, timeout=timeout, check=False, input=stdin, cwd=cwd)

    def stop(self) -> None:
        """Stop the node and wait for it to go."""
        try:
            pid = int(self.client.call("status")["pid"])
        except (OSError, board_client.NodeError):
            return
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        os.kill(pid, signal.SIGKILL)


@pytest.fixture
def workspace_node(node_binary, monkeypatch):
    """A running node on a short temporary root; in-process clients and child commands find it by the environment."""
    root = Path(tempfile.mkdtemp(prefix="ml", dir="/tmp"))
    state = root / "n"
    monkeypatch.setenv("ML_STACK_NODE_DIR", str(state))
    monkeypatch.setenv(node_start.BIN_ENV, str(node_binary))
    monkeypatch.setenv("ML_STACK_BOARD", BOARD)
    for name in STRIPPED:
        if name != "ML_STACK_BOARD":
            monkeypatch.delenv(name, raising=False)
    node = WorkspaceNode(state, board_client.Client(state), node_binary)
    node_start.ensure(state)
    try:
        yield node
    finally:
        node.stop()
        shutil.rmtree(root, ignore_errors=True)
