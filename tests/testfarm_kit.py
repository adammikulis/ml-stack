"""Real node processes for the tests-on-another-device suites: paired devices on loopback, this process as one session.

The nodes listen on 127.0.0.1 only and run no beacon, so nothing uses multicast or asks the person for network
permission. ``pool`` is device A (the asker, this process's session) and B (the host), already paired; ``third``
makes a node C, alone in a pool of its own, for the tests that need a stranger.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import test_on
from node_kit import STRIPPED, node_binary  # noqa: F401  (a fixture the ones below use)
from testfarm_tree import make_tree

from ml_stack import features, node_binary as node_binary_module, node_launch, node_supervise
from ml_stack.board import session as board_session
from ml_stack.board.client import Client
from ml_stack.testfarm.client import Shards, pool_devices

BOARD = "demo"
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Device:
    root: Path
    state: Path
    client: Client
    token: str

    def call(self, method: str, board: str = "", token: str = "", **params):
        return self.client.call(method, board, token, **params)


def start(binary: Path, name: str) -> Device:
    """A node process on a short temporary root, listening on loopback, with one session registered on the board."""
    root = Path(tempfile.mkdtemp(prefix="ml", dir="/tmp"))
    state = root / "node"
    state.mkdir(mode=0o700)
    node_supervise.point(state, binary, node_binary_module.sha256(binary))
    node_launch.ensure_node(state, extra=["--listen", "127.0.0.1:0"])
    client = Client(state)
    got = client.call("register", BOARD, "", model="claude-sonnet-5-5", harness="claude-code", session=name)
    return Device(root, state, client, got["token"])


def skip_unless_possible() -> None:
    if sys.version_info[:2] != (3, 13) or os.name == "nt":
        pytest.skip("shards run on Python 3.13, and these node processes are POSIX")


def pair(a: Device, b: Device) -> None:
    """Enrol ``b`` in ``a``'s pool with a pairing code."""
    code = a.call("pair_accept", token=a.token)
    b.call("pair_start", token=b.token, host="127.0.0.1", port=code["port"], passphrase=code["code"])


@pytest.fixture
def pool(node_binary, monkeypatch, tmp_path):  # noqa: F811
    """Two paired devices; this process is device A's session. Returns (a, b, shards, tree root)."""
    skip_unless_possible()
    a, b = start(node_binary, "a"), start(node_binary, "b")
    pair(a, b)
    for name in STRIPPED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ML_STACK_HOME", str(a.root))
    monkeypatch.setenv("ML_STACK_BOARD", BOARD)
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", a.token)
    monkeypatch.setenv("DEV_TEST_REUSE_DIR", str(tmp_path / "reuse"))
    features.switch("remote-tests", True)  # this machine (device A's root) has the experimental feature on
    tree = make_tree(tmp_path / "tree", tmp_path / "pids")
    made = [a, b]
    try:
        yield a, b, Shards(board_session.Session(a.client, BOARD, a.token)), tree
    finally:
        for one in made:
            node_launch.stop_node(one.state)
            shutil.rmtree(one.root, ignore_errors=True)


@pytest.fixture
def third(node_binary):  # noqa: F811
    """A node C alone in a pool of its own, started on demand; its device and the way to stop it."""
    skip_unless_possible()
    c = start(node_binary, "c")
    try:
        yield c
    finally:
        node_launch.stop_node(c.state)
        shutil.rmtree(c.root, ignore_errors=True)


def enable(b: Device) -> None:
    """Turn shards on at ``b`` the way its person would: this interpreter, this checkout, taking tests from its one pool peer."""
    (asker,) = pool_devices(b.state)
    b.call("shard_consent", token=b.token, enabled=True, python=sys.executable, repo=str(ROOT), allow=[asker["fingerprint"]])


def peer(a: Device) -> dict:
    """The one other device in ``a``'s pool."""
    (found,) = pool_devices(a.state)
    return found


def go(b_fp: str, tree: Path, files: list[str], reuse: bool = True, tier: str = "all") -> int:
    """`scripts/test` with a tier and test files, run on one device by its fingerprint, in this process."""
    args = argparse.Namespace(tier=tier, on=b_fp, split=False, base="main", timeout=300.0)
    return test_on.main(args, files, tree, lambda t, f: [sys.executable, "-m", "pytest", "-n", "1", f], reuse)


def jobs_on(b: Device) -> int:
    """How many shard folders ``b``'s node holds."""
    return len(list((b.state / "shards").iterdir()))


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def until(what: str, seconds: float, check) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")
