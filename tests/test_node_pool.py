"""Joining a pool by the beacon with real node processes: two devices (two state directories) enrol each other under
`open`, share a board, and a revoked one stays out; the CLI turns the network on; the check names the step that fails."""

from __future__ import annotations

import ast
import os
import socket
import time
from pathlib import Path

import pytest

from ml_stack import node_launch, node_pool, node_pool_check
from ml_stack.node_health import call
from tests import test_node_launch as launch

pytestmark = pytest.mark.slow
built, home = launch.built, launch.home  # the launcher's fixtures: the release node binary, and a short state root
install_runtime, waited = launch.install_runtime, launch.waited
LAN = bool(os.environ.get("ML_STACK_LAN_TESTS"))  # tests that listen on every interface exist only when asked for, at the machine
NETWORK_CALLS = ("node_pool.join(", "step_node(", "--network", "node_pool.start(")


def free_port(kind: int = socket.SOCK_DGRAM) -> int:
    with socket.socket(socket.AF_INET, kind) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def loopback_flags(bind: int, send: int) -> list[str]:
    return ["--listen", "127.0.0.1:0", "--beacon-bind", f"127.0.0.1:{bind}", "--beacon-send", f"127.0.0.1:{send}",
            "--advertise", "127.0.0.1", "--allow-loopback", "--sync-ms", "300"]


def two_devices(home: Path, built: Path) -> tuple[Path, Path]:
    install_runtime(built, "a")
    pa, pb = free_port(), free_port()
    a, b = home / "a", home / "b"
    node_launch.ensure_node(a, extra=loopback_flags(pa, pb))
    node_launch.ensure_node(b, extra=loopback_flags(pb, pa))
    return a, b


def say(state: Path, text: str) -> str:
    token = call(state, "register", {"model": "claude-sonnet-5-5", "harness": "claude-code", "session": state.name}, board="demo")["token"]
    call(state, "post", {"kind": "message", "fields": {"type": "status", "body": text}}, board="demo", token=token)
    return token


def bodies(state: Path, token: str) -> list[str]:
    rows = call(state, "read", {"kind": "message", "limit": 100}, board="demo", token=token)["entries"]
    return sorted(r["fields"]["body"] for r in rows)


def test_two_nodes_enrol_each_other_under_open_share_a_board_and_a_revoked_one_stays_out(home, built):
    a, b = two_devices(home, built)
    try:
        ta, tb = say(a, "from a"), say(b, "from b")
        node_pool.set_policy(a, "open")
        node_pool.set_policy(b, "open")
        waited("both to list each other", lambda: node_pool.others(node_pool.status(a)) and node_pool.others(node_pool.status(b)), 30)
        member = node_pool.others(node_pool.status(a))[0]
        assert member["by"] == "open" and node_pool.status(a)["pool"] == node_pool.status(b)["pool"]
        waited("the board to converge", lambda: bodies(a, ta) == bodies(b, tb) == ["from a", "from b"], 30)
        call(a, "member_revoke", {"fingerprint": member["fingerprint"]}, token=node_pool._token(a))
        time.sleep(2)  # several beacons and syncs: a revoked device is not let back in
        assert not node_pool.others(node_pool.status(a))
        assert any(m["status"] == "revoked" for m in node_pool.status(a)["members"])
    finally:
        node_launch.stop_node(a)
        node_launch.stop_node(b)


def test_the_check_stops_at_the_first_failing_step_and_says_what_to_change(home, tmp_path, monkeypatch):
    monkeypatch.setattr(node_pool_check, "route_address", lambda: "192.0.2.5")
    monkeypatch.setattr(node_pool_check, "is_wsl", lambda: False)
    lines: list[str] = []
    check = node_pool_check.Check(state=home / "c", port=1, beacon_port=2, wait_s=1, policy="open", binary=tmp_path / "missing" / "poolside-node")
    assert node_pool_check.walk(check, lines.append) is False
    joined = "\n".join(lines)
    assert lines[0].startswith("PASS  platform") and lines[1].startswith("FAIL  binary")
    assert "fix: " in lines[1] and "cargo build --release -p poolside-node" in lines[1]
    assert [line.split()[0] for line in lines] == ["PASS", "FAIL", *["SKIP"] * (len(node_pool_check.STEPS) - 2)], joined
    assert node_pool_check.summary(check).startswith("NOT READY")


def test_wsl_in_nat_mode_is_named_as_the_problem_with_the_exact_fix(monkeypatch, tmp_path):
    monkeypatch.setattr(node_pool_check, "is_wsl", lambda: True)
    monkeypatch.setattr(node_pool_check, "route_address", lambda: "172.20.1.2")
    monkeypatch.setattr(node_pool_check, "wsl_mode", lambda address: "nat")
    check = node_pool_check.Check(state=tmp_path, port=1, beacon_port=2, wait_s=0, policy="open", binary=None)
    step = node_pool_check.step_platform(check)
    assert not step.ok and "networkingMode=mirrored" in step.fix and "wsl --shutdown" in step.fix and ".wslconfig" in step.fix
    monkeypatch.setattr(node_pool_check, "wsl_mode", lambda address: "mirrored")
    assert node_pool_check.step_platform(check).ok


if LAN:
    def test_join_turns_the_network_on_for_a_node_that_ran_without_it_and_sets_the_policy(home, built):
        install_runtime(built, "a")
        state = home / "j"
        node_launch.ensure_node(state)
        assert node_pool.status(state)["listen"] is None, "a node started the old way has no network"
        port, beacon = free_port(socket.SOCK_STREAM), free_port()
        try:
            shown = node_pool.join(state, policy="open", port=port, beacon_port=beacon)
            assert shown["policy"] == "open" and shown["listen"].endswith(f":{port}") and shown["beacon"] is True
            assert shown["members"][0]["self"] is True and shown["pool"], "the first run made a pool with this device in it"
            shown = node_pool.join(state, policy="secure", port=port, beacon_port=beacon)
            assert shown["policy"] == "secure"
        finally:
            node_launch.stop_node(state)

    def test_a_node_that_hears_nothing_is_told_what_to_allow(home, built):
        install_runtime(built, "a")
        port, beacon = free_port(socket.SOCK_STREAM), free_port()
        check = node_pool_check.Check(state=home / "d", port=port, beacon_port=beacon, wait_s=1, policy="open", binary=None)
        try:
            assert node_pool_check.step_node(check).ok
            assert node_pool_check.step_listening(check).ok
            assert node_pool_check.step_beacon_send(check).ok
            assert node_pool_check.step_policy(check).ok
            step = node_pool_check.step_peer_beacon(check)
            assert not step.ok and "other device" in step.fix, "alone on a port nobody else uses, nothing is heard"
        finally:
            node_launch.stop_node(home / "d")


def test_only_the_opt_in_tests_here_turn_the_machines_network_on():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and node.name != "test_only_the_opt_in_tests_here_turn_the_machines_network_on":
            source = ast.get_source_segment(Path(__file__).read_text(encoding="utf-8"), node) or ""
            assert not any(call in source for call in NETWORK_CALLS), f"{node.name} listens beyond loopback outside `if LAN:`"


def test_turning_the_network_on_needs_a_person_to_agree_and_starts_nothing_without_one(home):
    with pytest.raises(node_pool.PoolError, match="needs a person"):
        node_pool.consent(1, 2, yes=False, interactive=False)
    with pytest.raises(node_pool.PoolError, match="stays off"):
        node_pool.consent(1, 2, yes=False, ask=lambda _: "n", interactive=True)
    node_pool.consent(1, 2, yes=False, ask=lambda _: "y", interactive=True)
    assert node_pool.main(["join", "--state", str(home / "never"), "--port", "1"]) == 1, "no terminal, no --yes: refused before any node starts"
    assert not (home / "never").exists()
