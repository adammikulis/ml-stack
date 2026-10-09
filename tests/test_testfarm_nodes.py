"""Tests on another device with two real node processes (TLS on loopback, no beacon, no multicast).

Device A asks, device B runs. B's node starts the real executor on this interpreter and checkout; the
shipped tree's `scripts/test` is a stub (tests/testfarm_tree.py), so everything from the CLI down to
the child process is the code that ships.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time

import pytest
import test_on
from testfarm_kit import BOARD, enable, go, gone, jobs_on, peer, until

from ml_stack import features, features_cli, node_launch
from ml_stack.fleet import shard_split, shard_tree
from ml_stack.testfarm import consent, devices
from ml_stack.testfarm.client import ShardError, Shards, choose

pytest_plugins = ("testfarm_kit",)  # the `pool` fixture: two paired node processes
pytestmark = pytest.mark.slow


def test_a_device_nobody_enabled_takes_nothing_and_is_listed_with_why(pool, capsys):
    a, _b, shards, tree = pool
    fp = peer(a)["fingerprint"]
    caps = shards.capability(fp)
    assert caps["accepts"] is False and "testfarm.consent" in caps["reason"]
    with pytest.raises(ShardError, match="off on this device"):
        shards.call(fp, "shard_put", id="ab" * 16, offset=0, data="00")
    assert go(fp, tree, ["tests/test_a.py"]) == test_on.NOT_RUN
    assert "does not take tests" in capsys.readouterr().err
    assert devices.command(["--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["caps"]["accepts"] is False and row["fingerprint"] == fp


def test_the_person_turns_a_device_on_names_whose_tests_it_takes_and_off_stops_the_next_upload(pool, monkeypatch, capsys):
    a, b, shards, _tree = pool
    fp = peer(a)["fingerprint"]
    monkeypatch.setenv("ML_STACK_HOME", str(b.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", b.token)
    assert consent.run(["on"]) == 1 and "experimental feature" in capsys.readouterr().out
    assert shards.capability(fp)["accepts"] is False, "the feature is off on B, so the switch stays off"
    features.switch("remote-tests", True)
    assert consent.run(["on"]) == 0 and "takes tests from nobody yet" in capsys.readouterr().out
    assert consent.run(["status", "--json"]) == 0 and json.loads(capsys.readouterr().out)["python"] == sys.executable
    monkeypatch.setenv("ML_STACK_HOME", str(a.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", a.token)
    asked = shards.capability(fp)
    assert asked["accepts"] is False and asked["allowed"] is False and "consent allow" in asked["reason"]
    monkeypatch.setenv("ML_STACK_HOME", str(b.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", b.token)
    named = peer(b)["name"]
    assert consent.run(["allow", named]) == 0 and "takes tests from 1 device" in capsys.readouterr().out
    monkeypatch.setenv("ML_STACK_HOME", str(a.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", a.token)
    assert shards.capability(fp)["accepts"] is True
    monkeypatch.setenv("ML_STACK_HOME", str(b.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", b.token)
    assert consent.run(["deny", named]) == 0 and "takes tests from nobody yet" in capsys.readouterr().out
    assert consent.run(["on", "--from", named]) == 0 and "takes tests from 1 device" in capsys.readouterr().out
    assert consent.run(["off"]) == 0
    monkeypatch.setenv("ML_STACK_HOME", str(a.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", a.token)
    assert shards.capability(fp)["accepts"] is False


def test_disabling_the_feature_switches_the_node_off_too(pool, monkeypatch, capsys):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    assert shards.capability(fp)["accepts"] is True
    monkeypatch.setenv("ML_STACK_HOME", str(b.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", b.token)
    features.switch("remote-tests", True)
    assert features_cli.main(["disable", "remote-tests"]) == 0
    assert features.enabled("remote-tests") is False and "test shards: off" in capsys.readouterr().out
    monkeypatch.setenv("ML_STACK_HOME", str(a.root))
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", a.token)
    assert shards.capability(fp)["accepts"] is False


def test_files_run_on_the_device_in_the_shape_of_a_local_run_and_the_pass_is_reused_once_per_content(pool, capsys):
    a, b, _shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    assert go(fp, tree, ["tests/test_a.py", "tests/test_b.py"]) == 0
    out = capsys.readouterr().out
    assert f"+ ran on {peer(a)['name']}: {sys.platform}, python 3.13" in out and "2 passed in" in out
    assert jobs_on(b) == 1
    assert go(fp, tree, ["tests/test_a.py", "tests/test_b.py"]) == 0
    assert "every test file passed there with this content (2 reused)" in capsys.readouterr().out
    assert jobs_on(b) == 1, "nothing was sent the second time"
    assert go(fp, tree, ["tests/test_a.py", "tests/test_b.py"], reuse=False) == 0
    assert jobs_on(b) == 2
    (tree / "tests" / "test_a.py").write_text("def test_one():\n    assert 1 == 1\n", encoding="utf-8")
    assert go(fp, tree, ["tests/test_a.py", "tests/test_b.py"]) == 0
    assert "reused: tests/test_b.py passed on" in capsys.readouterr().out and jobs_on(b) == 3, "only the changed file ran again"
    page = a.call("read", BOARD, a.token, kind="message", limit=200)["entries"]
    posted = [e["fields"]["body"] for e in page if e["fields"].get("body", "").startswith("test-result device=")]
    assert len(posted) >= 3 and all(f"fp={fp[:16]}" in line and f"platform={sys.platform}" in line for line in posted)


def test_a_failing_file_is_reported_with_its_message_and_is_not_kept_as_a_pass(pool, capsys):
    a, b, _shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    assert go(fp, tree, ["tests/test_a.py", "tests/test_fail.py"]) == 1
    assert "FAILED tests/test_fail.py::test_fail::test_one - boom: expected 1" in capsys.readouterr().out
    assert go(fp, tree, ["tests/test_a.py", "tests/test_fail.py"]) == 1
    out = capsys.readouterr().out
    assert "reused: tests/test_a.py passed on" in out and "reused: tests/test_fail.py" not in out


def test_the_gate_and_a_whole_tier_run_there_with_no_file_list(pool, capsys):
    a, b, _shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    assert go(fp, tree, [], tier="gate") == 0 and go(fp, tree, [], tier="full") == 0
    assert "2 passed" in capsys.readouterr().out


def test_a_tree_that_escapes_its_folder_is_refused_by_the_executor_on_the_device(pool, tmp_path):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    evil = tmp_path / "evil.tgz"
    with tarfile.open(evil, "w:gz") as tar:
        for name in ("tests/test_a.py", "../../escape.py"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"x\n\n"))
    data = evil.read_bytes()
    shard = "cd" * 16
    shards.call(fp, "shard_put", id=shard, offset=0, data=data.hex())
    shards.call(fp, "shard_start", id=shard, tree_sha256=hashlib.sha256(data).hexdigest(), size=len(data), tier="all", files=[], timeout_s=60)
    result = shards.wait(fp, shard, timeout_s=60, poll_s=0.2)
    assert result["exit"] == 70 and "path a shard does not carry" in result["output_tail"]
    assert not list(b.root.rglob("escape.py")) and not (b.root.parent / "escape.py").exists()


def test_cancel_ends_the_runner_and_the_child_it_started(pool):
    a, b, shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    shard = shards.send(fp, tree, "all", ["tests/test_hang.py"], 600)
    pids = tree.parent / "pids"
    until("the runner to write its pids", 60, pids.exists)
    until("both pids", 5, lambda: len(pids.read_text().split()) == 2)
    runner, child = (int(p) for p in pids.read_text().split())
    assert not gone(runner) and not gone(child)
    shards.cancel(fp, shard)
    with pytest.raises(ShardError, match="cancelled"):
        shards.wait(fp, shard, timeout_s=60, poll_s=0.2)
    until("the runner and its child to be gone", 30, lambda: gone(runner) and gone(child))
    assert shards.capability(fp)["free"] == 2


def test_at_most_two_run_at_once_and_a_third_is_told_to_wait(pool):
    a, b, shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    held = [shards.send(fp, tree, "all", ["tests/test_hang.py"], 600) for _ in range(2)]
    until("both to run", 30, lambda: shards.capability(fp)["free"] == 0)
    with pytest.raises(ShardError, match="2 shards are already running"):
        shards.send(fp, tree, "all", ["tests/test_a.py"], 60)
    for shard in held:
        shards.cancel(fp, shard)
    until("the slots to come back", 30, lambda: shards.capability(fp)["free"] == 2)


def test_a_requester_cannot_name_who_asks_and_only_the_node_ops_exist(pool):
    a, b, shards, _tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    with pytest.raises(ShardError, match="not an argument"):
        shards.call(fp, "shard_caps", by="claude-lead")
    with pytest.raises(ShardError, match="op is one of"):
        shards.call(fp, "members")
    with pytest.raises(ShardError, match="no other active device"):
        Shards(shards.session).call("f" * 64, "shard_caps")


def test_devices_are_chosen_by_name_or_fingerprint_and_a_shared_name_must_be_told_apart(pool):
    a, *_ = pool
    one = peer(a)
    assert choose(one["fingerprint"], [one]) == [one] and choose("all", [one]) == [one]
    with pytest.raises(ShardError, match="no active pool device"):
        choose("nobody", [one])
    twin = {**one, "fingerprint": "e" * 64}
    with pytest.raises(ShardError, match="two active pool device"):
        choose(one["name"], [one, twin])
    assert choose(twin["fingerprint"], [one, twin]) == [twin]


def test_a_pack_larger_than_a_request_may_carry_is_refused_before_anything_is_sent(pool, tmp_path, monkeypatch):
    a, b, shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    monkeypatch.setattr(shard_tree, "MOST_PACKED", 100)
    with pytest.raises(ShardError, match="at most"):
        shards.send(fp, tree, "all", ["tests/test_a.py"], 60)
    assert jobs_on(b) == 0


def test_a_device_that_cannot_be_reached_is_an_error_not_a_hang(pool, capsys):
    a, b, shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    node_launch.stop_node(b.state)
    began = time.monotonic()
    with pytest.raises(ShardError):
        shards.capability(fp)
    assert time.monotonic() - began < 30
    assert go(fp, tree, ["tests/test_a.py"]) == test_on.NOT_RUN
    assert "does not take tests: did not answer" in capsys.readouterr().err


def test_split_gives_this_machine_and_the_device_a_share_each_and_learns_durations(pool, monkeypatch, capsys):
    a, b, _shards, tree = pool
    enable(b)
    fp = peer(a)["fingerprint"]
    (tree / "tests" / "test_mac.py").write_text("import sys\nONLY = sys.platform == 'darwin'\n", encoding="utf-8")
    kept = []
    monkeypatch.setattr(test_on, "local", lambda root, tier, files: kept.append(files) or 0)
    history = test_on.history(tree)
    args = argparse.Namespace(tier="all", on=fp, split=True, base="main", timeout=300.0)
    assert test_on.main(args, ["tests/test_a.py", "tests/test_mac.py"], tree, lambda t, f: [], True) == 0
    assert kept == [["tests/test_mac.py"]], "a file that names darwin stays here"
    assert "1 passed" in capsys.readouterr().out and jobs_on(b) == 1
    assert "tests/test_a.py" in shard_split.load(history), "what the device measured is kept for the next split"
