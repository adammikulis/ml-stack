"""Test shards between two pool daemons on loopback: the sender packs a tree, the stand-in device runs it.

Both ends are real: pinned TLS, signed and sealed requests, the stand-in's `JobRunner` and `ShardHost`,
a real scratch checkout and a real pytest in it. The tree is a small repository this test builds, with a
`scripts/test` that records the argv it was given and runs pytest on the named files.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from shard_support import NAME, Standin, start

from ml_stack.fleet import shard_client, shard_split
from ml_stack.fleet.remote import PeerError
from ml_stack.fleet.shard_consent import run as consent_run
from ml_stack.fleet.shard_host import ShardHost
from ml_stack.fleet.shard_run import shard_command
from ml_stack.fleet.shard_spec import frame

STUB = '''\
import subprocess, sys, time
args = sys.argv[1:]
print("ARGV", args, flush=True)
junit = next(a for a in args if a.startswith("--junitxml="))
files = [a for a in args if a.endswith(".py")]
if "tests/test_sleep.py" in files:
    time.sleep(60)
sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", junit, *files]))
'''
FILES = {
    "tests/test_ok.py": "def test_a():\n    assert 1\n\n\ndef test_b():\n    assert True\n",
    "tests/test_bad.py": "def test_c():\n    assert 1 == 2, 'one is not two'\n\n\ndef test_d():\n    assert 1\n",
    "tests/test_sleep.py": "def test_s():\n    assert 1\n",
    "scripts/test": STUB,
    "pyproject.toml": "[project]\nname = 'tiny'\n",
}


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "tree"
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


@pytest.fixture
def sender(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "sender"))
    return tmp_path


@pytest.fixture
def standin(sender):
    made: list[Standin] = []

    def make(**kwargs) -> Standin:
        one = start(sender / "standin", **kwargs)
        made.append(one)
        return one

    yield make
    for one in made:
        one.stop()


def run(peer, tree, files, timeout=120):
    return shard_client.wait(peer, shard_client.send(peer, tree, files, 60), timeout_s=timeout, poll_s=0.2)


def test_capability_shows_consent_and_platform(standin):
    one = standin(enabled=False)
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    got = shard_client.capability(peer)
    assert got["accepts"] is False and got["platform"]["system"] == sys.platform
    one.flag["on"] = True
    assert shard_client.capability(peer)["accepts"] is True


def test_shard_runs_and_reports_per_file_failures_and_platform(standin, tree):
    one = standin()
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    result = run(peer, tree, ["tests/test_bad.py", "tests/test_ok.py"])
    assert result["exit"] == 1 and result["state"] == "failed"
    assert result["files"]["tests/test_ok.py"]["passed"] == 2
    assert result["files"]["tests/test_bad.py"] | {"wall_s": 0} == {
        "passed": 1, "failed": 1, "error": 0, "skipped": 0, "wall_s": 0}
    [failure] = result["failures"]
    assert failure["nodeid"].startswith("tests/test_bad.py::") and "one is not two" in failure["message"]
    assert result["platform"]["system"] == sys.platform and result["platform"]["python"].count(".") == 2
    assert {row[0].split("::")[-1] for row in result["tests"]} == {"test_a", "test_b", "test_c", "test_d"}
    argv = json.loads(result["output_tail"].splitlines()[0].removeprefix("ARGV ").replace("'", '"'))
    assert argv[:2] == ["all", "--no-reuse"] and argv[3:] == ["tests/test_bad.py", "tests/test_ok.py"]
    assert argv[2].startswith("--junitxml=")


def test_passing_shard_exits_zero(standin, tree):
    one = standin()
    result = run(shard_client.device_peer(NAME, host="127.0.0.1", port=one.port), tree, ["tests/test_ok.py"])
    assert result["exit"] == 0 and result["state"] == "done" and result["failures"] == []
    assert not list((one.host.folder).glob("shard-*"))


def test_time_limit_cancels_the_run(standin, tree):
    one = standin()
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    shard_id = shard_client.send(peer, tree, ["tests/test_sleep.py"], 1)
    began = time.monotonic()
    result = shard_client.wait(peer, shard_id, timeout_s=120, poll_s=0.2)
    assert result["exit"] != 0 and time.monotonic() - began < 60


def test_the_host_runs_only_the_test_runner_on_the_named_files(tmp_path):
    argv = shard_command(tmp_path / "tree", ["tests/test_a.py"], tmp_path / "j.xml")
    assert argv == [sys.executable, str(tmp_path / "tree" / "scripts" / "test"), "all", "--no-reuse",
                    f"--junitxml={tmp_path / 'j.xml'}", "tests/test_a.py"]


def test_consent_command_flips_the_setting_the_host_reads(tmp_path):
    from ml_stack.fleet.shard_routes import consent

    path = tmp_path / "settings.json"
    assert consent(None, path) is False
    assert consent_run(["on", str(tmp_path)]) == 0 and consent(None, path) is True
    assert consent_run(["off", str(tmp_path)]) == 0 and consent(None, path) is False
    assert isinstance(ShardHost(tmp_path / "s", lambda: consent(None, path)).capability()["accepts"], bool)


def test_split_balances_by_duration_and_keeps_mac_files_here(tmp_path):
    history = {"tests/a.py": 100.0, "tests/b.py": 50.0, "tests/c.py": 50.0, "tests/d.py": 1.0}
    targets = [shard_split.Target("local", 4), shard_split.Target("far", 4, 3.0)]
    plan = shard_split.split(list(history), targets, history, stay={"tests/d.py"})
    assert "tests/d.py" in plan["local"] and plan["local"] != list(history)
    assert sorted(name for part in plan.values() for name in part) == sorted(history)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "m.py").write_text("if sys.platform != 'darwin':\n    pass\n")
    (tmp_path / "tests" / "p.py").write_text("def test_x():\n    pass\n")
    assert shard_split.local_only(tmp_path, "tests/m.py") and not shard_split.local_only(tmp_path, "tests/p.py")


def test_split_weights_by_workers():
    history = {f"tests/{n}.py": 10.0 for n in "abcdefgh"}
    plan = shard_split.split(list(history), [shard_split.Target("local", 1), shard_split.Target("far", 3)], history)
    assert len(plan["far"]) > len(plan["local"])


def test_a_peer_that_cannot_be_reached_is_an_error_not_a_hang(sender, standin, tree):
    one = standin()
    peer = shard_client.device_peer(NAME, host="127.0.0.1", port=one.port)
    one.stop()
    with pytest.raises(PeerError):
        shard_client.capability(peer)


def test_frame_round_trips_a_header_and_tree():
    from ml_stack.fleet.shard_spec import split

    header, tree = split(frame({"id": "a" * 32}, b"tree"))
    assert header == {"id": "a" * 32} and tree == b"tree"


def test_unpaired_name_is_not_found(sender):
    with pytest.raises(shard_client.NoDevice):
        shard_client.device_peer("nobody")


@pytest.fixture
def farm(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_TEST_REUSE_DIR", str(tmp_path / "reuse"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import test_farm

    return test_farm


def farm_args(one, *rest):
    return ["--to", NAME, "--host", "127.0.0.1", "--port", str(one.port), *rest]


def test_farm_prints_the_remote_run_like_a_local_one_and_exits_with_its_status(standin, tree, farm, capsys):
    one = standin()
    status = farm.main(farm_args(one, "tests/test_bad.py", "tests/test_ok.py"), tree)
    out = capsys.readouterr().out
    assert status == 1
    assert f"ran on {NAME}: {sys.platform}" in out and "FAILED tests/test_bad.py::" in out
    assert "one is not two" in out and "3 passed, 1 failed" in out


def test_farm_exits_zero_when_the_shard_passes(standin, tree, farm):
    assert farm.main(farm_args(standin(), "tests/test_ok.py"), tree) == 0


def test_farm_refuses_a_device_with_shards_off(standin, tree, farm, capsys):
    status = farm.main(farm_args(standin(enabled=False), "tests/test_ok.py"), tree)
    assert status == farm.NOT_RUN and "does not take test shards" in capsys.readouterr().err


def test_farm_check_reports_the_capability(standin, tree, farm, capsys):
    one = standin()
    assert farm.main(farm_args(one, "--check"), tree) == 0
    assert "takes test shards: True" in capsys.readouterr().out
    one.flag["on"] = False
    assert farm.main(farm_args(one, "--check"), tree) == farm.NOT_RUN


def test_farm_split_runs_part_here_and_part_there_and_learns_durations(standin, tree, farm, capsys):
    one = standin()
    shard_split.record(farm.history_path(tree), {"files": {"tests/test_bad.py": {"wall_s": 90.0},
                                                         "tests/test_ok.py": {"wall_s": 90.0}}})
    status = farm.main(farm_args(one, "--split", "tests/test_bad.py", "tests/test_ok.py"), tree)
    out = capsys.readouterr().out
    assert status == 1 and out.count("files on") == 2
    assert set(shard_split.load(farm.history_path(tree))) == {"tests/test_bad.py", "tests/test_ok.py"}
