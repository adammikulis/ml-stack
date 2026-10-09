"""Tests on other devices, below the node: the executor a node starts, the per-device record and the report.

The executor runs as the node runs it (a child process given one folder) over a real git tree whose
`scripts/test` is a stub, so unpack, the scratch checkout, the runner process and the junit are real.
Hostile folders are the ones a sender could cause: the node checks the digest, the executor checks the rest.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
import test_on
from testfarm_tree import MARKERS, make_tree

from ml_stack import features
from ml_stack.fleet import shard_exec, shard_spec, shard_split, shard_tree
from ml_stack.testfarm import consent, devices, ledger, report

ROOT = Path(__file__).resolve().parents[1]
ID = "ab" * 16


def run_executor(folder: Path, env: dict | None = None) -> tuple[int, dict]:
    """Start the executor on ``folder`` the way the node does; its exit status and ``result.json``."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); from ml_stack.fleet.shard_exec import run; raise SystemExit(run(sys.argv[2]))"
    base = {k: v for k, v in os.environ.items() if k not in MARKERS and k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, "-E", "-c", code, str(ROOT / "src"), str(folder)], env={**base, **(env or {})},
                          capture_output=True, text=True, timeout=120, check=False)
    result = folder / "result.json"
    return done.returncode, json.loads(result.read_text()) if result.is_file() else {}


def job(tmp_path: Path, tier: str = "all", files: list[str] | None = None, tree: bytes | None = None, **header) -> Path:
    """A job folder the way the node leaves it: ``spec.json`` and ``tree.tgz``."""
    folder = tmp_path / "job"
    folder.mkdir()
    if tree is None:
        tree, _ = shard_tree.pack(make_tree(tmp_path / "src", tmp_path / "pids"))
    spec = {"id": ID, "tree_sha256": hashlib.sha256(tree).hexdigest(), "tier": tier, "files": files or [], "timeout_s": 120, **header}
    (folder / "spec.json").write_text(json.dumps(spec))
    (folder / "tree.tgz").write_bytes(tree)
    return folder


def archive(members: list[tuple[str, bytes, bytes]]) -> bytes:
    """A gzip tar of ``(name, type, content)`` members; type is ``0`` file, ``2`` symlink, ``5`` directory."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, kind, content in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(content) if kind == tarfile.REGTYPE else 0
            info.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
            tar.addfile(info, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    return out.getvalue()


# -- the executor -------------------------------------------------------------

def test_a_job_runs_the_named_files_and_the_result_says_where_and_how_it_went(tmp_path):
    code, result = run_executor(job(tmp_path, files=["tests/test_a.py", "tests/test_b.py"]))
    assert code == 0 and result["state"] == "done" and result["exit"] == 0
    assert result["files"]["tests/test_a.py"]["passed"] == 1 and result["files"]["tests/test_b.py"]["passed"] == 1
    assert result["platform"]["system"] == sys.platform and result["platform"]["python"].startswith("3.")
    assert "stub ran all ['tests/test_a.py', 'tests/test_b.py']" in result["output_tail"]
    assert report.summary(result).startswith("2 passed")


def test_a_failing_file_is_named_with_its_message_and_the_exit_is_the_runners(tmp_path):
    code, result = run_executor(job(tmp_path, files=["tests/test_a.py", "tests/test_fail.py"]))
    assert code == 1 and result["exit"] == 1 and result["state"] == "failed"
    assert [f["nodeid"] for f in result["failures"]] == ["tests/test_fail.py::test_fail::test_one"]
    assert "boom: expected 1" in result["failures"][0]["message"]
    assert not ledger.clean(result["files"]["tests/test_fail.py"]) and ledger.clean(result["files"]["tests/test_a.py"])


def test_a_whole_tier_with_no_file_list_counts_by_the_files_the_junit_names(tmp_path):
    code, result = run_executor(job(tmp_path, tier="full"))
    assert code == 0 and sorted(result["files"]) == ["tests/test_a.py", "tests/test_b.py"]


def test_the_gate_runs_with_no_files_and_reports_its_own_words(tmp_path):
    code, result = run_executor(job(tmp_path, tier="gate"))
    assert code == 0 and "gate: every structural check passed" in result["output_tail"]


def test_no_agent_marker_and_no_credential_reaches_the_runner(tmp_path):
    leaked = dict.fromkeys(MARKERS, "1") | {"ML_STACK_SHARD_TOKEN": "secret", "OPENAI_API_KEY": "sk-x"}
    code, result = run_executor(job(tmp_path, files=["tests/test_a.py"]), env=leaked)
    assert code == 0 and result["failures"] == [], result["failures"]


@pytest.mark.parametrize(("what", "make"), [
    ("an extra field", lambda t: {"argv": ["rm", "-rf", "/"]}),
    ("a tier that is an option", lambda t: {"tier": "--help"}),
    ("a file outside tests", lambda t: {"files": ["scripts/test"]}),
    ("a parent path", lambda t: {"files": ["tests/../scripts/test.py"]}),
    ("a node id", lambda t: {"files": ["tests/test_a.py::test_one"]}),
    ("an option as a file", lambda t: {"files": ["--rootdir=/"]}),
    ("a file the tree lacks", lambda t: {"files": ["tests/test_missing.py"]}),
    ("the gate with files", lambda t: {"tier": "gate", "files": ["tests/test_a.py"]}),
    ("a long time limit", lambda t: {"timeout_s": 99999}),
    ("a digest that does not match", lambda t: {"tree_sha256": "0" * 64}),
])
def test_a_job_that_breaks_a_rule_is_refused_and_nothing_runs(tmp_path, what, make):
    folder = job(tmp_path)
    spec = json.loads((folder / "spec.json").read_text())
    (folder / "spec.json").write_text(json.dumps({**spec, **make(tmp_path)}))
    code, result = run_executor(folder)
    assert code == shard_exec.REFUSED and result["state"] == "failed", what
    assert "stub ran" not in result["output_tail"] and result["output_tail"], what


@pytest.mark.parametrize(("what", "members"), [
    ("a parent path", [("tests/../../escape.py", tarfile.REGTYPE, b"x = 1\n")]),
    ("an absolute path", [("/tmp/escape.py", tarfile.REGTYPE, b"x = 1\n")]),
    ("a symlink", [("tests/link.py", tarfile.SYMTYPE, b"")]),
    ("a directory entry", [("tests/", tarfile.DIRTYPE, b"")]),
    ("a hidden path", [(".git/config", tarfile.REGTYPE, b"[core]\n")]),
    ("a backslash path", [("tests\\evil.py", tarfile.REGTYPE, b"x = 1\n")]),
    ("a drive path", [("C:/evil.py", tarfile.REGTYPE, b"x = 1\n")]),
])
def test_a_tree_with_anything_but_plain_shipped_files_is_refused(tmp_path, what, members):
    sentinel = tmp_path.parent / "escape.py"
    folder = job(tmp_path, files=[], tree=archive([("tests/test_a.py", tarfile.REGTYPE, b"x\n"), *members]))
    code, result = run_executor(folder)
    assert code == shard_exec.REFUSED and "stub ran" not in result["output_tail"], what
    assert not sentinel.exists() and not Path("/tmp/escape.py").exists(), what


def test_an_archive_that_inflates_past_the_limit_is_refused_before_it_is_read(tmp_path):
    bomb = archive([("tests/test_a.py", tarfile.REGTYPE, b"\0" * (300 << 20))])
    assert len(bomb) < 1 << 20
    code, result = run_executor(job(tmp_path, tree=bomb))
    assert code == shard_exec.REFUSED and "inflates" in result["output_tail"]


def test_the_time_limit_cancels_the_run(tmp_path):
    began = time.monotonic()
    code, result = run_executor(job(tmp_path, files=["tests/test_hang.py"], timeout_s=1))
    assert code == 124 and result["exit"] == 124 and time.monotonic() - began < 60
    for pid in (tmp_path / "pids").read_text().split():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pid), signal.SIGKILL)  # the node ends the group in real use; here nothing does


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


# -- the record and the report ------------------------------------------------

DEVICE = {"name": "win-box", "fingerprint": "a" * 64}
PLATFORM = {"system": "win32", "python": "3.13.5", "cpus": 8, "wsl": False}


def test_a_pass_belongs_to_one_device_one_platform_and_one_tier():
    base = ledger.key(DEVICE, PLATFORM, "all", "content")
    assert base == ledger.key(DEVICE, PLATFORM, "all", "content")
    other = {"name": "mac", "fingerprint": "b" * 64}
    assert len({base, ledger.key(other, PLATFORM, "all", "content"), ledger.key(DEVICE, {**PLATFORM, "system": "darwin"}, "all", "content"),
                ledger.key(DEVICE, {**PLATFORM, "wsl": True}, "all", "content"), ledger.key(DEVICE, {**PLATFORM, "python": "3.12.1"}, "all", "content"),
                ledger.key(DEVICE, PLATFORM, "full", "content"), ledger.key(DEVICE, PLATFORM, "all", "changed")}) == 7


def test_a_recorded_pass_is_found_for_that_device_only_and_a_timed_out_run_records_none(tmp_path):
    book = ledger.Ledger(tmp_path)
    good = {"exit": 0, "files": {"tests/test_a.py": {"passed": 2, "failed": 0, "error": 0}, "tests/test_b.py": {"passed": 0, "failed": 1, "error": 0}}}
    passes = book.passes_of(DEVICE, PLATFORM, "all", good, lambda f: f"content-of-{f}")
    assert list(passes.values()) == ["tests/test_a.py"], "only the file that ran clean"
    book.record(DEVICE, {"tier": "all", "exit": 0, "passed": 2, "failed": 0}, passes)
    (key,) = passes
    assert book.passed(DEVICE["fingerprint"], key)["file"] == "tests/test_a.py"
    assert book.passed("b" * 64, key) == {}, "another device does not have it"
    assert book.last(DEVICE["fingerprint"])["tier"] == "all"
    assert book.passes_of(DEVICE, PLATFORM, "all", {**good, "exit": 124}, lambda f: "x") == {}
    assert book.passes_of(DEVICE, PLATFORM, "all", good, lambda f: "") == {}, "a file that must always run is never kept"


def test_the_report_names_wsl_apart_from_windows_and_the_board_line_is_keyed_by_device():
    wsl = {**PLATFORM, "system": "linux", "wsl": True}
    assert report.where(wsl) == "linux (WSL), python 3.13.5, 8 cpus" and report.where(PLATFORM).startswith("win32,")
    result = {"exit": 1, "platform": wsl, "wall_s": 12.4, "cpu_s": 3.0, "output_tail": "", "files": {"tests/a.py": {"passed": 3, "failed": 1, "error": 0, "skipped": 2}},
              "failures": [{"nodeid": "tests/a.py::T::test_x", "state": "failed", "message": "assert 1 == 2\nmore"}]}
    lines = report.show(result, "ubuntu", ["tests/b.py"])
    assert lines[0] == "+ ran on ubuntu: linux (WSL), python 3.13.5, 8 cpus" and "reused: tests/b.py passed on ubuntu with this content" in lines
    assert "FAILED tests/a.py::T::test_x - assert 1 == 2" in lines and lines[-1].startswith("3 passed, 1 failed, 2 skipped in 12.4s (exit 1")
    line = report.board_line({"name": "ubuntu", "fingerprint": "c" * 64}, "all", 1, result, "d" * 40)
    assert "device=ubuntu" in line and f"fp={'c' * 16}" in line and "platform=linux-wsl" in line and "failed=1" in line


def test_the_device_table_says_why_a_device_takes_nothing():
    rows = [{"name": "mac", "caps": {"accepts": True, "platform": {**PLATFORM, "system": "darwin"}, "free": 1, "most_active": 2}, "last": {"exit": 0, "tier": "all", "passed": 4, "failed": 0, "at": 0}},
            {"name": "off-box", "caps": {"accepts": False, "reason": "test shards are off on this device"}, "last": {}}]
    text = "\n".join(devices.table(rows))
    assert "darwin, python 3.13.5" in text and "1/2" in text and "PASS all 4 passed" in text
    assert "off-box" in text and "test shards are off on this device" in text


# -- the command line ---------------------------------------------------------

@pytest.mark.parametrize("rest", [["--rootdir=/"], ["-k", "x"], ["tests/test_a.py::test_one"], ["../tests/test_a.py"], ["scripts/test"],
                                  ["/etc/passwd"], ["tests/sub/test_a.py"], ["tests/.hidden.py"]])
def test_a_remote_run_takes_test_files_and_nothing_else(rest, capsys):
    args = argparse.Namespace(tier="all", on="somewhere", split=False, base="main", timeout=60.0)
    assert test_on.main(args, rest, ROOT, lambda tier, file: [], True) == 4
    assert "test files tests/NAME.py only" in capsys.readouterr().err


def test_the_gate_takes_no_files_and_a_job_command_does_not_run_elsewhere(capsys):
    args = argparse.Namespace(tier="gate", on="somewhere", split=False, base="main", timeout=60.0)
    assert test_on.main(args, ["tests/test_a.py"], ROOT, lambda tier, file: [], True) == 4
    assert "the gate takes no files" in capsys.readouterr().err
    args.tier = "submit"
    assert test_on.main(args, [], ROOT, lambda tier, file: [], True) == 4
    assert "does not run on another device" in capsys.readouterr().err


def test_the_tiers_a_node_accepts_are_the_ones_the_runner_has():
    assert set(shard_spec.TIERS) | {"quick"} == set(test_on.TIERS)
    with pytest.raises(shard_spec.Refused):
        shard_spec.check_header({"id": ID, "tree_sha256": "0" * 64, "tier": "heavy"})


def test_nothing_is_sent_or_taken_until_the_experimental_feature_is_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path))
    args = argparse.Namespace(tier="all", on="somewhere", split=False, base="main", timeout=60.0)
    assert test_on.main(args, ["tests/test_a.py"], ROOT, lambda tier, file: [], True) == 4
    assert "experimental feature" in capsys.readouterr().err
    assert features.enabled(consent.FEATURE) is False and consent.run(["on"]) == 1
    assert "ml-stack features enable remote-tests" in capsys.readouterr().out
