"""Test-run tree hashes, local artifacts and activity evidence."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from poolhouse.activity import gate
from tests.activity_support import entries, person, ring

__all__ = ["person", "ring"]
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "test"


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "test_a.py").write_text("def test_one():\n    assert True\n\ndef test_two():\n    assert True\n")
    return root


def script():
    loader = importlib.machinery.SourceFileLoader("scripts_test_runner", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def pytest_in(_root: Path) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_a.py"]


def recorded(repo: Path, status: int = 0) -> bool:
    junit = repo.parent / "result.xml"
    junit.write_text(f'<testsuite tests="{1 if status else 2}" failures="{status}"/>')
    return gate.record_run(gate.Run(repo, "full", pytest_in(repo), junit, status, 0.1))


def test_the_tree_hash_follows_the_content_and_ignores_the_real_index(repo):
    first = gate.tree_hash(repo)
    assert len(first) == 40 and gate.tree_hash(repo) == first
    (repo / "test_a.py").write_text("def test_one():\n    assert False\n")
    assert gate.tree_hash(repo) != first
    assert subprocess.run(["git", "-C", str(repo), "status", "--short"], capture_output=True,
                          text=True).stdout.startswith("??")


def test_the_tree_hash_is_empty_outside_a_checkout(tmp_path):
    assert gate.tree_hash(tmp_path / "nowhere") == ""


def test_a_green_run_is_recorded_against_its_tree_and_found_as_evidence(person, repo):
    assert recorded(repo)
    [e] = [x for x in entries() if x.kind == "test.result"]
    tree = gate.tree_hash(repo)
    assert (e.outcome, e.refs["tree"], e.refs["tier"], e.subject) == ("pass", tree, "full", f"tree:{tree[:12]}")
    assert (e.meta["passed"], e.meta["failed"], e.meta["exit"]) == (2, 0, 0) and e.meta["seconds"] >= 0
    assert gate.evidence(tree, "full").id == e.id
    assert gate.evidence(tree, "slow") is None


def test_a_red_run_is_not_evidence_and_a_changed_tree_has_none(person, repo):
    (repo / "test_a.py").write_text("def test_one():\n    assert False\n")
    assert recorded(repo, 1)
    [e] = [x for x in entries() if x.kind == "test.result"]
    assert (e.outcome, e.meta["failed"], e.meta["passed"]) == ("fail", 1, 0)
    assert gate.evidence(e.refs["tree"]) is None
    (repo / "test_a.py").write_text("def test_one():\n    assert True\n")
    assert gate.evidence(gate.tree_hash(repo)) is None


def test_the_same_command_on_the_same_tree_has_the_same_command_hash(person, repo):
    recorded(repo)
    recorded(repo)
    one, two = [x for x in entries() if x.kind == "test.result"]
    assert one.refs["command"] == two.refs["command"] and one.refs["tree"] == two.refs["tree"]


def test_runner_writes_local_evidence_without_user_activity(repo, monkeypatch, capsys):
    runner = script()
    runner.ROOT = repo

    def execute(command, workers, *, env):
        junit = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--junitxml=")))
        junit.write_text('<testsuite tests="3" failures="1"/>')
        return 1

    def forbidden(*args, **kwargs):
        pytest.fail("local test evidence accessed user activity")

    monkeypatch.setattr(runner.testslots_runner, "run_pytest", execute)
    monkeypatch.setattr(gate.writer, "record", forbidden)
    monkeypatch.setattr(gate.writer, "directory", forbidden)
    assert runner.run(runner.pytest_command(2, "test_a.py"), tier="quick") == 1
    artifact = Path(capsys.readouterr().out.split("test: evidence ", 1)[1].strip())
    try:
        payload = json.loads(artifact.read_text())
        assert payload["refs"]["tree"] == gate.tree_hash(repo)
        assert payload["refs"]["tier"] == "quick"
        assert payload["outcome"] == "fail"
        assert payload["meta"]["passed"] == 2
        assert payload["meta"]["failed"] == payload["meta"]["exit"] == 1
        assert payload["run"]["requested_command"] == runner.pytest_command(2, "test_a.py")
        assert payload["run"]["runner_interpreter"] == sys.executable
        assert payload["run"]["root"] == str(repo.resolve())
        assert payload["run"]["tree_after"] == payload["refs"]["tree"]
        if os.name == "nt":
            import win32security

            from poolhouse.windows_private import _user

            descriptor = win32security.GetNamedSecurityInfo(
                str(artifact), win32security.SE_FILE_OBJECT, win32security.OWNER_SECURITY_INFORMATION)
            assert descriptor.GetSecurityDescriptorOwner() == _user()
        else:
            assert artifact.stat().st_mode & 0o777 == 0o600
        assert payload["process"] == {"pid": os.getpid(), "started": psutil.Process().create_time()}
        assert not artifact.with_name(artifact.name.removesuffix(".result.json") + ".xml").exists()
    finally:
        artifact.unlink(missing_ok=True)


def test_runner_refuses_pass_when_source_changes_during_execution(repo, monkeypatch, capsys):
    runner = script()
    runner.ROOT = repo
    before = gate.tree_hash(repo)

    def execute(command, workers, *, env):
        (repo / "test_a.py").write_text("def test_changed(): pass\n")
        return 0

    monkeypatch.setattr(runner.testslots_runner, "run_pytest", execute)
    assert runner.run(runner.pytest_command(1, "test_a.py"), tier="quick") == 4
    captured = capsys.readouterr()
    assert "source tree changed" in captured.err
    artifact = Path(captured.out.split("test: evidence ", 1)[1].strip())
    try:
        payload = json.loads(artifact.read_text())
        assert payload["outcome"] == "fail"
        assert payload["refs"]["tree"] == before
        assert payload["run"]["tree_after"] != before
    finally:
        artifact.unlink(missing_ok=True)
