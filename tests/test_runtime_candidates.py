import hashlib
import json
import runpy
from pathlib import Path

import pytest

from ml_stack.fleet import runtime_candidates as candidates

COMMIT = "a" * 40


def test_frozen_headless_fixed_dependency_probe(monkeypatch):
    from ml_stack import agent_dependency

    called = []
    monkeypatch.setattr(candidates.sys, "argv", ["ml-stack-headless", "--check-agent-runtime"])
    monkeypatch.setattr(agent_dependency, "report", lambda: called.append(True) or 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(Path(__file__).parents[1] / "packaging/launcher-headless.py"), run_name="__main__")
    assert stopped.value.code == 0 and called == [True]


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    source = tmp_path / "reviewed.zip"
    source.write_bytes(b"reviewed immutable artifact")
    monkeypatch.setattr(candidates, "_verify", lambda archive, commit: "Poolside.app")
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_candidate_retains_exact_identity_and_rechecks_artifact(tmp_path, artifact):
    source, sha256 = artifact
    registry = candidates.Candidates(tmp_path / "owned")
    row = registry.register(source, COMMIT, sha256)
    assert row["commit"] == COMMIT
    assert row["sha256"] == sha256
    assert registry.select("Poolside.app") == row
    assert registry.select("Other.app") is None
    (registry.path / (sha256 + ".zip")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        registry.select("Poolside.app")


def test_registration_checks_expected_digest_before_execution(tmp_path, artifact, monkeypatch):
    source, _ = artifact
    called = []
    monkeypatch.setattr(candidates, "_verify", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="reviewed artifact"):
        candidates.Candidates(tmp_path / "owned").register(source, COMMIT, "b" * 64)
    assert not called


@pytest.mark.parametrize("commit,digest", [("a" * 8, "b" * 64), (COMMIT, "b" * 8), ("../" + "a" * 37, "b" * 64)])
def test_candidate_requires_full_hex_identity(tmp_path, commit, digest):
    with pytest.raises(ValueError, match="full lowercase"):
        candidates.Candidates(tmp_path / "owned").register(tmp_path / "absent", commit, digest)


def test_private_registry_and_artifacts_refuse_symlinks(tmp_path, artifact):
    source, sha256 = artifact
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="plain owned"):
        candidates.Candidates(linked)
    registry = candidates.Candidates(tmp_path / "owned")
    (registry.path / (sha256 + ".zip")).symlink_to(source)
    with pytest.raises(ValueError, match="symbolic link"):
        registry.register(source, COMMIT, sha256)


def test_candidate_requires_private_storage(tmp_path):
    target = tmp_path / "runtime-candidates"
    target.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="account-private"):
        candidates.Candidates(tmp_path)


def test_candidate_refuses_writable_managed_parent(tmp_path):
    root = tmp_path / "owned"
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    with pytest.raises(ValueError, match="other writers"):
        candidates.Candidates(root)


@pytest.mark.parametrize("name", ["candidates.db", "candidates.db.wal", "candidates.db.shadow", "candidates.db.shm", "candidates.lock"])
def test_registry_auxiliary_symlinks_are_refused(tmp_path, name):
    registry = candidates.Candidates(tmp_path / "owned")
    (registry.path / name).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="auxiliary"):
        registry.select("Poolside.app")


@pytest.mark.parametrize("ready,commit", [(False, COMMIT), (True, "b" * 40)])
def test_candidate_probe_refuses_unready_or_wrong_source(tmp_path, monkeypatch, ready, commit):
    monkeypatch.setattr(candidates.sys, "platform", "darwin")

    def unpack(archive, into):
        executable = into / "Poolside.app/Contents/MacOS/ml-stack-headless"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"frozen")

    def run(argv, **kwargs):
        if argv[0] == "codesign":
            return None
        return type("Result", (), {"stdout": json.dumps({"commit": commit, "platform": "darwin", "machine": candidates.platform.machine(), "runtime_ready": ready})})()

    monkeypatch.setattr(candidates, "unpack", unpack)
    monkeypatch.setattr(candidates.subprocess, "run", run)
    with pytest.raises(ValueError, match="readiness"):
        candidates._verify(Path("archive.zip"), COMMIT)


def test_signature_failure_never_executes_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates.sys, "platform", "darwin")
    monkeypatch.setattr(candidates, "unpack", lambda archive, into: (into / "Poolside.app").mkdir())
    called = []

    def run(argv, **kwargs):
        called.append(argv)
        raise candidates.subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(candidates.subprocess, "run", run)
    with pytest.raises(candidates.subprocess.CalledProcessError):
        candidates._verify(Path("archive.zip"), COMMIT)
    assert len(called) == 1 and called[0][0] == "codesign"
