"""`scripts/redteam_coverage.py --check` skips its scan only when nothing it reads has changed."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gates import _redteam_memo as memo, _store  # noqa: E402

MAP = """[[group]]
match = ["cli:*"]
status = "covered"
tests = ["tests/test_mapped.py::test_a"]
"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "cache"))
    monkeypatch.delenv(_store.FORCE, raising=False)
    tree = tmp_path / "tree"
    for name, text in {"src/ml_stack/a.py": "A = 1\n", "src/ml_stack/data/x.json": "{}\n",
                       "pyproject.toml": "[project]\n", memo.MAP: MAP,
                       "docs/redteam/coverage.json": "{}\n", "tests/test_mapped.py": "def test_a(): pass\n",
                       "tests/test_other.py": "def test_b(): pass\n", "README.md": "x\n"}.items():
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text, encoding="utf-8")
    return tree


@pytest.mark.parametrize("name", ["src/ml_stack/a.py", "src/ml_stack/data/x.json", "pyproject.toml",
                                  memo.MAP, "docs/redteam/coverage.json", "tests/test_mapped.py"])
def test_an_input_that_changes_changes_the_fingerprint(root, name):
    before = memo.inputs(root)
    assert memo.inputs(root) == before
    with (root / name).open("a", encoding="utf-8") as out:
        out.write("\n# changed\n")
    assert memo.inputs(root) != before


@pytest.mark.parametrize("name", ["README.md", "tests/test_other.py"])
def test_a_file_the_check_does_not_read_leaves_the_fingerprint_alone(root, name):
    before = memo.inputs(root)
    (root / name).write_text("changed\n", encoding="utf-8")
    assert memo.inputs(root) == before


def test_a_new_source_file_and_a_deleted_one_change_the_fingerprint(root):
    before = memo.inputs(root)
    (root / "src/ml_stack/b.py").write_text("", encoding="utf-8")
    added = memo.inputs(root)
    assert added != before
    (root / "src/ml_stack/b.py").unlink()
    assert memo.inputs(root) == before


def test_a_recorded_pass_is_found_only_for_the_same_inputs(root, monkeypatch):
    first = memo.inputs(root)
    assert memo.passed(root, first) is None
    memo.record(root, first, "3 surfaces")
    assert memo.passed(root, first) == "3 surfaces"
    assert memo.passed(root, "other") is None
    monkeypatch.setenv(_store.FORCE, "1")
    assert memo.passed(root, first) is None


def test_a_damaged_record_is_no_pass(root):
    first = memo.inputs(root)
    memo.record(root, first, "3 surfaces")
    (_store.directory() / f"{memo.name(root)}.json").write_text("{not json", encoding="utf-8")
    assert memo.passed(root, first) is None


def script():
    spec = importlib.util.spec_from_file_location("redteam_coverage_memo", REPO / "scripts/redteam_coverage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stubbed(monkeypatch, failures):
    """The script with its scan replaced by a counter, so only whether it ran is measured."""
    cov, scans = script(), []
    monkeypatch.setattr(cov, "discover", lambda: scans.append(1) or {})
    monkeypatch.setattr(cov, "check", lambda *args: list(failures))
    return cov, scans


def test_a_passing_check_is_not_scanned_again(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv(_store.FORCE, raising=False)
    cov, scans = stubbed(monkeypatch, [])
    assert cov.main(["--check"]) == 0
    first = capsys.readouterr().out
    assert cov.main(["--check"]) == 0
    assert capsys.readouterr().out == first
    assert len(scans) == 1


def test_a_failing_check_is_scanned_every_time(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv(_store.FORCE, raising=False)
    cov, scans = stubbed(monkeypatch, ["a surface has no row"])
    assert cov.main(["--check"]) == 1
    assert cov.main(["--check"]) == 1
    assert len(scans) == 2
    assert "FAIL a surface has no row" in capsys.readouterr().err


def test_write_and_a_forced_run_always_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv(_store.FORCE, raising=False)
    cov, scans = stubbed(monkeypatch, [])
    monkeypatch.setattr(cov, "OUT", tmp_path / "coverage.json")
    cov.main(["--check"])
    cov.main(["--write"])
    cov.main(["--check", "--write"])
    monkeypatch.setenv(_store.FORCE, "1")
    cov.main(["--check"])
    assert len(scans) == 4


@pytest.mark.slow
def test_the_real_check_prints_the_same_cached_as_computed(tmp_path):
    env = {**os.environ, "TMPDIR": str(tmp_path), "PYTHONPATH": str(REPO / "src")}
    env.pop(_store.FORCE, None)
    command = [sys.executable, str(REPO / "scripts/redteam_coverage.py"), "--check"]
    started = time.monotonic()
    computed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    scanned = time.monotonic() - started
    started = time.monotonic()
    cached = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    remembered = time.monotonic() - started
    forced = subprocess.run(command, env={**env, _store.FORCE: "1"}, capture_output=True,
                            text=True, check=False)
    print(f"scan {scanned:.1f}s, remembered {remembered:.1f}s")
    assert (computed.returncode, cached.returncode, forced.returncode) == (0, 0, 0)
    assert computed.stdout == cached.stdout == forced.stdout
    assert remembered < scanned / 2
