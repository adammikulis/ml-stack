"""Third review of test result reuse: runner-owned files, exit status 5, a live tree callable, the snapshot's
reach, same-size edits, store failures and board error classes."""

# ruff: noqa: F811
from __future__ import annotations

import http.client
import os
import pickle
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import testreuse_key as keys
import testreuse_plugin as plugin
import testreuse_store as storage
from test_testreuse import FILE, Recorder, hows, project  # noqa: F401  (fixture)
from test_testreuse_review import attempt, kinds

from ml_stack.activity import reuse
from ml_stack.workspace import testboard

pytestmark = pytest.mark.slow
HELPER = "src/ml_stack/helper.py"


def last_entry(store):
    return reuse.entry(store.folder, reuse.rows(store.folder)[-1]["id"])


# -- (1) files the runner itself writes during a launch are not the checkout moving -------------------------
def test_a_runner_owned_file_written_during_the_launch_does_not_move_the_tree(project, tmp_path):
    store = storage.Store(tmp_path / "store")

    def timing(real):
        def launch(command):
            status = real(command)
            (project / ".full-tier-last.json").write_text("{}")
            return status
        return launch

    report, _ = attempt(project, store, launch=timing)
    assert report.status == 0 and "pass" in kinds(store)


# -- (2) exit status 5 stays non-zero ---------------------------------------------------------------------------
def test_a_selection_that_collects_nothing_exits_five_and_passes_no_file(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    report, _ = attempt(project, store, extra=("-k", "nomatch"))
    assert report.status == 5 and not report.outcomes[0].passed and kinds(store) == []


def test_a_module_level_skip_is_passed_and_unstored(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text("import pytest\n\npytest.skip('not here', allow_module_level=True)\n")
    report, _ = attempt(project, store)
    assert report.outcomes[0].passed and kinds(store) == [] and report.status == 0


# -- (3) the tree is read live, and commit, tree and cleanliness are read together --------------------------------
def git(path, *args):
    return subprocess.run(["git", "-C", str(path), "-c", "user.name=T", "-c", "user.email=t@example.invalid", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_an_entry_carries_the_commit_and_cleanliness_of_the_moment_the_tests_ran(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / ".gitignore").write_text("__pycache__/\n")
    git(project, "init", "-q")
    git(project, "add", ".")
    git(project, "commit", "-q", "-m", "one")

    class Committer(Recorder):
        def claimed(self, file, key):
            git(project, "commit", "-q", "--allow-empty", "-m", "two")
            return 0

    attempt(project, store, events=Committer())
    entry = last_entry(store)
    assert entry["commit"] == git(project, "rev-parse", "HEAD") and entry["clean"] is True


def test_a_tree_that_differs_from_the_one_at_the_hit_check_stores_nothing(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    calls = []

    def tree():
        calls.append(1)
        return "a" if len(calls) == 1 else "b"

    attempt(project, store, tree=tree)
    assert "pass" not in kinds(store)


def test_the_runner_hands_the_session_a_live_tree_function():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "test").read_text()
    assert "lambda: tree_hash(ROOT)" in text


# -- (5) what the snapshot watches --------------------------------------------------------------------------------
def test_edits_in_other_agents_worktrees_do_not_move_the_tree(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    other = project / ".claude/worktrees/other"
    other.mkdir(parents=True)
    (other / "f.txt").write_text("1")

    def edit_sibling(real):
        def launch(command):
            (other / "f.txt").write_text("22")
            return real(command)
        return launch

    report, _ = attempt(project, store, launch=edit_sibling)
    assert report.status == 0 and "pass" in kinds(store)


def test_a_directory_named_venv_below_the_root_is_still_watched(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/venv").mkdir()
    (project / "tests/venv/data.txt").write_text("1")

    def edit(real):
        def launch(command):
            (project / "tests/venv/data.txt").write_text("22")
            return real(command)
        return launch

    attempt(project, store, launch=edit)
    assert "pass" not in kinds(store)


def test_a_checkout_too_large_to_watch_stores_nothing_and_still_runs(project, tmp_path, monkeypatch):
    store = storage.Store(tmp_path / "store")
    monkeypatch.setattr(keys, "SNAPSHOT_BUDGET_S", -1.0)
    report, _ = attempt(project, store)
    assert report.status == 0 and report.outcomes[0].passed and "pass" not in kinds(store)


# -- (6) same-size edits that restore the modification time -----------------------------------------------------------
def test_a_same_size_edit_with_the_old_modification_time_is_recorded_by_content(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / HELPER).write_text("VALUE = 1\n")
    (project / FILE).write_text("from ml_stack.helper import VALUE\n\n\ndef test_a():\n    assert VALUE in (1, 2)\n")
    attempt(project, store)
    stat = (project / HELPER).stat()

    class Sneaker(Recorder):
        def claimed(self, file, key):
            (project / HELPER).write_text("VALUE = 2\n")
            os.utime(project / HELPER, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            return 0

    attempt(project, store, extra=("-k", "test_a"), events=Sneaker())
    (project / HELPER).write_text("VALUE = 1\n")
    assert hows(attempt(project, store, extra=("-k", "test_a"))[0]) == ["ran"]


# -- (7) failures after the tests passed ---------------------------------------------------------------------------------
def test_a_store_that_cannot_be_written_means_ran_not_stored(project, tmp_path, monkeypatch):
    store = storage.Store(tmp_path / "store")

    def broken(self, fields, kind="pass"):
        raise OSError("disk full")

    monkeypatch.setattr(storage.Store, "put", broken)
    report, _ = attempt(project, store)
    assert report.status == 0 and report.outcomes[0].passed and "not stored" in report.outcomes[0].detail


def test_a_truncated_junit_file_does_not_abort_the_runner(project, tmp_path):
    store = storage.Store(tmp_path / "store")

    def truncate(real):
        def launch(command):
            status = real(command)
            junit = Path(next(w.split("=", 1)[1] for w in command if w.startswith("--junitxml=")))
            junit.write_text(junit.read_text()[:40])
            return status
        return launch

    report, _ = attempt(project, store, launch=truncate)
    assert "pass" not in kinds(store) and report.status != 0


def test_claims_of_earlier_files_are_released_when_a_later_file_fails_to_plan(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/test_b.py").write_text("def test_b():\n    pass\n")

    class Breaks(Recorder):
        def claimed(self, file, key):
            if file.endswith("test_b.py"):
                raise RuntimeError("board")
            return 0

    with pytest.raises(RuntimeError):
        attempt(project, store, "tests/test_a.py", "tests/test_b.py", events=Breaks())
    key = keys.lookup(project, "tests/test_a.py", [sys.executable, "-m", "pytest", "-q", "tests/test_a.py",
                                                  "tests/test_b.py"]).key
    assert store.claimed(key) is None


# -- (8) board error classes -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("error", [sqlite3.Error("x"), subprocess.SubprocessError("x"), http.client.HTTPException("x"),
                                   struct.error("x"), StopIteration(), pickle.PickleError("x")])
def test_a_board_action_survives_these_error_classes(error):
    assert testboard.guarded("x", lambda: (_ for _ in ()).throw(error)) is None


# -- (9) the stat wrapper and the sqlite audit event ----------------------------------------------------------------------
def test_stat_accepts_keywords_and_directory_descriptors_under_the_plugin(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(
        "import os\n\n\ndef test_a():\n    folder = os.path.dirname(__file__)\n    os.stat(path=__file__)\n"
        "    fd = os.open(folder, os.O_RDONLY)\n    try:\n"
        "        try:\n            os.stat('flag.txt', dir_fd=fd)\n        except FileNotFoundError:\n            pass\n"
        "        else:\n            raise AssertionError('flag')\n    finally:\n        os.close(fd)\n")
    report, _ = attempt(project, store)
    assert report.outcomes[0].passed and "pass" in kinds(store)
    (project / "tests/flag.txt").write_text("x")
    assert hows(attempt(project, store)[0]) == ["ran"]


def test_a_path_relative_to_a_directory_descriptor_is_resolved_against_that_directory(tmp_path):
    descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        assert Path(plugin._resolve("x.txt", descriptor)).resolve() == (tmp_path / "x.txt").resolve()
    finally:
        os.close(descriptor)


def test_a_sqlite_connection_to_a_checkout_file_is_a_recorded_read():
    row = plugin.ROOT / "x.db"
    plugin.CURRENT.append("tests/zz.py")
    try:
        plugin._audit("sqlite3.connect", (str(row),))
    finally:
        plugin.CURRENT.pop()
    assert "x.db" in plugin.FILES["tests/zz.py"]["reads"]
    plugin.FILES.pop("tests/zz.py")


APIS = [("import os\n", "os.utime(tmp_path)"), ("import os\n", "os.fstat(0)"),
        ("import os\n", "list(os.scandir(tmp_path))"), ("import sqlite3\n", "sqlite3.connect(':memory:')")]


@pytest.mark.parametrize(("imports", "call"), APIS)
def test_each_unobservable_api_alone_bars_a_file(project, tmp_path, imports, call):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(f"{imports}\n\ndef test_a(tmp_path):\n    {call}\n")
    report, _ = attempt(project, store)
    assert report.outcomes[0].passed and "pass" not in kinds(store) and "cannot observe" in report.outcomes[0].detail


# -- (4) identity comes from the environment the harness assigned ---------------------------------------------------------
def test_the_runner_takes_its_identity_from_the_environment_only():
    import inspect

    import testjobs

    text = (Path(__file__).resolve().parents[1] / "scripts" / "test").read_text()
    assert not inspect.signature(testboard.acting).parameters
    assert '"--agent"' not in text and '"--label"' not in text and not hasattr(testboard, "LABEL_ENV")
    assert testjobs.Jobs.passthrough({"agent": "x", "label": "y", "task": "t"}) == ["--task", "t"]
