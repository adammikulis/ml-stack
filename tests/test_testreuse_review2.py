"""Second review of test result reuse: keys memoised across a launch, edits that revert, the stat wrapper,
commit evidence, notices that must not raise, and arguments the digest dropped."""

# ruff: noqa: F811
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import testreuse_key as keys
import testreuse_store as storage
from taskboard_kit import board  # noqa: F401  (fixture)
from test_testreuse import FILE, Recorder, hows, project  # noqa: F401  (fixture)
from test_testreuse_review import attempt, kinds

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import testboard, testruns

pytestmark = pytest.mark.slow
HELPER = "src/ml_stack/helper.py"


# -- N1: the key a result is stored under is the key of the bytes that ran ------------------------------
def test_a_test_file_edited_after_the_hit_check_is_not_stored_under_the_old_key(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    original = (project / FILE).read_text()

    def edit_then_run(real):
        def launch(command):
            (project / FILE).write_text(original + "# v2\n")
            return real(command)
        return launch

    attempt(project, store, launch=edit_then_run)
    (project / FILE).write_text(original)
    assert hows(attempt(project, store)[0]) == ["ran"]


# -- N2: a run that touched its own inputs stores nothing -----------------------------------------------
def test_an_input_edited_and_put_back_during_the_run_stores_nothing(project, tmp_path):
    store = storage.Store(tmp_path / "store")

    def flap(real):
        def launch(command):
            (project / HELPER).write_text("VALUE = 1  # v2\n")
            status = real(command)
            (project / HELPER).write_text("VALUE = 1\n")
            return status
        return launch

    attempt(project, store, launch=flap)
    assert "pass" not in kinds(store)


def test_a_file_created_in_an_ignored_directory_during_the_run_stores_nothing(project, tmp_path):
    store = storage.Store(tmp_path / "store")

    def generate(real):
        def launch(command):
            (project / "dist").mkdir()
            (project / "dist/gen.txt").write_text("x")
            return real(command)
        return launch

    attempt(project, store, launch=generate)
    assert "pass" not in kinds(store)


def test_an_unknown_tree_hash_stores_nothing(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store, tree="")
    assert "pass" not in kinds(store)


# -- N3: the stat wrapper must behave like os.stat, and unobservable APIs bar reuse --------------------
def test_copying_a_symlink_without_following_it_works_under_the_plugin(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(
        "import shutil\n\n\ndef test_a(tmp_path):\n    target = tmp_path / 't'\n    target.write_text('x')\n"
        "    link = tmp_path / 'l'\n    link.symlink_to(target)\n"
        "    shutil.copy2(link, tmp_path / 'c', follow_symlinks=False)\n    assert (tmp_path / 'c').is_symlink()\n")
    report, _ = attempt(project, store)
    assert report.outcomes[0].passed and "pass" in kinds(store)


@pytest.mark.parametrize("api", ["os.utime(tmp_path)", "os.fstat(0)", "sqlite3.connect(':memory:')",
                                 "list(os.scandir(tmp_path))"])
def test_a_file_using_an_api_the_runner_cannot_observe_is_not_reusable(project, tmp_path, api):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(f"import os\nimport sqlite3\n\n\ndef test_a(tmp_path):\n    {api}\n")
    report, _ = attempt(project, store)
    assert report.outcomes[0].passed and "pass" not in kinds(store)
    assert "cannot observe" in report.outcomes[0].detail


# -- N4: evidence names a clean commit whose tree it ran ----------------------------------------------------
def checkout(kit) -> Path:
    with GraphStore(kit.ws.base / "coordination.db") as graph:
        return Path(next(n["attrs"]["project"] for n in graph.nodes("task-worktree")
                         if n["attrs"].get("task") == kit.task["id"]))


def git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def json_fields(kit) -> dict:
    where = checkout(kit)
    return {"lookup": "a" * 64, "file": "tests/test_a.py", "outcome": "pass",
              "manifest": {"files": {}, "dirs": {}, "dists": [], "env": {}, "tree": ""},
              "manifest_digest": "d" * 64, "command": ["pytest"], "junit_sha256": "e" * 64,
              "counts": {"tests": 1, "failed": 0, "skipped": 0}, "tree": git(where, "rev-parse", "HEAD^{tree}"),
              "commit": git(where, "rev-parse", "HEAD"), "clean": True, "runner": {"pid": 1, "started": 1.0},
              "agent": {"id": "alice"}}


def put(kit, base: Path, **over) -> str:
    with GraphStore(kit.ws.base / "coordination.db") as graph:
        scope = testruns.task_scope(graph, kit.task["id"])
    fields = {**json_fields(kit), **over}
    return storage.Store(base / scope).put(fields, "pass" if fields["outcome"] == "pass" else "fail")


def start(kit, monkeypatch, tmp_path):
    base = tmp_path / "reuse"
    monkeypatch.setattr(testruns, "STORE_BASE", base)
    kit.board.claim(kit.child, kit.task["id"], kit.allocation["allocation_id"])
    return base


def test_a_clean_entry_for_the_named_commit_is_accepted_and_says_what_it_covers(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    entry = put(board, base)
    head = git(checkout(board), "rev-parse", "HEAD")
    saved = board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": entry, "commit": head})
    assert saved["test_evidence"][0]["covers"] == "tests/test_a.py only"


def test_evidence_without_a_commit_is_refused(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="commit"):
        board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": put(board, base)})


def test_an_entry_from_a_dirty_checkout_is_never_evidence(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    entry = put(board, base, clean=False)
    head = git(checkout(board), "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="uncommitted"):
        board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": entry, "commit": head})


def test_an_entry_whose_tree_is_not_the_commits_tree_is_refused(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    entry = put(board, base, tree="t" * 40)
    head = git(checkout(board), "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="tree"):
        board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": entry, "commit": head})


# -- N5: notices and the session never raise out of a run ---------------------------------------------------
class Boom:
    base = Path("/nonexistent")

    class bus:
        @staticmethod
        def get(seq):
            raise ZeroDivisionError("bus")

    class board:
        @staticmethod
        def list(token):
            return [{"name": "#p", "project": True, "member": True}]

        @staticmethod
        def listeners(row):
            raise ZeroDivisionError("listeners")

    @staticmethod
    def send(token, to, kind, body, **kw):
        return {"seq": 3}


def test_job_done_survives_any_error_from_the_board():
    events = testboard.BoardEvents(testboard.Acting(Boom(), "t", {"id": "alice", "parent": ""}))
    events.job_done("j", {"argv": ["fast"]}, {"exit": 0}, 7)
    events.claimed("tests/x.py", "k" * 64)


def test_acting_survives_any_error_from_the_workspace(monkeypatch):
    monkeypatch.setattr(testboard.cli, "_context", lambda args: (_ for _ in ()).throw(ZeroDivisionError("x")))
    assert testboard.acting("someone") is None


# -- N6: an error that belongs to no file is not a pass -----------------------------------------------------
def test_a_session_level_error_in_the_junit_stores_nothing(project, tmp_path):
    store = storage.Store(tmp_path / "store")

    def add_error(real):
        def launch(command):
            real(command)
            junit = Path(next(w.split("=", 1)[1] for w in command if w.startswith("--junitxml=")))
            tree = ET.parse(junit)  # noqa: S314 - the file pytest just wrote
            suite = next(tree.getroot().iter("testsuite"))
            ET.SubElement(ET.SubElement(suite, "testcase", classname="", name="session"), "error", message="x")
            tree.write(junit)
            return 1
        return launch

    attempt(project, store, launch=add_error)
    assert "pass" not in kinds(store)


# -- arguments and plugins ------------------------------------------------------------------------------------
def test_option_values_the_digest_does_not_know_are_still_in_it(project):
    base = keys.lookup(project, FILE, ["-q"]).key
    assert keys.lookup(project, FILE, ["-q", "--log-cli-level", "INFO"]).key != keys.lookup(
        project, FILE, ["-q", "--log-cli-level", "DEBUG"]).key
    assert keys.lookup(project, FILE, ["-q", FILE, "tests"]).key == base


def test_a_pytest_plugins_module_named_by_a_conftest_is_part_of_the_hit(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/conftest.py").write_text("pytest_plugins = ['ml_stack.other']\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]
    (project / "src/ml_stack/other.py").write_text("OTHER = 2\n")
    assert hows(attempt(project, store)[0]) == ["ran"]


def test_a_dash_p_module_is_part_of_the_hit(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store, extra=("-p", "ml_stack.other"))
    assert hows(attempt(project, store, extra=("-p", "ml_stack.other"))[0]) == ["reused"]
    (project / "src/ml_stack/other.py").write_text("OTHER = 2\n")
    assert hows(attempt(project, store, extra=("-p", "ml_stack.other"))[0]) == ["ran"]


def submission(commit: str) -> dict:
    return {"artifacts": {"r.json": "a" * 64}, "checks": [{"name": "Worker claims tests", "passed": True}],
            "summary": "done", "provenance": {"commit": commit, "environment": "x", "model": "qwen", "runtime": "r"}}


def test_a_proposal_carries_the_verified_entries_it_cites(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    entry = put(board, base)
    head = git(checkout(board), "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="at most 16"):
        board.board.submit(board.child, board.task["id"], {**submission(head), "test_entries": "nope"})
    proposal = board.board.submit(board.child, board.task["id"], {**submission(head), "test_entries": [entry]})
    assert [f["id"] for f in proposal["test_evidence"]] == [entry]


def test_a_proposal_for_another_commit_cannot_cite_the_entry(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="commit"):
        board.board.submit(board.child, board.task["id"], {**submission("f" * 40), "test_entries": [put(board, base)]})


def test_the_project_is_the_tasks_checkout_not_the_working_directory(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    entry = put(board, base)
    head = git(checkout(board), "rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)
    saved = board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": entry, "commit": head})
    assert [f["id"] for f in saved["test_evidence"]] == [entry]


def test_an_entry_from_the_working_directorys_project_is_not_accepted(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    entry = storage.Store(base / testruns.scope(tmp_path)).put({
        **json_fields(board), "lookup": "b" * 64})
    monkeypatch.chdir(tmp_path)
    head = git(checkout(board), "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="verifies"):
        board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": entry, "commit": head})


def test_a_failing_entry_is_not_evidence_of_passing(board, tmp_path, monkeypatch):
    base = start(board, monkeypatch, tmp_path)
    entry = put(board, base, outcome="fail")
    head = git(checkout(board), "rev-parse", "HEAD")
    with pytest.raises(ValueError, match=r"verifies|pass"):
        board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": entry, "commit": head})


def test_the_services_environment_does_not_choose_the_store(board, tmp_path, monkeypatch):
    forged = tmp_path / "forged"
    start(board, monkeypatch, tmp_path)
    entry = put(board, forged)
    monkeypatch.setenv("DEV_TEST_REUSE_DIR", str(forged))
    head = git(checkout(board), "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="verifies"):
        board.board.checkpoint(board.child, board.task["id"], {"summary": "ok", "test_entry": entry, "commit": head})


def test_removing_a_temporary_directory_is_not_a_write_into_the_checkout(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(
        "import tempfile\n\n\ndef test_a():\n    with tempfile.TemporaryDirectory() as folder:\n"
        "        open(folder + '/x', 'w').write('1')\n")
    report, _ = attempt(project, store)
    assert report.outcomes[0].passed and "pass" in kinds(store)


def test_library_code_that_uses_sqlite_does_not_bar_a_file_that_does_not(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / HELPER).write_text("import sqlite3\n\nVALUE = 1\n\n\ndef db():\n    return sqlite3.connect(':memory:')\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]


def test_a_module_an_earlier_file_already_imported_is_still_in_a_later_files_manifest(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    for name in ("a", "b"):
        (project / f"tests/test_{name}.py").write_text(
            "from ml_stack.helper import VALUE\n\n\ndef test_x():\n    assert VALUE == 1\n")
    attempt(project, store, "tests/test_a.py", "tests/test_b.py")
    (project / HELPER).write_text("VALUE = 1  # edited\n")
    assert hows(attempt(project, store, "tests/test_a.py", "tests/test_b.py")[0]) == ["ran", "ran"]


def test_a_file_edited_between_the_hit_check_and_the_start_of_the_launch_is_not_stored(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    original = (project / FILE).read_text()

    class Editor(Recorder):
        def claimed(self, file, key):
            (project / file).write_text(original + "# v2\n")
            return 0

    attempt(project, store, events=Editor())
    (project / FILE).write_text(original)
    assert hows(attempt(project, store)[0]) == ["ran"]
