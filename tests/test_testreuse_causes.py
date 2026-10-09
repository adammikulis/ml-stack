"""Why test result reuse never hit in normal use, and the proof that it does now without a false green:
bytecode caches are not test output, a pinned distribution is compared the way an import finds it, a
``heavy`` label never bars a file, and one store serves every worktree of a repository."""

# ruff: noqa: F811
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
import testreuse_facts as facts
import testreuse_key as keys
import testreuse_plugin as plugin
import testreuse_store as storage
from test_testreuse import FILE, hows, project  # noqa: F401  (fixture)
from test_testreuse_review import attempt

from ml_stack.activity import reuse
from ml_stack.workspace import testruns

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[1]
OUTSIDE = "/nonexistent-site-packages/pkg"


def writer(project, name: str, body: str) -> str:
    """A test file under ``project`` whose one test runs ``body``; returns its path."""
    rel = f"tests/{name}"
    (project / rel).write_text("import os\n\n\ndef test_it():\n" + "".join(f"    {line}\n" for line in body.splitlines()))
    return rel


def violations(path: str, mode: str) -> set[str]:
    """What the plugin's audit records for opening ``path`` with ``mode`` as a test body would."""
    plugin.FILES.pop("probe", None)
    plugin.CURRENT.append("probe")
    try:
        plugin._audit("open", (path, mode, 0))
        return set(plugin.FILES["probe"]["violations"])
    finally:
        plugin.CURRENT.pop()
        plugin.FILES.pop("probe", None)


# -- (1) bytecode caches are regenerable, not output ---------------------------------------------------------
@pytest.mark.parametrize("where", [f"{OUTSIDE}/__pycache__/m.cpython-313.pyc", f"{OUTSIDE}/__pycache__/m.cpython-313.pyc.123"])
def test_a_bytecode_file_written_anywhere_is_not_a_violation(where):
    assert violations(where, "wb") == set()


def test_a_real_write_outside_the_tree_is_still_a_violation():
    assert violations(f"{OUTSIDE}/data.txt", "w") == {f"wrote {OUTSIDE}/data.txt"}
    assert violations(f"{OUTSIDE}/__pycache__/notes.txt", "w")
    assert violations(f"{OUTSIDE}/evil.pyc", "wb")


def test_creating_a_pycache_directory_is_allowed_and_creating_any_other_is_not():
    plugin.FILES.pop("probe", None)
    plugin.CURRENT.append("probe")
    try:
        plugin._audit("os.mkdir", (f"{OUTSIDE}/__pycache__", 0o777, -1))
        plugin._audit("os.mkdir", (f"{OUTSIDE}/newdir", 0o777, -1))
        assert plugin.FILES["probe"]["violations"] == {f"mkdir {OUTSIDE}/newdir"}
    finally:
        plugin.CURRENT.pop()
        plugin.FILES.pop("probe", None)


def test_a_run_that_writes_bytecode_outside_the_tree_is_stored_and_a_run_that_writes_data_there_is_not(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    good = writer(project, "test_bytecode.py", f"""try:
    open({OUTSIDE + '/__pycache__/m.cpython-313.pyc'!r}, 'wb')
except OSError:
    pass""")
    bad = writer(project, "test_data.py", f"""try:
    open({OUTSIDE + '/data.txt'!r}, 'w')
except OSError:
    pass""")
    report, _ = attempt(project, store, good, bad)
    details = {o.file: o.detail for o in report.outcomes}
    assert "not stored" not in details[good] and f"wrote {OUTSIDE}/data.txt" in details[bad]
    assert [reuse.entry(store.folder, r["id"])["file"] for r in reuse.rows(store.folder)] == [good]
    assert hows(attempt(project, store, good, bad)[0]) == ["reused", "ran"]


def test_a_stored_result_survives_deleting_the_bytecode_and_changing_the_bytecode_setting(project, tmp_path, monkeypatch):
    store = storage.Store(tmp_path / "store")
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    assert hows(attempt(project, store)[0]) == ["ran"]
    for cache in project.rglob("__pycache__"):
        shutil.rmtree(cache)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    report, launches = attempt(project, store)
    assert hows(report) == ["reused"] and launches == []
    assert not list(project.rglob("*.pyc"))
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE")
    assert hows(attempt(project, store)[0]) == ["reused"]


# -- (2) a pinned distribution is compared the way an import finds it ------------------------------------------
def test_pins_hold_for_an_installed_distribution_and_name_the_one_that_changed():
    version = metadata.version("pytest")
    assert facts.changed_pins([f"pytest=={version}"]) == ""
    said = facts.changed_pins([f"pytest=={version}x"])
    assert "pytest" in said and version in said


def test_a_distribution_installed_twice_resolves_to_the_copy_an_import_finds_first(tmp_path, monkeypatch):
    for place, version in (("first", "2.0"), ("second", "1.0")):
        info = tmp_path / place / f"zzreuse_twice-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: zzreuse-twice\nVersion: {version}\n")
    monkeypatch.syspath_prepend(str(tmp_path / "second"))
    monkeypatch.syspath_prepend(str(tmp_path / "first"))
    assert facts.installed_pins(["zzreuse-twice==2.0"]) == ["zzreuse-twice==2.0"]
    assert facts.changed_pins(["zzreuse-twice==2.0"]) == ""
    assert "1.0" not in facts.changed_pins(["zzreuse-twice==2.0"])


def test_a_pin_that_is_no_longer_installed_is_a_change():
    assert facts.changed_pins(["zzreuse-absent==1.0"]) == "installed distribution changed: zzreuse-absent 1.0 -> missing"


def test_an_unstored_pass_says_which_input_changed(project, tmp_path, monkeypatch):
    store = storage.Store(tmp_path / "store")
    monkeypatch.setattr(keys, "manifest_holds", lambda root, manifest, closures: "changed: src/ml_stack/helper.py")
    report, _ = attempt(project, store)
    assert "(not stored: changed: src/ml_stack/helper.py)" in report.outcomes[0].detail
    assert reuse.rows(store.folder) == []


# -- (3) a label schedules, it never blocks ---------------------------------------------------------------------
def test_a_file_listed_as_heavy_and_marked_so_at_run_time_is_reused_like_any_other(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/heavy-modules.txt").write_text("test_a.py\n")
    (project / "tests/conftest.py").write_text(
        "import pytest\n\n\ndef pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.add_marker(pytest.mark.heavy)\n")
    first, _ = attempt(project, store)
    assert hows(first) == ["ran"] and "not reusable" not in first.outcomes[0].detail
    second, launches = attempt(project, store)
    assert hows(second) == ["reused"] and launches == []


# -- (4) one store serves every worktree of a repository -----------------------------------------------------------
def git(where: Path, *words: str) -> None:
    subprocess.run(["git", "-C", str(where), *words], check=True, capture_output=True)


def test_every_worktree_of_a_repository_without_an_origin_shares_one_scope(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    git(main, "init", "-q")
    git(main, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    git(main, "worktree", "add", "-q", "--detach", str(tmp_path / "second"))
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    assert testruns.scope(main) == testruns.scope(tmp_path / "second") != testruns.scope(other)


def run_tests(where: Path, store: Path, *files: str) -> tuple[int, str]:
    outer = {k: v for k, v in os.environ.items() if not k.startswith("DEV_TEST_")}  # not a run nested in a lease
    env = {**outer, "DEV_TEST_REUSE_DIR": str(store), "DEV_TEST_REUSE_CANARY": "0"}
    done = subprocess.run([sys.executable, "scripts/test", "all", *files], cwd=where, env=env, capture_output=True,
                          text=True, check=False)
    return done.returncode, done.stdout + done.stderr


def test_a_real_run_twice_reports_zero_then_all_reused_and_a_second_worktree_reuses_the_first(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    for tree in (first, second):
        git(ROOT, "worktree", "add", "-q", "--detach", str(tree), "HEAD")
    try:
        names = ("tests/test_reuse_probe_a.py", "tests/test_reuse_probe_b.py")
        for tree in (first, second):
            for number, name in enumerate(names):
                (tree / name).write_text(f"def test_probe():\n    assert {number} + 1 == {number + 1}\n")
        store = tmp_path / "store"
        one, said = run_tests(first, store, *names)
        assert one == 0 and "test: 2 file(s) ran, 0 reused" in said, said
        again, said = run_tests(first, store, *names)
        assert again == 0 and "test: 0 file(s) ran, 2 reused" in said, said
        elsewhere, said = run_tests(second, store, *names)
        assert elsewhere == 0 and "test: 0 file(s) ran, 2 reused" in said, said
    finally:
        for tree in (first, second):
            git(ROOT, "worktree", "remove", "--force", str(tree))
