"""Defects found reviewing test result reuse, each reproduced against real pytest runs of tiny projects,
real job tables and the real task board."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# ruff: noqa: F811
import testjobs
import testreuse_key as keys
import testreuse_plugin as plugin
import testreuse_run as run
import testreuse_store as storage
from taskboard_kit import board  # noqa: F401  (fixture)
from test_testreuse import (  # noqa: F401  (fixture)
    AGENT,
    FILE,
    SCRIPTS,
    Recorder,
    hows,
    launcher,
    project,
)

from ml_stack.activity import reuse
from ml_stack.workspace import testboard, testruns

pytestmark = pytest.mark.slow


def attempt(root, store, *files, **options):
    """Run files through a Session; options: ``launch`` (wraps the pytest launcher), ``tree``, ``extra``, ``events``."""
    named = list(files or (FILE,))
    command = [sys.executable, "-m", "pytest", "-q", *options.get("extra", ()), *named]
    session = run.Session(root, command, named, store, options.get("events") or Recorder())
    session.draw = lambda: 1.0
    runs: list = []
    base = launcher(root, runs)
    wrap = options.get("launch")
    report = session.run(wrap(base) if wrap else base, options.get("tree", "treehash0001"), True)
    return report, runs


def kinds(store):
    return [r["kind"] for r in reuse.rows(store.folder)]


# -- (1) a run that did not finish cleanly stores nothing ---------------------------------------------
def test_a_launch_status_other_than_zero_stores_no_pass(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    report, _ = attempt(project, store, launch=lambda real: lambda command: real(command) or 2)
    assert report.status != 0 and "pass" not in kinds(store)


def test_a_junit_with_fewer_tests_than_were_collected_stores_no_pass(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text("def test_a():\n    pass\n\n\ndef test_b():\n    pass\n")

    def drop_one(real):
        def launch(command):
            status = real(command)
            junit = Path(next(w.split("=", 1)[1] for w in command if w.startswith("--junitxml=")))
            tree = ET.parse(junit)  # noqa: S314 - the file pytest just wrote
            suite = next(tree.getroot().iter("testsuite"))
            suite.remove(next(suite.iter("testcase")))
            tree.write(junit)
            return status
        return launch

    attempt(project, store, launch=drop_one)
    assert "pass" not in kinds(store)
    assert hows(attempt(project, store)[0]) == ["ran"]


# -- (2) a tree that moves during the run is never stored ---------------------------------------------
def test_a_tree_that_changed_during_the_run_stores_nothing_and_fails_the_run(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    calls = []
    report, _ = attempt(project, store, tree=lambda: "tree-after" if len(calls) > 1 else calls.append(1) or "tree-before")
    assert kinds(store) == [] and report.status == 4


# -- (3) the conftest chain is part of the closure ---------------------------------------------------
def test_a_module_imported_only_by_conftest_is_part_of_the_hit(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/conftest.py").write_text("from ml_stack.other import OTHER  # noqa\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]
    (project / "src/ml_stack/other.py").write_text("OTHER = 2\n")
    assert hows(attempt(project, store)[0]) == ["ran"]


def test_the_runners_own_code_is_part_of_the_key(project):
    (project / "scripts").mkdir()
    (project / "scripts/testreuse_plugin.py").write_text("# one\n")
    before = keys.lookup(project, FILE, ["-q"]).key
    (project / "scripts/testreuse_plugin.py").write_text("# two\n")
    assert keys.lookup(project, FILE, ["-q"]).key != before


# -- (4) run-time imports do not depend on file order ------------------------------------------------
def test_a_module_imported_by_computed_name_is_not_missed_by_the_second_file(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    body = "import importlib\n\nNAME = 'ml_stack.helper'\n\n\ndef test_x():\n    assert importlib.import_module(NAME).VALUE == 1\n"
    (project / "tests/test_a.py").write_text(body)
    (project / "tests/test_b.py").write_text(body)
    attempt(project, store, "tests/test_a.py", "tests/test_b.py")
    (project / "src/ml_stack/helper.py").write_text("VALUE = 1  # edited\n")
    assert hows(attempt(project, store, "tests/test_a.py", "tests/test_b.py")[0]) == ["ran", "ran"]


# -- (5) a process spawned from library code is covered ----------------------------------------------
def test_a_child_process_spawned_by_library_code_does_not_hide_a_data_change(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "src/ml_stack/data.txt").write_text("1")
    (project / "src/ml_stack/helper.py").write_text(
        "import subprocess\nimport sys\nfrom pathlib import Path\n\nDATA = Path(__file__).with_name('data.txt')\n"
        "VALUE = int(subprocess.run([sys.executable, '-c', f'print(open({str(DATA)!r}).read())'],\n"
        "                           capture_output=True, text=True, check=True).stdout)\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]
    (project / "src/ml_stack/data.txt").write_text("2")
    assert hows(attempt(project, store)[0]) == ["ran"]


def test_documents_and_packaging_are_part_of_a_spawning_files_manifest(project):
    (project / "tests/test_a.py").write_text("import subprocess\n\n\ndef test_a():\n    subprocess.run(['true'])\n")
    (project / "docs").mkdir()
    (project / "docs/guide.md").write_text("one")
    closures = keys.Closures(project)
    before = keys.build_manifest(project, FILE, {}, closures)
    (project / "docs/guide.md").write_text("two")
    after = keys.build_manifest(project, FILE, {}, keys.Closures(project))
    assert before["tree"] != after["tree"]


# -- (6) what the plugin sees --------------------------------------------------------------------------
def test_a_file_that_appears_changes_a_test_that_asked_whether_it_exists(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(
        "from pathlib import Path\n\n\ndef test_a():\n    assert not Path(__file__).with_name('flag.txt').exists()\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]
    (project / "tests/flag.txt").write_text("x")
    assert hows(attempt(project, store)[0]) == ["ran"]


@pytest.mark.parametrize("body", [
    "os.mkdir(os.path.join(os.path.dirname(__file__), 'made'))",
    "open(os.path.join(os.path.dirname(__file__), 'made.txt').encode(), 'w').write('x')",
    "os.rename(__file__, __file__ + '.moved'); os.rename(__file__ + '.moved', __file__)",
])
def test_a_test_that_changes_the_checkout_is_not_stored(project, tmp_path, body):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(f"import os\n\n\ndef test_a():\n    {body}\n")
    report, _ = attempt(project, store)
    assert "pass" not in kinds(store) and "not stored" in report.outcomes[0].detail


def test_a_path_prefix_is_compared_by_parts():
    assert not plugin._write_ok("/devices/x") and plugin._write_ok("/dev/null")


def test_a_session_fixture_read_is_attributed_to_every_file_that_runs_with_it(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/data.txt").write_text("1")
    (project / "tests/conftest.py").write_text(
        "from pathlib import Path\n\nimport pytest\n\n\n@pytest.fixture(scope='session')\ndef data():\n"
        "    return Path(__file__).with_name('data.txt').read_text()\n")
    for name in ("a", "b"):
        (project / f"tests/test_{name}.py").write_text("def test_x(data):\n    assert data\n")
    attempt(project, store, "tests/test_a.py", "tests/test_b.py")
    (project / "tests/data.txt").write_text("2")
    assert hows(attempt(project, store, "tests/test_a.py", "tests/test_b.py")[0]) == ["ran", "ran"]


# -- (7) every worker must have reported -------------------------------------------------------------------
def test_a_missing_worker_record_stores_nothing(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    for name in ("a", "b"):
        (project / f"tests/test_{name}.py").write_text("def test_x():\n    pass\n")

    def lose_one(real):
        def launch(command):
            status = real(command)
            folder = Path(os.environ["DEV_TEST_REUSE_RECORD"])
            workers = [p for p in folder.glob("*.json") if "worker" in json.loads(p.read_text())["meta"]]
            workers[0].unlink()
            return status
        return launch

    attempt(project, store, "tests/test_a.py", "tests/test_b.py", launch=lose_one, extra=("-n", "2"))
    assert "pass" not in kinds(store)


# -- (8) the arguments that can change a result ---------------------------------------------------------
def test_plugin_names_roots_ignores_and_versions_are_in_the_key(project):
    base = keys.lookup(project, FILE, ["-q"]).key
    assert keys.lookup(project, FILE, ["-q", "-p", "xdist"]).key != base
    assert keys.lookup(project, FILE, ["-q", "-p", "testslots_pytest", "-p", "testreuse_plugin"]).key == base
    assert keys.lookup(project, FILE, ["-q", "--ignore", "tests/other"]).key != base
    assert keys.lookup(project, FILE, ["-q", "-c", "other.ini"]).key != base
    assert keys.lookup(project, FILE, ["-q", "--rootdir=elsewhere"]).key != base
    assert keys.lookup(project, FILE, ["-q"]).parts.get("pytest")


# -- (10) exit codes and admission ----------------------------------------------------------------------
def script():
    loader = importlib.machinery.SourceFileLoader("runner_under_review", str(SCRIPTS / "test"))
    spec = importlib.util.spec_from_loader("runner_under_review", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_a_file_skipped_at_module_level_is_passed_but_unstored_and_exits_zero(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text("import pytest\n\npytest.skip('not here', allow_module_level=True)\n")
    report, _ = attempt(project, store)
    assert report.status == 0 and report.outcomes[0].passed and "pass" not in kinds(store)


FAKE = f"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
sys.path.insert(0, {str(SCRIPTS.parent / "src")!r})
import testjobs
import testreuse_store as st
jobs = testjobs.Jobs(Path(os.environ["JOBS_HOME"]))
job = sys.argv[2]
me = {{"pid": os.getpid(), "started": st.process_start(os.getpid())}}
jobs.write(job, "status.json", {{"state": "running", "at": time.time(), **me}})
time.sleep(30)
"""


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_HOME", str(tmp_path / "store"))
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    (tmp_path / "tests").mkdir(exist_ok=True)
    return testjobs.Jobs(tmp_path / "store"), fake, tmp_path


def test_a_foreground_run_is_not_refused_by_a_live_whole_tier_job(table):
    jobs, fake, root = table
    first = jobs.submit(["full"], {"id": "alice"}, root, fake, {})
    second = jobs.submit(["fast"], {"id": "bob"}, root, fake, {"exclusive": False})
    assert second != first
    for job in (first, second):
        jobs.cancel(job, "alice" if job == first else "bob")


def test_the_whole_tier_check_is_atomic_and_counts_gate(table):
    jobs, fake, root = table
    wins, refusals = [], []

    def go():
        try:
            wins.append(jobs.submit(["gate"], {"id": "alice"}, root, fake, {}))
        except testjobs.Refused:
            refusals.append(1)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(refusals) == 3
    jobs.cancel(wins[0], "alice")


def test_a_refused_second_whole_tier_submit_has_its_own_exit_code(table, monkeypatch, capsys):
    jobs, fake, root = table
    jobs.submit(["full"], {"id": "alice"}, root, fake, {})
    module = script()
    monkeypatch.setenv("DEV_TEST_REUSE_DIR", str(root / "store"))
    monkeypatch.setattr(module, "jobs_store", lambda: jobs)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["scripts/test", "submit", "full"])
    monkeypatch.setattr(module.testboard, "acting", lambda *a: None)
    assert module.main() == 6
    assert "already runs a whole tier" in capsys.readouterr().err


# -- (11) smaller defects ------------------------------------------------------------------------------------
def test_a_claim_file_still_being_written_is_not_deleted(tmp_path):
    store = storage.Store(tmp_path / "store")
    (store.folder / "inflight").mkdir(parents=True)
    path = store.folder / "inflight" / "k.json"
    path.write_text("")
    assert store.claimed("k") is not None and path.exists()


def test_a_process_with_no_recorded_start_is_not_alive():
    assert not storage.alive(2 ** 22 + 11, 0.0) and not storage.alive(os.getpid(), 0.0)


def test_a_torn_last_chain_line_is_recovered_not_fatal(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    with store.chain.open("a") as handle:
        handle.write('{"seq": 1, "kind": "pa')
    assert reuse.chain_ok(store.folder) and hows(attempt(project, store)[0]) == ["reused"]
    (project / "src/ml_stack/helper.py").write_text("VALUE = 1  # edited\n")
    attempt(project, store)
    assert reuse.chain_ok(store.folder) and kinds(store)[-1] == "pass"


def test_an_unauthenticated_caller_cannot_cancel_an_unauthenticated_job(table):
    jobs, fake, root = table
    job = jobs.submit(["tests/x.py"], {"id": ""}, root, fake, {})
    with pytest.raises(testjobs.Refused):
        jobs.cancel(job, "")
    jobs.cancel(job, "", secret=jobs.read(job, "spec.json")["secret"])


def test_failing_class_tests_get_node_ids_with_the_class(table):
    jobs, fake, root = table
    job = jobs.submit(["tests/x.py"], {"id": "alice"}, root, fake, {})
    (root / "tests/test_x.py").write_text("")
    (jobs.path(job) / "junit-1.xml").write_text(
        '<testsuites><testsuite><testcase classname="tests.test_x.TestC" name="test_a"><failure/></testcase>'
        "</testsuite></testsuites>")
    assert jobs.result(job, root)["failing"] == ["tests/test_x.py::TestC::test_a"]
    jobs.cancel(job, "alice")


def test_a_job_that_raises_systemexit_still_records_its_end(tmp_path, monkeypatch):
    module = script()
    jobs = testjobs.Jobs(tmp_path / "store")
    jobs.write("j1", "spec.json", {"id": "j1", "argv": ["fast"], "owner": {"id": ""}})
    monkeypatch.setattr(module, "main", lambda argv: (_ for _ in ()).throw(SystemExit(3)))
    args = type("A", (), {"agent": "", "label": "", "task": ""})()
    module.run_job(jobs, "j1", args)
    assert jobs.read("j1")["state"] in ("failed", "done") and jobs.read("j1")["exit"] == 3


def test_a_board_notice_never_raises_out_of_a_run():
    class Broken:
        base = Path("/nonexistent")

        def __getattr__(self, name):
            raise RuntimeError(name)

    events = testboard.BoardEvents(testboard.Acting(Broken(), "token", {"id": "alice", "parent": ""}))
    assert events.claimed(FILE, "k" * 64) == 0
    events.canary_mismatch(FILE, "detail")
    events.job_done("j", {}, {"exit": 0}, 0)


def test_a_malformed_canary_rate_does_not_crash_and_zero_is_reported(project, tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_TEST_REUSE_CANARY", "often")
    store = storage.Store(tmp_path / "store")
    report, _ = attempt(project, store)
    assert report.status == 0
    monkeypatch.setenv("DEV_TEST_REUSE_CANARY", "0")
    report, _ = attempt(project, store)
    assert "canary off" in report.lines()[-1]


def test_unregistered_checkouts_do_not_share_a_scope(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    assert testruns.scope(one) != testruns.scope(two)
