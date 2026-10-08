"""Per-file test result reuse: keys, the closure, entry integrity, single flight, canaries and reports,
driven through real pytest runs of tiny test files in a temporary project."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import testreuse_key as keys
import testreuse_run as run
import testreuse_store as storage

from ml_stack.activity import reuse

pytestmark = pytest.mark.slow

AGENT = {"id": "alice", "label": "a", "parent": "", "source": "test"}
TEST_A = "from ml_stack.helper import VALUE\n\n\ndef test_a():\n    assert VALUE == 1\n"
FILE = "tests/test_a.py"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv("DEV_TEST_REUSE_CANARY", raising=False)
    root = tmp_path / "p"
    (root / "src" / "ml_stack").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/ml_stack/__init__.py").write_text("")
    (root / "src/ml_stack/helper.py").write_text("VALUE = 1\n")
    (root / "src/ml_stack/other.py").write_text("OTHER = 1\n")
    (root / FILE).write_text(TEST_A)
    return root


class Recorder(run.Events):
    def __init__(self):
        super().__init__()
        self.agent = AGENT
        self.seen = []

    def waiting(self, file, owner):
        self.seen.append(("waiting", file))

    def wait_failed(self, file, owner):
        self.seen.append(("wait_failed", file))

    def canary_mismatch(self, file, detail):
        self.seen.append(("canary", file, detail))

    def finished(self, file, key, outcome, entry):
        self.seen.append(("finished", file, outcome))


def launcher(root: Path, runs: list):
    def launch(command: list[str]) -> int:
        runs.append(command)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(SCRIPTS)])}
        done = subprocess.run([*command, "-p", "testreuse_plugin", "-p", "no:cacheprovider"], cwd=root, env=env,
                              capture_output=True, text=True, check=False)
        return done.returncode
    return launch


def attempt(root: Path, store: storage.Store, *files: str, **options):
    """Run ``files`` (default tests/test_a.py) through a Session; returns (report, pytest launches).

    Options: ``draw`` (the canary draw, default no canary), ``events``, ``reuse_on`` and ``extra`` pytest flags.
    """
    named = list(files or (FILE,))
    command = [sys.executable, "-m", "pytest", "-q", *options.get("extra", ()), *named]
    session = run.Session(root, command, named, store, options.get("events") or Recorder())
    draw = options.get("draw", 1.0)
    session.draw = lambda: draw
    launches: list = []
    report = session.run(launcher(root, launches), "treehash0001", options.get("reuse_on", True))
    return report, launches


def hows(report) -> list[str]:
    return [o.how.split(" from ")[0] for o in report.outcomes]


def test_an_unchanged_passing_file_is_reused_and_reported_as_reused(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    first, launches = attempt(project, store)
    assert hows(first) == ["ran"] and first.status == 0 and len(launches) == 1
    second, launches = attempt(project, store)
    assert hows(second) == ["reused"] and second.status == 0 and launches == []
    line = second.lines()[0]
    assert "reused from " in line and "tree treehash00" in line and second.counts() == (0, 1)
    assert second.lines()[-1].startswith("test: 0 file(s) ran, 1 reused")


def test_a_changed_module_in_the_import_closure_invalidates_the_hit(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    (project / "src/ml_stack/helper.py").write_text("VALUE = 1  # edited\n")
    report, launches = attempt(project, store)
    assert hows(report) == ["ran"] and len(launches) == 1
    assert "changed: src/ml_stack/helper.py" in report.outcomes[0].detail


def test_a_change_outside_the_closure_keeps_the_hit(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    (project / "src/ml_stack/other.py").write_text("OTHER = 2\n")
    report, launches = attempt(project, store)
    assert hows(report) == ["reused"] and launches == []


def test_the_key_is_content_not_path(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    twin = tmp_path / "elsewhere" / "twin"
    shutil.copytree(project, twin)
    report, launches = attempt(twin, store)
    assert hows(report) == ["reused"] and launches == []


def test_changed_test_file_arguments_and_environment_change_the_key(project, monkeypatch):
    base = keys.lookup(project, FILE, ["-q", "-m", "not slow"])
    assert keys.lookup(project, FILE, ["-q", "-m", "slow"]).key != base.key
    assert keys.lookup(project, FILE, ["-q", "-m", "not slow", "-n", "4", FILE]).key == base.key
    monkeypatch.setenv("CI", "1")
    assert keys.lookup(project, FILE, ["-q", "-m", "not slow"]).key != base.key
    monkeypatch.delenv("CI")
    (project / FILE).write_text(TEST_A + "\n")
    assert keys.lookup(project, FILE, ["-q", "-m", "not slow"]).key != base.key


def test_a_failure_is_recorded_for_attribution_and_never_reused(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text("def test_a():\n    assert False\n")
    red, _ = attempt(project, store)
    assert red.status != 0 and not red.outcomes[0].passed
    assert [r["kind"] for r in reuse.rows(store.folder)] == ["fail"]
    again, launches = attempt(project, store)
    assert hows(again) == ["ran"] and len(launches) == 1
    (project / FILE).write_text("def test_a():\n    assert True\n")
    green, _ = attempt(project, store)
    assert green.status == 0 and [r["kind"] for r in reuse.rows(store.folder)] == ["fail", "fail", "pass"]


def test_marked_skipped_and_state_touching_files_are_not_stored(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/test_heavy.py").write_text("import pytest\n\n\n@pytest.mark.heavy\ndef test_h():\n    pass\n")
    (project / "tests/test_skip.py").write_text("import pytest\n\n\ndef test_s():\n    pytest.skip('no')\n")
    (project / "tests/test_write.py").write_text(
        f"def test_w():\n    open({str(project / 'outside.txt')!r}, 'w').write('x')\n")
    report, _ = attempt(project, store, "tests/test_heavy.py", "tests/test_skip.py", "tests/test_write.py")
    details = {o.file: o.detail for o in report.outcomes}
    assert "not reusable" in details["tests/test_heavy.py"]
    assert "a test was skipped" in details["tests/test_skip.py"]
    assert "wrote" in details["tests/test_write.py"]
    assert reuse.rows(store.folder) == []
    again, launches = attempt(project, store, "tests/test_heavy.py", "tests/test_skip.py", "tests/test_write.py")
    assert hows(again) == ["ran", "ran", "ran"] and len(launches) == 1


def test_a_data_file_the_test_read_is_part_of_the_hit(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / "tests/data.txt").write_text("one")
    (project / FILE).write_text(
        "from pathlib import Path\n\n\ndef test_a():\n    assert Path(__file__).with_name('data.txt').read_text()\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]
    (project / "tests/data.txt").write_text("two")
    report, _launches = attempt(project, store)
    assert hows(report) == ["ran"] and "changed: tests/data.txt" in report.outcomes[0].detail


def test_a_directory_listing_the_test_read_is_part_of_the_hit(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    (project / FILE).write_text(
        "import os\nfrom pathlib import Path\n\n\ndef test_a():\n"
        "    assert os.listdir(Path(__file__).parent)\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]
    (project / "tests/test_new.py").write_text("def test_n():\n    pass\n")
    assert hows(attempt(project, store)[0]) == ["ran"]


def entry_path(store: storage.Store) -> Path:
    return next((store.folder / "entries").glob("*.json"))


def test_an_edited_entry_is_a_miss(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    path = entry_path(store)
    entry = json.loads(path.read_text())
    entry["counts"]["tests"] = 99
    path.write_text(json.dumps(entry))
    report, _launches = attempt(project, store)
    assert hows(report) == ["ran"] and "failed verification" in report.outcomes[0].detail


def test_an_entry_with_a_recomputed_hash_but_no_chain_row_is_a_miss(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    path = entry_path(store)
    entry = json.loads(path.read_text())
    entry["junit_sha256"] = "0" * 64
    entry["entry_sha256"] = reuse.entry_hash(entry)
    forged = path.with_name(entry["entry_sha256"][:20] + ".json")
    forged.write_text(json.dumps(entry))
    (store.folder / "latest" / entry["lookup"]).write_text(entry["entry_sha256"][:20])
    assert store.hit(entry["lookup"], project, keys.Closures(project))[0] is None


def test_a_broken_chain_makes_every_entry_a_miss(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    attempt(project, store, extra=("-k", "a"))
    rows = reuse.rows(store.folder)
    rows[0]["at"] = 1.0
    store.chain.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert not reuse.chain_ok(store.folder)
    report, _ = attempt(project, store)
    assert hows(report) == ["ran"]


def test_an_entry_missing_a_field_is_a_miss(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    path = entry_path(store)
    entry = json.loads(path.read_text())
    del entry["agent"]
    path.write_text(json.dumps(entry))
    assert hows(attempt(project, store)[0]) == ["ran"]


def test_an_entry_records_the_runner_the_agent_the_command_and_the_junit_hash(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    entry = json.loads(entry_path(store).read_text())
    assert entry["agent"] == AGENT and entry["runner"]["pid"] == os.getpid()
    assert entry["runner"]["started"] > 0 and len(entry["junit_sha256"]) == 64
    assert entry["command"][:3] == [sys.executable, "-m", "pytest"] and entry["tree"] == "treehash0001"
    assert entry["counts"] == {"tests": 1, "failed": 0, "skipped": 0} and entry["outcome"] == "pass"


def test_a_sampled_hit_is_re_executed_and_agreement_is_reported(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    attempt(project, store)
    report, launches = attempt(project, store, draw=0.0)
    assert hows(report) == ["ran  canary agrees"] and len(launches) == 1
    assert "canary of" in report.outcomes[0].detail and report.status == 0


def test_a_cached_pass_that_fails_fresh_is_reported_disabled_and_recorded(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    gate = tmp_path / "outside-the-checkout"
    gate.write_text("up")
    (project / FILE).write_text(
        f"from pathlib import Path\n\n\ndef test_a():\n    assert Path({str(gate)!r}).read_text() == 'up'\n")
    attempt(project, store)
    assert hows(attempt(project, store)[0]) == ["reused"]
    gate.write_text("down")
    events = Recorder()
    report, launches = attempt(project, store, draw=0.0, events=events)
    assert report.status != 0 and "CANARY MISMATCH" in report.outcomes[0].how
    assert [e[0] for e in events.seen] == ["canary"] and len(launches) == 1
    assert [r["kind"] for r in reuse.rows(store.folder)][-1] == "incident"
    gate.write_text("up")
    after, launches = attempt(project, store)
    assert hows(after) == ["ran"] and "disabled" in after.outcomes[0].detail and len(launches) == 1


def test_a_key_in_flight_is_waited_for_and_its_result_reused(project, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "POLL_S", 0.05)
    store = storage.Store(tmp_path / "store")
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        key = keys.lookup(project, FILE, [sys.executable, "-m", "pytest", "-q", FILE]).key
        (store.folder / "inflight").mkdir(parents=True)
        (store.folder / "inflight" / f"{key}.json").write_text(json.dumps({
            "pid": owner.pid, "started": storage.process_start(owner.pid), "agent": {"id": "bob"}}))
        events = Recorder()
        result = {}
        waiter = threading.Thread(target=lambda: result.update(
            report=attempt(project, store, events=events)[0]))
        waiter.start()
        time.sleep(0.4)
        assert waiter.is_alive() and ("waiting", FILE) in events.seen
        attempt(project, store, reuse_on=False)
        owner.kill()
        owner.wait()
        waiter.join(30)
        assert not waiter.is_alive()
    finally:
        owner.kill()
    outcome = result["report"].outcomes[0]
    assert outcome.how.startswith("reused from") and "while this request waited" in outcome.detail


def test_a_waiter_runs_the_file_itself_when_the_owner_dies_without_a_result(project, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "POLL_S", 0.05)
    store = storage.Store(tmp_path / "store")
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    key = keys.lookup(project, FILE, [sys.executable, "-m", "pytest", "-q", FILE]).key
    (store.folder / "inflight").mkdir(parents=True)
    (store.folder / "inflight" / f"{key}.json").write_text(json.dumps({
        "pid": owner.pid, "started": storage.process_start(owner.pid), "agent": {"id": "bob"}}))
    events = Recorder()
    result = {}
    waiter = threading.Thread(target=lambda: result.update(report=attempt(project, store, events=events)))
    waiter.start()
    time.sleep(0.4)
    owner.kill()
    owner.wait()
    waiter.join(60)
    report, launches = result["report"]
    assert hows(report) == ["ran"] and len(launches) == 1 and report.status == 0
    assert ("wait_failed", FILE) in events.seen


def test_a_claim_from_a_dead_process_is_recovered(tmp_path):
    store = storage.Store(tmp_path / "store")
    (store.folder / "inflight").mkdir(parents=True)
    (store.folder / "inflight" / "k.json").write_text(json.dumps({"pid": 2 ** 22 + 7, "started": 1.0}))
    assert store.claimed("k") is None and store.claim("k", {"agent": AGENT}) is None
    assert store.claim("k", {"agent": AGENT})["pid"] == os.getpid()
    store.release("k")
    assert store.claimed("k") is None


def test_a_run_with_no_named_files_refreshes_the_store_without_reusing(project, tmp_path):
    store = storage.Store(tmp_path / "store")
    command = [sys.executable, "-m", "pytest", "-q"]
    session = run.Session(project, command, None, store, Recorder())
    launches: list = []
    report = session.run(launcher(project, launches), "treehash0001", False)
    assert report.status == 0 and [o.file for o in report.outcomes] == [FILE]
    assert [r["kind"] for r in reuse.rows(store.folder)] == ["pass"]
    explicit, launches = attempt(project, store)
    assert hows(explicit) == ["reused"] and launches == []


def test_selectors_name_files_only(project):
    command = [sys.executable, "-m", "pytest", "-q", FILE]
    assert run.selectors(command, project) == [FILE]
    assert run.selectors([*command, f"{FILE}::test_a"], project) is None
    assert run.selectors([sys.executable, "-m", "pytest", "tests"], project) is None
    assert run.selectors([sys.executable, "-m", "pytest", "-q"], project) is None
