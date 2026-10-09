"""`scripts/gaterun.py` reuses a passed gate step on an unchanged tree and runs independent steps side by side.

Every case runs real subprocesses against a real git repository. The reuse cases hold the fast path (a
cache that answers) to the full path (every step run) on the same seeded violation, and the two cases
at the end break the key and the lanes on purpose to show the agreement checks can fail."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gaterun  # noqa: E402

SCAN = ("import sys, pathlib\n"
        "root, counter = sys.argv[1], sys.argv[2]\n"
        "open(counter, 'a').write('x')\n"
        "bad = [p for p in pathlib.Path(root, 'src').rglob('*.py') if 'VIOLATION' in p.read_text()]\n"
        "sys.exit(1 if bad else 0)\n")
MEET = ("import sys, time, pathlib\n"
        "mine, other = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
        "mine.write_text('here')\n"
        "end = time.monotonic() + float(sys.argv[3])\n"
        "while time.monotonic() < end:\n"
        "    if other.exists():\n"
        "        sys.exit(0)\n"
        "    time.sleep(0.02)\n"
        "sys.exit(1)\n")


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", *args],
                   cwd=root, check=True, capture_output=True)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "tree"
    (root / "src").mkdir(parents=True)
    (root / "src" / "mod.py").write_text("VALUE = 1\n")
    git(root, "init", "-q", "-b", "0.3dev")
    git(root, "add", "src/mod.py")
    git(root, "commit", "-q", "-m", "base")
    return root


class Scan:
    """A step that scans the tree for a seeded violation and counts how often it really ran."""

    def __init__(self, root: Path, home: Path) -> None:
        self.root, self.counter = root, home / "ran"
        self.counter.write_text("")
        self.step = gaterun.Step("scan", self.run, ["scan"])

    def run(self) -> int:
        return subprocess.run([sys.executable, "-c", SCAN, str(self.root), str(self.counter)],
                              check=False).returncode

    def ran(self) -> int:
        return len(self.counter.read_text())

    def gate(self, folder: Path, reuse: bool) -> gaterun.Gate:
        return gaterun.Gate(self.root, "0.3dev", folder, reuse)

    def verdict(self, folder: Path, reuse: bool) -> gaterun.Outcome:
        return self.gate(folder, reuse).run([[self.step]])[0]

    def seed(self) -> None:
        (self.root / "src" / "mod.py").write_text("VALUE = 1  # VIOLATION\n")

    def fix(self) -> None:
        (self.root / "src" / "mod.py").write_text("VALUE = 1\n")


@pytest.fixture
def scan(tree, tmp_path):
    return Scan(tree, tmp_path), tmp_path / "memory"


def test_a_pass_on_an_unchanged_tree_is_reused_without_running_the_step(scan):
    step, folder = scan
    first, second = step.verdict(folder, True), step.verdict(folder, True)
    assert (first.status, first.reused) == (0, "")
    assert (second.status, bool(second.reused)) == (0, True)
    assert step.ran() == 1


def test_a_seeded_violation_fails_on_the_fast_path_exactly_as_on_the_full_path(scan):
    step, folder = scan
    assert step.verdict(folder, True).status == 0
    step.seed()
    fast, full = step.verdict(folder, True), step.verdict(folder, False)
    assert fast.status == full.status == 1
    assert fast.reused == ""
    assert step.ran() == 3


def test_a_failure_is_never_stored_so_it_is_found_again(scan):
    step, folder = scan
    step.seed()
    assert [step.verdict(folder, True).status for _ in range(3)] == [1, 1, 1]
    assert step.ran() == 3


def test_removing_the_violation_returns_to_the_remembered_pass_and_the_full_path_agrees(scan):
    step, folder = scan
    assert step.verdict(folder, True).status == 0
    step.seed()
    assert step.verdict(folder, True).status == 1
    step.fix()
    fast, full = step.verdict(folder, True), step.verdict(folder, False)
    assert fast.status == full.status == 0
    assert fast.reused
    assert step.ran() == 3


def test_an_untracked_file_is_part_of_the_key(scan):
    step, folder = scan
    held = step.gate(folder, True)
    before = held.key(step.step)
    (step.root / "src" / "new.py").write_text("X = 1\n")
    assert held.key(step.step) != before


def test_the_steps_identity_and_the_base_tip_are_part_of_the_key(scan):
    step, folder = scan
    held = step.gate(folder, True)
    other = gaterun.Step("scan", step.run, ["scan", "--other"])
    assert held.key(step.step) != held.key(other)
    git(step.root, "checkout", "-q", "-b", "work")
    git(step.root, "commit", "-q", "--allow-empty", "-m", "move")
    assert held.key(step.step) == gaterun.Gate(step.root, "0.3dev", folder, True).key(step.step)
    assert held.key(step.step) != gaterun.Gate(step.root, "HEAD", folder, True).key(step.step)


def test_a_store_that_cannot_be_written_only_costs_the_reuse(scan, tmp_path):
    step, _ = scan
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    assert [step.verdict(blocked / "inside", True).status for _ in range(2)] == [0, 0]
    assert step.ran() == 2


def test_a_key_that_ignored_the_tree_would_pass_a_seeded_violation_the_full_path_fails(scan, monkeypatch):
    step, folder = scan
    monkeypatch.setattr(gaterun, "tree_hash", lambda root: "always-the-same")
    assert step.verdict(folder, True).status == 0
    step.seed()
    assert (step.verdict(folder, True).status, step.verdict(folder, False).status) == (0, 1)


def meet(tmp_path: Path, mine: str, other: str, wait: float = 20.0) -> gaterun.Step:
    return gaterun.Step(mine, lambda: subprocess.run(
        [sys.executable, "-c", MEET, str(tmp_path / mine), str(tmp_path / other), str(wait)], check=False).returncode)


def test_independent_lanes_run_at_the_same_time_and_report_in_lane_order(tree, tmp_path):
    held = gaterun.Gate(tree, "0.3dev", None, False)
    outcomes = held.run([[meet(tmp_path, "a", "b")], [meet(tmp_path, "b", "a")]])
    assert [(o.name, o.status) for o in outcomes] == [("a", 0), ("b", 0)]


def test_a_failing_side_lane_fails_the_gate_as_it_does_run_in_one_lane(tree, tmp_path):
    steps = [gaterun.Step("ok", lambda: 0), gaterun.Step("bad", lambda: 3), gaterun.Step("after", lambda: 0)]
    held = gaterun.Gate(tree, "0.3dev", None, False)
    side = held.run([[steps[0]], [steps[1], steps[2]]])
    one = held.run([steps])
    assert sorted((o.name, o.status) for o in side) == sorted((o.name, o.status) for o in one)


def test_the_same_two_steps_in_one_lane_cannot_meet_so_the_concurrency_check_can_fail(tree, tmp_path):
    held = gaterun.Gate(tree, "0.3dev", None, False)
    outcomes = held.run([[meet(tmp_path, "a", "b", 1.0), meet(tmp_path, "b", "a", 1.0)]])
    assert [o.status for o in outcomes] == [1, 0]
