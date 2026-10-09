"""scripts/land against real temporary git repositories with a fast fake gate."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.land_support import DEV, ROOT, Project, git


@pytest.fixture
def proj(tmp_path):
    return Project(tmp_path)


def mod(n: int, body: str = "") -> dict[str, str]:
    return {f"src/poolhouse/m{n}.py": f"VALUE = {n}\n{body}", f"tests/test_m{n}.py": f"import poolhouse.m{n}\n"}


def test_plan_covers_stacked_and_rebased_branches_by_patch_id(proj):
    a = proj.branch("a", mod(1))
    proj.branch("b", mod(2), start="a")
    git(proj.root, "checkout", "-q", "-b", "newer")
    Project.write(proj.root, "docs/other.md", "x\n")
    proj.commit(proj.root, "chore: moves target")
    git(proj.root, "checkout", "-q", DEV)
    git(proj.root, "merge", "-q", "--ff-only", "newer")
    proj.branch("r", {}, start=DEV)
    git(proj.base / "r", "cherry-pick", git(a, "rev-parse", "a"))
    code, out, summary = proj.land("plan", "a", "b", "r")
    assert code == 0
    assert [e["branch"] for e in summary["order"]] == ["b"]
    assert sorted(summary["covered"]) == ["a", "r"]
    assert summary["order"][0]["commits"] == 2
    assert "covered: r is contained in b" in out


def test_plan_predicts_conflicts_and_lists_dirty_worktrees(proj):
    proj.branch("x", {"src/poolhouse/mod.py": "VALUE = 2\n"})
    proj.branch("y", {"src/poolhouse/mod.py": "VALUE = 3\n"})
    dirty = proj.branch("z", mod(5))
    (dirty / "scratch.txt").write_text("wip")
    _, _, summary = proj.land("plan", "x", "y", "z")
    flagged = {e["branch"]: e for e in summary["order"]}
    assert flagged["y"]["conflicts_with_plan"] == ["src/poolhouse/mod.py"] or \
        flagged["x"]["conflicts_with_plan"] == ["src/poolhouse/mod.py"]
    assert summary["dirty"] == ["z"]
    assert (dirty / "scratch.txt").read_text() == "wip"


def test_plan_cover_uses_branches_with_worktrees_and_refuses_main(proj):
    proj.branch("a", mod(1))
    _, _, summary = proj.land("plan", "--cover")
    assert [e["branch"] for e in summary["order"]] == ["a"]
    code, _, summary = proj.land("plan", "main")
    assert code == 4 and summary["status"] == "error"


def test_conflicting_branch_is_ejected_and_the_rest_verified(proj):
    proj.branch("x", {"src/poolhouse/mod.py": "VALUE = 2\n"})
    proj.branch("y", {"src/poolhouse/mod.py": "VALUE = 3\n"})
    proj.branch("z", mod(5))
    code, out, summary = proj.land("run", "x", "y", "z")
    assert code == 0 and summary["status"] == "verified"
    assert len(summary["ejected"]) == 1 and summary["ejected"][0]["check"] == "merge"
    assert sorted(summary["merged"] + [summary["ejected"][0]["branch"]]) == ["x", "y", "z"]
    assert "EJECTED" in out and "owner=" in out


def test_conflicts_stop_halts_the_run(proj):
    proj.branch("x", {"src/poolhouse/mod.py": "VALUE = 2\n"})
    proj.branch("y", {"src/poolhouse/mod.py": "VALUE = 3\n"})
    code, _, summary = proj.land("run", "--conflicts=stop", "x", "y")
    assert code == 2 and summary["status"] == "stopped"
    assert proj.calls() == []


def test_docs_only_diff_runs_budgets_and_clean_diff_without_tests(proj):
    proj.branch("d", {"docs/a.md": "changed\n", "HANDOFF.md": "- item\n"})
    code, _, summary = proj.land("run", "d")
    assert code == 0 and summary["kind"] == "docs"
    assert [c["check"] for c in summary["checks"]] == ["budgets", "clean-diff"]
    assert proj.calls() == []


def test_code_diff_runs_gate_and_affected_selectors_once(proj):
    proj.branch("a", mod(1))
    proj.branch("b", mod(2))
    code, _, summary = proj.land("run", "a", "b")
    assert code == 0 and summary["kind"] == "code" and summary["full"] == ""
    calls = proj.calls()
    assert calls.count("gate") == 1
    suites = [c for c in calls if c.startswith("all")]
    assert len(suites) == 1 and "tests/test_m1.py" in suites[0] and "tests/test_m2.py" in suites[0]


def test_infrastructure_diff_starts_one_background_full_run(proj):
    proj.branch("i", {"pyproject.toml": "[project]\nname='x'\n"})
    code, _, summary = proj.land("run", "i")
    assert code == 0 and summary["full"]["status"] == "started"
    deadline = time.time() + 20
    def started():
        return [c for c in proj.calls() if c.split()[0] == "full"]

    while not started() and time.time() < deadline:
        time.sleep(0.1)
    assert started() == ["full --background"]


def test_second_full_run_is_refused_while_one_is_in_flight(proj):
    proj.branch("i", {"pyproject.toml": "[project]\nname='x'\n"})
    proj.branch("j", {"pyproject.toml": "[project]\nname='y'\n", "src/poolhouse/m9.py": "V = 1\n"})
    lock = Path(git(proj.root, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "land"
    lock.mkdir(exist_ok=True)
    (lock / "full.lock").write_text(f"{os.getpid()}\n")
    _, _, summary = proj.land("run", "i")
    assert summary["full"]["status"] == "refused"


def test_identical_tree_reuses_recorded_passes(proj):
    proj.branch("a", mod(1))
    proj.land("run", "a")
    first = len(proj.calls())
    _, out, summary = proj.land("run", "a")
    assert len(proj.calls()) == first
    assert [c["status"] for c in summary["checks"]] == ["reused", "reused"]
    assert "reused from" in out


def test_bisect_ejects_the_branch_that_breaks_the_gate(proj):
    proj.branch("a", mod(1))
    proj.branch("bad", {"BAD_GATE": "1\n", **mod(2)})
    proj.branch("c", mod(3))
    proj.branch("d", mod(4))
    code, out, summary = proj.land("run", "a", "bad", "c", "d")
    assert code == 0 and summary["status"] == "verified"
    assert [e["branch"] for e in summary["ejected"]] == ["bad"]
    assert summary["ejected"][0]["check"] == "gate"
    assert sorted(summary["merged"]) == ["a", "c", "d"]
    assert "EJECTED bad check=gate" in out
    assert len([c for c in proj.calls() if c == "gate"]) <= 6


def test_failing_test_file_is_attributed_by_selector(proj):
    proj.branch("a", mod(1))
    proj.branch("bad", mod(2, "# FAILME\n") | {"tests/test_m2.py": "import poolhouse.m2  # FAILME\n"})
    _, _, summary = proj.land("run", "a", "bad")
    assert [e["branch"] for e in summary["ejected"]] == ["bad"]
    assert "tests/test_m2.py" in summary["ejected"][0]["evidence"]
    assert summary["merged"] == ["a"]


def test_failure_that_also_fails_on_the_target_is_baseline(proj):
    Project.write(proj.root, "BAD_GATE", "1\n")
    proj.commit(proj.root, "chore: known failure")
    proj.branch("a", mod(1))
    code, _, summary = proj.land("run", "a")
    assert summary["baseline"] == ["gate"] and summary["ejected"] == []
    assert code == 0 and summary["checks"][0]["status"] == "baseline"


def test_finish_refuses_a_dirty_primary_and_prints_the_lead_command(proj):
    proj.branch("a", mod(1))
    proj.land("run", "a")
    (proj.root / "stray.txt").write_text("x")
    before = git(proj.root, "rev-parse", DEV)
    code, out, summary = proj.land("finish", "--apply")
    assert code == 3 and summary["status"] == "blocked"
    assert "git -C" in out and "merge --ff-only land/" in out
    assert git(proj.root, "rev-parse", DEV) == before


def test_finish_is_a_dry_run_unless_applied_then_fast_forwards_and_cleans_up(proj):
    a = proj.branch("a", mod(1))
    kept = proj.branch("k", mod(2))
    (kept / "wip.txt").write_text("wip")
    proj.land("run", "a", "k")
    before = git(proj.root, "rev-parse", DEV)
    code, _, summary = proj.land("finish")
    assert code == 0 and summary["status"] == "dry-run"
    assert git(proj.root, "rev-parse", DEV) == before and a.exists()
    code, _, summary = proj.land("finish", "--apply")
    assert code == 0 and summary["status"] == "landed"
    assert git(proj.root, "rev-parse", DEV) != before
    assert not a.exists() and kept.exists()
    assert "a" in summary["removed"]
    assert [r["branch"] for r in summary["kept"]] == ["k"] and "dirty" in summary["kept"][0]["reason"]
    assert "a" not in git(proj.root, "branch", "--list", "a")
    assert not [t for t in git(proj.root, "worktree", "list").splitlines() if "-land-" in t]


def test_finish_keeps_a_branch_that_still_has_unique_patches(proj):
    a = proj.branch("a", mod(1))
    proj.land("run", "a")
    Project.write(a, "src/poolhouse/extra.py", "E = 1\n")
    proj.commit(a, "feat: after the batch")
    _, _, summary = proj.land("finish", "--apply")
    assert [r["branch"] for r in summary["kept"]] == ["a"]
    assert a.exists()


def test_land_never_pushes_forces_or_names_main():
    text = "".join((ROOT / "scripts" / name).read_text() for name in
                   ("land", "land_git.py", "land_plan.py", "land_run.py", "land_finish.py", "land_check.py"))
    for forbidden in ('"push"', "--force", "-f\"", '"-D"'):
        assert forbidden not in text
