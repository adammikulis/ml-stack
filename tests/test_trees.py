"""The worktree registry, the orphan rule, close/sweep and the notices, on real git repositories."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack import trees, trees_notice

REPO = Path(__file__).resolve().parent.parent
HOUR = 3600.0
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def git(where: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(where), *args], env=ENV, text=True, capture_output=True, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def commit(where: Path, name: str) -> None:
    (where / name).write_text(name)
    git(where, "add", name)
    git(where, "commit", "-qm", "c", "--no-verify")


def make_repo(tmp_path: Path) -> Path:
    """A repository on branch dev with one commit."""
    root = tmp_path / "main"
    root.mkdir()
    git(root, "init", "-q", "-b", "dev")
    commit(root, "base")
    return root


@pytest.fixture
def main(tmp_path):
    return make_repo(tmp_path)


def tree(main: Path, name: str, owner: str = "", now: float = 0.0) -> Path:
    """A worktree beside the checkout, claimed by ``owner`` when given."""
    path = main.parent / name
    git(main, "worktree", "add", "-q", "-b", name, str(path))
    if owner:
        trees.claim(main, path, owner, now, {"purpose": "test"})
    return path


def status(main: Path, now: float) -> dict[str, str]:
    return {Path(r["path"]).name: r["status"] for r in trees.rows(main, now)}


def test_a_running_owners_clean_fresh_tree_is_never_swept(main):
    path = tree(main, "fresh", owner="worker-a")
    assert status(main, 10) == {"fresh": "active"}
    assert trees.sweep(main, 10) == []
    assert path.exists()


def test_a_finished_owner_with_commits_is_an_orphan_and_the_line_names_the_actions(main):
    path = tree(main, "work", owner="worker-a")
    commit(path, "f1")
    debt = trees.finish(main, "worker-a", 100)
    assert [Path(r["path"]).name for r in debt] == ["work"]
    assert status(main, 200) == {"work": "waiting"}                 # just finished: waiting to land, not an orphan
    assert "waiting to land" in trees.lines(main, 200)[0]
    row = trees.orphans(trees.rows(main, 100 + 3 * HOUR))[0]
    text = trees.line(row, 100 + 3 * HOUR)
    for word in ("ORPHAN", "work", "unlanded=1", "dirty=0", "worker-a", "--landed", "--bundle", "--abandon"):
        assert word in text
    assert trees.sweep(main, 200) == []
    assert path.exists()


def test_a_landed_tree_whose_owner_stopped_is_swept(main):
    path = tree(main, "done", owner="worker-a")
    commit(path, "f1")
    git(main, "merge", "-q", "--ff-only", "done")
    trees.finish(main, "worker-a", 100)
    assert status(main, 101) == {"done": "landed"}
    swept = trees.sweep(main, 101)
    assert [s["branch"] for s in swept] == ["done"]
    assert not path.exists()
    assert "done" not in git(main, "branch", "--format=%(refname:short)").split()


def test_a_dirty_tree_refuses_close_landed_and_abandon_needs_a_reason(main):
    path = tree(main, "dirty", owner="worker-a")
    (path / "scratch.txt").write_text("unsaved")
    trees.finish(main, "worker-a", 100)
    with pytest.raises(trees.Refused, match="dirty"):
        trees.close(main, path, ("landed", ""), 101)
    with pytest.raises(trees.Refused, match="reason"):
        trees.close(main, path, ("abandon", " "), 101)
    done = trees.close(main, path, ("abandon", "scratch only"), 101)
    assert not path.exists()
    assert done["reason"] == "scratch only"


def test_bundle_is_created_and_verified_before_the_tree_goes(main):
    path = tree(main, "keep", owner="worker-a")
    commit(path, "f1")
    head = git(path, "rev-parse", "HEAD")
    trees.finish(main, "worker-a", 100)
    done = trees.close(main, path, ("bundle", ""), 101)
    assert not path.exists()
    bundle = Path(done["bundle"])
    git(main, "bundle", "verify", str(bundle))
    assert head in git(main, "bundle", "list-heads", str(bundle))


def test_bundle_refuses_dirty_work_and_a_tree_with_nothing_to_bundle(main):
    dirty = tree(main, "dirty", owner="a")
    commit(dirty, "f1")
    (dirty / "x").write_text("x")
    clean = tree(main, "clean", owner="b")
    trees.finish(main, "a", 100)
    trees.finish(main, "b", 100)
    with pytest.raises(trees.Refused, match="dirty"):
        trees.close(main, dirty, ("bundle", ""), 101)
    with pytest.raises(trees.Refused, match="no unlanded"):
        trees.close(main, clean, ("bundle", ""), 101)
    assert dirty.exists()


def test_close_refuses_an_active_owners_tree_unless_the_owner_asks(main):
    path = tree(main, "live", owner="worker-a")
    with pytest.raises(trees.Refused, match="still active"):
        trees.close(main, path, ("landed", ""), 10, caller="the-lead")
    assert path.exists()
    trees.close(main, path, ("landed", ""), 10, caller="worker-a")
    assert not path.exists()


def test_close_refuses_the_primary_checkout(main):
    with pytest.raises(trees.Refused):
        trees.close(main, main, ("abandon", "why"), 1)


def test_scan_claims_trees_made_by_hand_as_unknown_and_they_age_into_orphans(main):
    path = main.parent / "byhand"
    git(main, "worktree", "add", "-q", "-b", "byhand", str(path))
    commit(path, "f1")
    assert trees.rows(main, 0)[0]["owner"] == trees.UNKNOWN
    assert status(main, 10) == {"byhand": "active"}
    assert status(main, 2 * HOUR) == {"byhand": "waiting"}
    assert status(main, 4 * HOUR) == {"byhand": "orphan"}
    claimed = trees.scan(main, 3, ("claimant", path))
    assert claimed["trees"][str(path.resolve())]["owner"] == "claimant"


def test_scan_drops_a_tree_that_is_gone(main):
    path = tree(main, "gone", owner="a")
    git(main, "worktree", "remove", "--force", str(path))
    assert trees.scan(main, 1)["trees"] == {}


def test_an_owner_whose_process_has_ended_is_stopped_and_one_whose_process_runs_is_not(main):
    gone = tree(main, "gone", owner="a")
    commit(gone, "f1")
    running = tree(main, "running", owner="b")
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    trees.claim(main, gone, "a", 5, {"pid": done.pid})
    trees.claim(main, running, "b", 5, {"pid": os.getpid()})
    assert status(main, 6) == {"gone": "waiting", "running": "active"}


def test_the_gate_counts_orphans_only_past_the_grace(main):
    path = tree(main, "late", owner="a")
    commit(path, "f1")
    trees.finish(main, "a", 1)
    pol = trees.current_policy(main)
    assert pol["grace_h"] == 2.0
    assert trees.past_grace(trees.rows(main, HOUR), pol, HOUR) == []
    assert trees.orphans(trees.rows(main, HOUR)) == [] and len(trees.waiting(trees.rows(main, HOUR))) == 1
    assert len(trees.past_grace(trees.rows(main, 2 * HOUR + 2), pol, 2 * HOUR + 2)) == 1


def test_the_gate_checker_reports_a_late_orphan_and_reads_the_grace_from_the_environment(main, monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    from gates import orphan_trees
    path = tree(main, "late", owner="a")
    commit(path, "f1")
    trees.finish(main, "a", 1.0)
    assert len(orphan_trees.find(main)) == 1          # stopped long ago by the real clock
    monkeypatch.setenv("ML_STACK_TREES_GRACE_H", "1e9")
    assert orphan_trees.find(main) == []
    assert orphan_trees.HARD and orphan_trees.NAME == "orphan-trees"


@pytest.mark.parametrize("commits,expect", [(9, False), (10, True)])
def test_ahead_notice_fires_at_the_threshold(main, commits, expect):
    path = tree(main, "big", owner="a", now=0)
    for i in range(commits):
        commit(path, f"f{i}")
    assert bool(trees_notice.notify(main, 10)) is expect


def test_a_fresh_tree_under_the_thresholds_never_notifies(main):
    tree(main, "fresh", owner="a")
    for now in (1, HOUR, 3 * HOUR):
        assert trees_notice.notify(main, now) == []


def test_behind_and_age_thresholds_and_policy_overrides(main, monkeypatch):
    start = time.time()          # commit ages are real timestamps, so the clock here is real too
    path = tree(main, "old", owner="a", now=start)
    commit(path, "mine")
    for i in range(21):
        commit(main, f"m{i}")
    assert trees_notice.notify(main, start + 60)[0].keys == ["behind"]
    assert trees_notice.notify(main, start + 5 * HOUR)[0].keys == ["age"]       # behind is unchanged: not told again
    monkeypatch.setenv("ML_STACK_TREES_MAX_BEHIND", "100")
    monkeypatch.setenv("ML_STACK_TREES_MAX_AGE_H", "100")
    later = trees_notice.notify(main, start + 8 * HOUR)       # still inside the owner's 12 hour sign of life
    assert later == []


def test_notices_are_rate_limited_to_one_per_interval(main):
    path = tree(main, "big", owner="a", now=0)
    for i in range(10):
        commit(path, f"f{i}")
    assert len(trees_notice.notify(main, 100)) == 1
    assert trees_notice.notify(main, 100 + 29 * 60) == []
    commit(path, "more")
    assert trees_notice.notify(main, 100 + 31 * 60)[0].keys == ["ahead"]       # changed, and the interval passed
    commit(path, "again")
    assert trees_notice.notify(main, 100 + 32 * 60) == []                      # changed but inside the interval


def test_routing_owner_first_then_the_coordinator_when_silent_or_stopped(main):
    trees.set_lead(main, "lead")
    path = tree(main, "big", owner="a", now=0)
    for i in range(10):
        commit(path, f"f{i}")
    assert trees_notice.notify(main, 100)[0].to == ["a"]
    assert trees_notice.notify(main, 100 + 31 * 60)[0].to == ["lead"]         # unchanged, owner silent: escalate once
    assert trees_notice.notify(main, 100 + 90 * 60) == []                      # and then nothing, unchanged
    trees.finish(main, "a", 4000)
    assert trees_notice.notify(main, 4000 + 31 * 60)[0].to == ["lead"]


def test_a_finished_tree_is_told_as_waiting_to_land_at_once_and_as_an_orphan_after_the_grace(main):
    trees.set_lead(main, "lead")
    path = tree(main, "stale", owner="a", now=0)
    commit(path, "f1")
    trees.finish(main, "a", 10)
    notes = trees_notice.notify(main, 11)
    assert notes[0].keys == ["waiting"] and notes[0].to == ["lead"]
    assert "waiting to land" in notes[0].text and "ORPHAN" not in notes[0].text
    late = trees_notice.notify(main, 10 + 3 * HOUR)
    assert late[0].keys == ["orphan"] and "ORPHAN" in late[0].text


def test_the_report_exits_one_on_an_orphan_and_close_runs_through_the_script(main):
    path = tree(main, "stale", owner="a")
    commit(path, "f1")
    trees.finish(main, "a", 1)
    script = str(REPO / "scripts" / "worktrees")
    env = {**ENV, "PYTHONPATH": str(REPO / "src")}

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, script, *args], cwd=main, env=env, text=True,
                              capture_output=True, check=False)

    report = run()
    assert report.returncode == 1 and "ORPHAN stale" in report.stdout
    assert run("close", str(path), "--landed").returncode == 2 and path.exists()
    assert run("close", str(path), "--abandon", "test").returncode == 0 and not path.exists()


@pytest.mark.parametrize("name", ["-evil", "--git-dir=x", "a b", "line\nfeed", "quote'\"", "$(touch x)"])
def test_a_tree_named_like_an_option_or_a_command_is_only_a_path_to_git(tmp_path, name):
    hostile = tmp_path / name
    hostile.mkdir()
    git(hostile, "init", "-q")
    assert Path(trees.git(hostile, "rev-parse", "--show-toplevel")).name == hostile.resolve().name
    with pytest.raises(RuntimeError):
        trees.git(tmp_path / ("missing " + name), "status")
    assert not (tmp_path / "x").exists()
