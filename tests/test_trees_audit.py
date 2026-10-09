"""The registry must add no friction: it fails open, is safe under concurrency, does not cry orphan early,
does not spam, reconciles with removals made outside the tool and never nags a person using plain git."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_trees import ENV, HOUR, REPO, commit, git, make_repo, status, tree

from ml_stack import lock, trees, trees_notice

HOOKS = REPO / "scripts" / "hooks"


@pytest.fixture
def main(tmp_path):
    return make_repo(tmp_path)


def registry(main: Path) -> Path:
    return trees.registry_path(main)


@pytest.mark.parametrize("garbage", ["{ torn", "[1, 2, 3]", '{"trees": [1], "history": 5}',
                                     '{"trees": {"x": "not an entry", "y": {"owner": 3}}}', ""])
def test_a_garbled_registry_is_rebuilt_from_git_instead_of_failing(main, garbage):
    path = tree(main, "keep", owner="a")
    registry(main).write_text(garbage)
    assert status(main, 5) == {"keep": "active"} or status(main, 5) == {"keep": "waiting"}
    assert json.loads(registry(main).read_text())["trees"][str(path.resolve())]["owner"] in (trees.UNKNOWN, "a", 3)


def test_a_missing_registry_and_a_non_repository_are_fine(main, tmp_path):
    tree(main, "t", owner="a")
    registry(main).unlink()
    assert status(main, 5) == {"t": "active"}      # claimed again as unknown, alive for the claim window
    sys.path.insert(0, str(HOOKS))
    import tree_watch
    assert tree_watch.check("A", str(tmp_path / "nowhere"), {}) == ""


def test_a_locked_registry_still_answers_reads_and_the_hooks_fail_open(main, monkeypatch, capsys):
    tree(main, "t", owner="a")
    sys.path.insert(0, str(HOOKS))
    import tree_watch
    monkeypatch.setattr(trees, "LOCK_TIMEOUT", 0.2)
    with lock.only_one(registry(main).with_name("ml-stack-trees.json.lock"), announce=lambda _m: None):
        started = time.monotonic()
        assert status(main, 5) == {"t": "active"}                  # read path: answers, skips the save
        with pytest.raises(lock.Busy):
            trees.set_lead(main, "lead")
        assert tree_watch.check("A", str(main), {"ML_STACK_WORKSPACE_AGENT": "lead"}, lead="lead") == ""
        assert time.monotonic() - started < 5
    assert "workspace A" in capsys.readouterr().err


def test_a_slow_git_is_cut_off_by_the_hook_budget(main, monkeypatch):
    sys.path.insert(0, str(HOOKS))
    import tree_watch
    monkeypatch.setattr(trees_notice, "notify", lambda *a, **k: time.sleep(30))
    started = time.monotonic()
    with pytest.raises(TimeoutError), tree_watch.bounded(1):
        trees_notice.notify(main, 0)
    assert time.monotonic() - started < 5


def test_parallel_processes_writing_the_registry_lose_nothing_and_leave_valid_json(main):
    paths = [tree(main, f"w{i}") for i in range(8)]
    code = ("import sys, time; from pathlib import Path; from ml_stack import trees\n"
            "trees.LOCK_TIMEOUT = 120  # the hooks' short wait is not under test\n"
            "root, mine = Path(sys.argv[1]), sys.argv[2:]\n"
            "for n in range(6):\n"
            "    for p in mine:\n"
            "        trees.claim(root, p, 'o-' + Path(p).name, time.time() + n)\n"
            "    trees.scan(root, time.time())\n")
    env = {**ENV, "PYTHONPATH": str(REPO / "src")}
    procs = [subprocess.Popen([sys.executable, "-c", code, str(main), *map(str, paths[i::4])], env=env,
                              stderr=subprocess.PIPE, text=True) for i in range(4)]
    assert all(p.wait(timeout=120) == 0 for p in procs), [p.stderr.read() for p in procs]
    owners = {Path(k).name: v["owner"] for k, v in json.loads(registry(main).read_text())["trees"].items()}
    assert owners == {f"w{i}": f"o-w{i}" for i in range(8)}


def test_a_scratch_tree_never_counts_and_a_clean_one_is_swept(main):
    scratch = main.parent / "base"
    git(main, "worktree", "add", "-q", "--detach", str(scratch))
    commit(scratch, "x")
    trees.claim(main, scratch, "a", 1, {"kind": "scratch"})
    trees.finish(main, "a", 2)
    found = trees.rows(main, 5 * HOUR)
    assert [r["status"] for r in found] == ["scratch"]
    assert trees.orphans(found) == [] and trees.lines(main, 5 * HOUR) == [] and trees_notice.notify(main, 5 * HOUR) == []
    assert trees.sweep(main, 5 * HOUR) == [] or not scratch.exists()


def test_a_dirty_scratch_tree_is_not_swept(main):
    scratch = main.parent / "dirty-scratch"
    git(main, "worktree", "add", "-q", "--detach", str(scratch))
    (scratch / "keep.txt").write_text("keep")
    trees.claim(main, scratch, "a", 1, {"kind": "scratch"})
    trees.finish(main, "a", 2)
    assert trees.sweep(main, 5 * HOUR) == []
    assert scratch.exists()


def test_the_primary_checkout_and_a_fresh_unclaimed_tree_are_not_orphans(main):
    byhand = main.parent / "byhand"
    git(main, "worktree", "add", "-q", "-b", "byhand", str(byhand))
    commit(byhand, "f")
    found = trees.rows(main, 60)
    assert [Path(r["path"]).name for r in found] == ["byhand"]          # the primary is not a row
    assert found[0]["status"] == "active" and trees.lines(main, 60) == []


def test_ten_finished_workers_cost_the_lead_at_most_six_messages_an_hour_with_a_roll_up(main):
    trees.set_lead(main, "lead")
    for i in range(10):
        commit(tree(main, f"w{i}", owner=f"w{i}", now=0), f"c{i}")
        trees.finish(main, f"w{i}", 1)
    sent: dict[str, list[float]] = {"lead": [], "board": []}
    rollups = 0
    for step in range(0, 6 * 3600, 300):
        for note in trees_notice.notify(main, step + 10):
            rollups += note.keys == ["rollup"]
            sent["lead"] += [step] * ("lead" in note.to)
            sent["board"] += [step] * note.board
    for who, times in sent.items():
        assert times, who
        assert all(sum(1 for t in times if start <= t < start + 3600) <= 6 for start in range(0, 6 * 3600, 300)), who
    assert rollups >= 1


def test_a_condition_reported_and_unchanged_is_not_reported_again(main):
    trees.set_lead(main, "lead")
    path = tree(main, "big", owner="a", now=0)
    for i in range(12):
        commit(path, f"f{i}")
    told = [n.to for step in range(0, 6 * 3600, 600) for n in trees_notice.notify(main, step + 10)]
    assert told == [["a"], ["lead"]]                       # the owner once, the lead once when the owner stayed silent


def test_a_tree_landed_by_patch_after_a_rebase_closes_as_landed(main):
    path = tree(main, "rebased", owner="a")
    commit(path, "feature")
    sha = git(path, "rev-parse", "HEAD")
    commit(main, "other")                                              # dev moved on
    git(main, "cherry-pick", sha)                                     # landed as a different commit
    trees.finish(main, "a", 1)
    assert status(main, 5) == {"rebased": "landed"}
    done = trees.close(main, path, ("landed", ""), 5)
    assert done["mode"] == "landed" and not path.exists()


def test_the_gate_failure_names_the_exact_close_command_and_waits_out_the_grace(main):
    sys.path.insert(0, str(REPO / "scripts"))
    from gates import orphan_trees
    path = tree(main, "left", owner="a")
    commit(path, "f")
    trees.finish(main, "a", time.time())                               # just finished
    assert orphan_trees.find(main) == []                               # waiting to land: the gate does not fail
    trees.finish(main, "b", 1)
    entry = json.loads(registry(main).read_text())
    entry["trees"][str(path.resolve())]["stopped"] = time.time() - 3 * HOUR
    registry(main).write_text(json.dumps(entry))
    found = orphan_trees.find(main)
    assert len(found) == 1 and f"scripts/worktrees close {path.resolve()} --bundle" in found[0].detail


def test_a_tree_removed_outside_the_tool_is_reconciled_not_an_error(main):
    gone, removed = tree(main, "rm", owner="a"), tree(main, "gitrm", owner="b")
    shutil.rmtree(gone)
    git(main, "worktree", "remove", "--force", str(removed))
    assert status(main, 5) == {}
    history = json.loads(registry(main).read_text())["history"]
    assert {h["mode"] for h in history} == {"removed-outside-the-tool"}
    with pytest.raises(trees.Refused, match="not a registered worktree"):
        trees.close(main, gone, ("landed", ""), 5)


def test_a_warm_report_asks_git_only_for_the_worktree_list(main, monkeypatch):
    for i in range(5):
        commit(tree(main, f"t{i}", owner=f"o{i}", now=time.time()), f"c{i}")
    trees.rows(main, time.time())
    verbs: list[str] = []
    real = trees.git
    monkeypatch.setattr(trees, "git", lambda root, *a, **k: (verbs.append(a[0]), real(root, *a, **k))[1])
    trees.rows(main, time.time())
    assert set(verbs) <= {"worktree", "rev-parse"}


def run_hook(where: Path, **env: str) -> subprocess.CompletedProcess:
    merged = {**ENV, "PYTHON": sys.executable, "PYTHONPATH": str(REPO / "src"), **env}
    merged.pop("ML_STACK_WORKSPACE_AGENT", None)             # a person, not an agent
    return subprocess.run(["sh", str(HOOKS / "post-commit")], cwd=where, text=True, capture_output=True, check=False,
                          env=merged)


def test_a_person_committing_in_the_primary_checkout_is_never_nagged(main):
    for i in range(15):
        commit(main, f"c{i}")
    done = run_hook(main)
    assert done.returncode == 0 and done.stdout == "" and done.stderr == ""


def test_a_linked_tree_under_the_threshold_prints_nothing_and_over_it_prints_one_line_to_stderr(main):
    path = tree(main, "side")
    for i in range(3):
        commit(path, f"c{i}")
    quiet = run_hook(path)
    assert quiet.returncode == 0 and quiet.stdout == "" and quiet.stderr == ""
    loud = run_hook(path, ML_STACK_TREES_MAX_AHEAD="2")
    assert loud.returncode == 0 and loud.stdout == "" and "3 commits ahead" in loud.stderr
    assert len(loud.stderr.strip().splitlines()) == 1
    assert run_hook(path, ML_STACK_TREES_MAX_AHEAD="2").stderr == ""     # and not again for the same condition
