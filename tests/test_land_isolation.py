"""One failing request never fails another: each way a request can go wrong ejects only that request.

Every case runs the real runner over a real temporary repository, a bare origin and the real board,
with the stub gate in ``land_support`` standing in for the project's tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from land_support import DEV, Project, git
from land_world import World, mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import land_board
import land_entries
import land_git as lg
import land_recover as recover
import land_run

pytest_plugins = ["node_kit"]


@pytest.fixture
def world(monkeypatch, tmp_path, workspace_node):
    return World(monkeypatch, tmp_path, workspace_node)


def plus(n: int, **extra: str) -> dict[str, str]:
    """A mapped module and its test, plus extra files."""
    return {**mod(n), **extra}


def on_origin(world: World, path: str) -> bool:
    return bool(git(world.origin, "ls-tree", "-r", "--name-only", DEV, "--", path))


def test_moved_tip_and_wrong_sha_are_refused_alone(world):
    stale, _ = world.ready("stale", plus(20))
    wrong = world.ask("bob", "wrongsha", world.branch("other", plus(21)))
    good, _ = world.ready("good", plus(22))
    Project.write(world.proj.base / "stale", "src/poolhouse/extra.py", "X = 1\n")
    Project.commit(world.proj.base / "stale", "feat: moved")
    world.runner.once()
    assert world.status(stale) == "refused" and "request it again" in world.detail(stale)
    assert world.status(good) == "landed" and on_origin(world, "src/poolhouse/m22.py")
    assert world.status(wrong) == "needs-review"
    assert not any("refused" in t for t in world.texts("bob"))


def test_conflict_ejects_only_the_conflicting_request_and_names_the_requester(world):
    first, _ = world.ready("x", {"src/poolhouse/mod.py": "VALUE = 2\n"})
    second, _ = world.ready("y", {"src/poolhouse/mod.py": "VALUE = 3\n"}, who="bob")
    other, _ = world.ready("z", plus(23), who="alice")
    world.runner.once()
    states = {world.status(first), world.status(second)}
    assert states == {"landed", "needs-human"} and world.status(other) == "landed"
    loser = first if world.status(first) == "needs-human" else second
    who = "alice" if loser == first else "bob"
    assert any("needs-human" in t and "merge conflict" in t for t in world.texts(who))


def test_red_test_ejects_the_culprit_with_its_test_name_and_the_rest_lands(world):
    one, _ = world.ready("g1", plus(24))
    bad, _ = world.ready("bad", mod(25, "# FAILME\n"), who="bob")
    two, _ = world.ready("g2", plus(26))
    result = world.runner.once()
    assert result["status"] == "landed", result
    assert world.status(bad) == "failed" and "tests/test_m25.py" in world.detail(bad)
    assert world.status(one) == world.status(two) == "landed"
    assert on_origin(world, "src/poolhouse/m24.py") and not on_origin(world, "src/poolhouse/m25.py")
    assert any("failed" in t and "tests/test_m25.py" in t for t in world.texts("bob"))
    assert not any("failed" in t for t in world.texts("alice"))


def test_red_gate_on_the_combined_tree_is_bisected_to_each_culprit(world):
    ok, _ = world.ready("fine", plus(27))
    bad1, _ = world.ready("red1", plus(28, BAD_GATE_1="x\n"), who="bob")
    bad2, _ = world.ready("red2", plus(29, BAD_GATE_2="x\n"), who="bob")
    result = world.runner.once()
    assert result["status"] == "landed", result
    assert world.status(ok) == "landed"
    assert world.status(bad1) == world.status(bad2) == "failed"
    assert "gate" in world.detail(bad1) and not on_origin(world, "BAD_GATE_1")


def test_a_failure_only_the_pair_causes_blames_the_later_merge(world):
    first, _ = world.ready("ca", plus(30, COMBO_A="x\n"))
    second, _ = world.ready("cb", plus(31, COMBO_B="x\n"), who="bob")
    other, _ = world.ready("cc", plus(32))
    world.runner.once()
    assert world.status(first) == world.status(other) == "landed"
    assert world.status(second) == "failed" and on_origin(world, "COMBO_A") and not on_origin(world, "COMBO_B")


def test_a_hanging_request_is_stopped_alone_and_the_rest_lands(world):
    one, _ = world.ready("h1", plus(33))
    hang, _ = world.ready("hang", plus(34, HANG_GATE="x\n"), who="bob")
    two, _ = world.ready("h2", plus(35))
    result = world.make_runner(stall_s=4.0).once()
    assert result["status"] == "split", result
    assert world.status(one) == world.status(two) == "landed"
    assert world.status(hang) == "needs-human" and "no progress" in world.detail(hang)
    assert on_origin(world, "src/poolhouse/m33.py") and not on_origin(world, "HANG_GATE")
    assert not list(world.proj.base.glob("*-land-*"))


def test_a_gate_past_its_limit_is_timed_out_and_a_request_has_a_shorter_limit(world):
    runner = world.make_runner(stall_s=99.0, request_s=1.0, batch_s=3.0)
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    one, two = {"id": "a"}, {"id": "b"}
    world.lead.claim("branch", DEV)
    assert runner.stream(sleeper, [one])[2] == "timeout"
    assert runner.stream(sleeper, [one, two])[2] == "timeout"
    assert runner.limit([one]) < runner.limit([one, two])


def test_a_merge_step_that_crashes_ejects_that_branch_and_merges_the_rest(world, monkeypatch):
    world.branch("boom", plus(36))
    world.branch("calm", plus(37))
    real = land_run.merge_one

    def merge_one(wt, branch):
        if branch == "boom":
            raise RuntimeError("git exploded")
        return real(wt, branch)

    monkeypatch.setattr(land_run, "merge_one", merge_one)
    batch = land_run.create(world.proj.root, DEV, ["boom", "calm"])
    assert land_run.merge_all(batch, "eject")
    assert [b for b, _ in batch.merged] == ["calm"]
    assert [(e["branch"], e["check"]) for e in batch.ejected] == [("boom", "merge")]
    assert "crashed" in batch.ejected[0]["evidence"]
    assert not lg.dirty(batch.wt)


def test_a_missing_worktree_does_not_stop_the_batch(world):
    gone, _ = world.ready("gone", plus(38))
    other, _ = world.ready("here", plus(39))
    shutil.rmtree(world.proj.base / "gone")
    result = world.runner.once()
    assert result["status"] == "landed", result
    assert world.status(gone) == world.status(other) == "landed"


def test_an_exception_for_one_request_settles_that_request_and_the_loop_continues(world, monkeypatch):
    kaboom, _ = world.ready("kaboom", plus(41))
    fine, _ = world.ready("fine2", plus(42), who="bob")
    real = land_entries.check

    def check(root, target, branch, tip):
        if branch == "kaboom":
            raise RuntimeError("disk on fire")
        return real(root, target, branch, tip)

    monkeypatch.setattr(land_entries, "check", check)
    result = world.runner.once()
    assert result["status"] == "landed", result
    assert world.status(kaboom) == "needs-human" and "disk on fire" in world.detail(kaboom)
    assert world.status(fine) == "landed"
    assert any("disk on fire" in t for t in world.texts("alice"))


class Crash(BaseException):
    """A runner process dying: nothing in the runner may catch it."""


def crash_at(monkeypatch, name: str, when=lambda *a: True):
    """Make ``Runner.<name>`` raise ``Crash`` when ``when(*args)`` holds; the function that repairs it."""
    real = getattr(land_board.Runner, name)

    def boom(self, *args, **kw):
        if when(*args):
            raise Crash(name)
        return real(self, *args, **kw)

    monkeypatch.setattr(land_board.Runner, name, boom)
    return lambda: monkeypatch.setattr(land_board.Runner, name, real)


def merges(world: World, since: str) -> int:
    return int(git(world.proj.root, "rev-list", "--merges", "--count", f"{since}..{DEV}"))


def test_a_crash_after_the_fast_forward_resumes_without_landing_again(world, monkeypatch):
    first, _ = world.ready("r1", plus(43))
    second, _ = world.ready("r2", plus(44), who="bob")
    before = world.local_head()
    repair = crash_at(monkeypatch, "landed")
    runner = world.runner
    with pytest.raises(Crash):
        runner.once()
    runner.lock.release()
    head = world.local_head()
    assert head != before and world.origin_head() == before
    assert world.status(first) == "running"
    repair()
    assert world.make_runner().once()["status"] == "idle"
    assert world.status(first) == world.status(second) == "landed"
    assert world.local_head() == head == world.origin_head()
    assert merges(world, before) == 2
    assert [c for c in world.proj.calls() if c.startswith("gate")] == ["gate"]


def test_a_crash_before_the_fast_forward_cleans_up_and_lands_once(world, monkeypatch):
    first, _ = world.ready("m1", plus(45))
    second, _ = world.ready("m2", plus(46), who="bob")
    before = world.local_head()
    repair = crash_at(monkeypatch, "land", lambda *a: a[0] == "finish")
    real_sweep = recover.sweep
    monkeypatch.setattr(recover, "sweep", lambda root: [])
    runner = world.runner
    with pytest.raises(Crash):
        runner.once()
    runner.lock.release()
    monkeypatch.setattr(recover, "sweep", real_sweep)
    assert git(world.proj.root, "branch", "--list", "land/*")
    repair()
    assert world.make_runner().once()["status"] == "landed"
    assert world.status(first) == world.status(second) == "landed"
    assert not git(world.proj.root, "branch", "--list", "land/*")
    assert not list(world.proj.base.glob("*-land-*"))
    assert merges(world, before) == 2 and world.origin_head() == world.local_head()


def test_a_dead_runners_claim_is_taken_over_and_a_live_one_is_not(world):
    rival = world.runner_session("rival-lander")
    dead = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    rival.claim("branch", DEV, pid=dead.pid)
    probe = world.make_runner()
    assert probe.once()["status"] == "runner-held", "a live holder is not displaced"
    probe.lock.release()
    dead.kill()
    dead.wait()
    assert world.runner.once()["status"] == "idle"
    world.runner.close()
    rival.claim("branch", DEV, pid=os.getpid())
    assert world.make_runner().once()["status"] == "runner-held"


def test_two_runners_cannot_run_at_once_even_as_the_same_identity(world):
    first, second = world.runner, world.make_runner()
    assert first.once()["status"] == "idle"
    held = second.once()
    assert held["status"] == "runner-held" and "another runner process" in held["owner"]
    first.close()
    assert second.once()["status"] == "idle"


def test_a_request_already_on_the_development_branch_is_not_landed_again(world):
    rid, _ = world.ready("dup", plus(47))
    git(world.proj.root, "merge", "-q", "--no-ff", "-m", "chore: merge dup", "dup")
    git(world.proj.root, "push", "-q", "origin", DEV)
    head = world.local_head()
    world.runner.once()
    assert world.status(rid) == "landed" and "already holds" in world.detail(rid)
    assert world.local_head() == head and not [c for c in world.proj.calls() if c.startswith("gate")]


def move_origin(world: World, files: dict[str, str]) -> str:
    """Another clone pushes a commit to origin; the new origin head."""
    other = world.proj.base / "clone"
    if not other.exists():
        git(world.proj.base, "clone", "-q", "-b", DEV, str(world.origin), str(other))
        git(other, "config", "user.name", "Other")
        git(other, "config", "user.email", "other@example.invalid")
        git(other, "config", "commit.gpgsign", "false")
    for rel, text in files.items():
        Project.write(other, rel, text)
    Project.commit(other, "feat: someone else landed")
    git(other, "push", "-q", "origin", f"HEAD:{DEV}")
    return git(other, "rev-parse", "HEAD")


def move_after_gate(world: World, files: dict[str, str]) -> land_board.Runner:
    """A runner whose origin moves right after its batch is verified, before the push."""
    runner, real = world.runner, world.runner.land

    def land(*args, **kw):
        out = real(*args, **kw)
        if args[0] == "run":
            move_origin(world, files)
        return out

    runner.land = land
    return runner


def test_a_rejected_push_merges_the_new_tip_regates_and_pushes_without_dropping_anyone(world):
    first, _ = world.ready("p1", plus(48))
    second, _ = world.ready("p2", plus(49), who="bob")
    result = move_after_gate(world, {"elsewhere.txt": "x\n"}).once()
    assert result["status"] == "landed", result
    assert world.status(first) == world.status(second) == "landed"
    assert world.origin_head() == world.local_head()
    assert on_origin(world, "elsewhere.txt") and on_origin(world, "src/poolhouse/m48.py")
    assert on_origin(world, "src/poolhouse/m49.py")


def test_a_rejected_push_that_conflicts_is_reported_not_dropped_and_not_repeated(world):
    rid, _ = world.ready("q1", {"src/poolhouse/mod.py": "VALUE = 2\n"})
    runner = move_after_gate(world, {"src/poolhouse/mod.py": "VALUE = 9\n"})
    result = runner.once()
    assert result["status"] == "landed-unpushed" and "does not merge cleanly" in result["push"], result
    assert world.status(rid) == "landed-unpushed"
    batch_head = world.local_head()
    assert world.origin_head() != batch_head and not lg.dirty(world.proj.root)
    sent = len(world.texts("alice"))
    assert any("push failed" in t for t in world.texts("alice"))
    assert runner.once()["status"] == "idle"
    assert len(world.texts("alice")) == sent and world.local_head() == batch_head


def test_a_rejected_push_whose_merged_tree_is_red_goes_back_to_the_batch_head(world):
    rid, _ = world.ready("s1", plus(50))
    runner = move_after_gate(world, {"BAD_GATE_late": "x\n"})
    result = runner.once()
    assert result["status"] == "landed-unpushed" and "re-gate" in result["push"], result
    assert world.status(rid) == "landed-unpushed" and world.origin_head() != world.local_head()
    assert not git(world.proj.root, "ls-files", "BAD_GATE_late")
