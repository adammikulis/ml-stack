"""The board-fed landing runner over real git repositories, a bare origin and a real node."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from land_support import DEV, Project, git
from land_world import SLOW_TEST, World, mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import land_board

from poolhouse.board import session
from poolhouse.board.client import Denied, Invalid
from poolhouse.workspace import landing, limits

pytest_plugins = ["node_kit"]


@pytest.fixture
def world(monkeypatch, tmp_path, workspace_node):
    return World(monkeypatch, tmp_path, workspace_node)


def test_two_requests_land_in_queue_order_as_one_batch_and_push_only_origin_development(world):
    first, _ = world.ready("a", mod(1))
    second, _ = world.ready("b", mod(2), who="bob")
    before = world.origin_head()
    result = world.runner.once()
    assert result["status"] == "landed" and sorted(result["merged"]) == ["a", "b"]
    assert world.status(first) == world.status(second) == "landed"
    assert world.origin_head() == world.local_head() != before
    assert git(world.origin, "branch", "--list").split() == [DEV]
    assert [c for c in world.proj.calls() if c.startswith("gate")] == ["gate"]
    assert not (world.proj.base / "a").exists() and not (world.proj.base / "b").exists()
    assert any("landed" in t for t in world.texts("alice"))


def test_conflicting_pair_reports_needs_human_and_lands_the_other(world):
    one, _ = world.ready("x", {"src/poolhouse/mod.py": "VALUE = 2\n"})
    two, _ = world.ready("y", {"src/poolhouse/mod.py": "VALUE = 3\n"}, who="bob")
    result = world.runner.once()
    assert {world.status(one), world.status(two)} == {"landed", "needs-human"}, result
    stuck = next(r for r in world.requests().values() if r["status"] == "needs-human")
    assert "merge conflict" in stuck["detail"]
    assert world.origin_head() == world.local_head()
    who = "alice" if stuck["id"] == one else "bob"
    assert any("needs-human" in t for t in world.texts(who))


def test_red_gate_does_not_push_and_names_the_failing_test(world):
    rid, _ = world.ready("bad", mod(3, "# FAILME\n"))
    before_origin, before_local = world.origin_head(), world.local_head()
    result = world.runner.once()
    assert result["status"] != "landed"
    assert world.status(rid) == "failed"
    assert "tests/test_m3.py" in world.detail(rid) and f"clean {DEV}" in world.detail(rid)
    assert world.origin_head() == before_origin and world.local_head() == before_local


def test_moved_branch_tip_is_refused_not_landed(world):
    rid, _ = world.ready("moving", mod(4))
    Project.write(world.proj.base / "moving", "src/poolhouse/extra.py", "X = 1\n")
    Project.commit(world.proj.base / "moving", "feat: more")
    before = world.origin_head()
    world.runner.once()
    assert world.status(rid) == "refused"
    assert "request it again" in world.detail(rid)
    assert world.origin_head() == before and world.local_head() == before


def test_unreviewed_request_is_marked_needs_review_and_never_lands(world):
    sha = world.branch("raw", mod(5))
    rid = world.ask("alice", "raw", sha)
    before = world.origin_head()
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "needs-review" and "review" in world.detail(rid)
    assert world.origin_head() == before
    with pytest.raises(Denied):
        landing.review(world.who["alice"], rid, sha, "accept")
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "needs-review"


def test_review_must_name_the_exact_sha_and_a_rejection_blocks(world):
    sha = world.branch("rev", mod(6))
    rid = world.ask("alice", "rev", sha)
    with pytest.raises(Invalid, match="exact commit"):
        landing.review(world.who["carol"], rid, "0" * 40, "accept")
    landing.review(world.who["carol"], rid, sha, "accept")
    landing.review(world.who["bob"], rid, sha, "reject")
    world.runner.once()
    assert world.status(rid) == "needs-review"


def test_only_landing_level_models_may_request_or_review_and_main_is_never_named(world):
    sha = world.branch("low", mod(7))
    with pytest.raises(Denied, match="lowest model tier"):
        world.ask("haiku", "low", sha)
    rid = world.ask("alice", "low", sha)
    with pytest.raises(Denied, match="lowest model tier"):
        landing.review(world.who["haiku"], rid, sha, "accept")
    for bad in ("main", "master"):
        with pytest.raises(Invalid):
            landing.request(world.who["alice"], {"branch": bad, "sha": sha, "selectors": ["t"]})
    with pytest.raises(Invalid):
        landing.request(world.who["alice"], {"branch": "low", "sha": sha, "selectors": ["t"], "target": "main"})
    with pytest.raises(Invalid):
        landing.request(world.who["alice"], {"branch": "low", "sha": "abc", "selectors": ["t"]})


def test_a_reviewer_who_is_gone_before_landing_leaves_the_request_waiting(world):
    kid = world.node.member("kid", model="claude-opus-5-5", parent=world.members["bob"])
    sha = world.branch("gone", mod(8))
    rid = world.ask("alice", "gone", sha)
    landing.review(world.node.session(kid), rid, sha, "accept")
    assert world.status(rid) == "queued"
    world.who["bob"].retire(kid.name)
    before = world.origin_head()
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "needs-review" and "no live independent accept" in world.detail(rid)
    assert world.origin_head() == before


def test_pause_and_resume_gate_the_runner_and_a_helper_may_not(world):
    rid, _ = world.ready("p", mod(8))
    helper = world.node.session(world.node.member("helper", parent=world.members["bob"]))
    with pytest.raises(Denied):
        landing.brake(helper, True)
    landing.brake(world.who["carol"], True, "checking something")
    assert world.runner.once()["status"] == "paused"
    assert world.status(rid) == "queued"
    assert "PAUSED by" in "\n".join(landing.status_lines(world.who["bob"], limits.root()))
    landing.brake(world.who["bob"], False)
    assert world.runner.once()["status"] == "landed"
    assert world.status(rid) == "landed"


def test_cancel_by_the_requester_or_a_session_that_steers_but_not_by_a_helper_of_another(world):
    rid, _ = world.ready("c", mod(9))
    helper = world.node.session(world.node.member("helper", parent=world.members["bob"]))
    with pytest.raises(Denied):
        landing.cancel(helper, rid)
    landing.cancel(world.who["alice"], rid)
    before = world.origin_head()
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "cancelled" and world.origin_head() == before


def test_new_request_for_a_branch_supersedes_the_old_one(world):
    old, _ = world.ready("s", mod(10))
    Project.write(world.proj.base / "s", "src/poolhouse/more.py", "Y = 1\n")
    new_sha = Project.commit(world.proj.base / "s", "feat: more")
    new = world.ask("alice", "s", new_sha)
    assert world.status(old) == "superseded"
    assert world.status(new) == "needs-review"


def test_a_request_for_another_target_is_left_to_the_runner_of_that_target(world):
    sha = world.branch("elsewhere", mod(14))
    rid = landing.request(world.who["alice"], {"branch": "elsewhere", "sha": sha, "selectors": ["t"], "target": "other"})["id"]
    landing.review(world.who["carol"], rid, sha, "accept")
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "queued"


def test_runner_never_pushes_main_or_a_missing_remote(world):
    with pytest.raises(ValueError, match="never touches main"):
        world.make_runner(target="main").push()
    git(world.proj.root, "checkout", "-q", "-b", "side")
    assert f"not {DEV}" in world.runner.push()
    git(world.proj.root, "checkout", "-q", DEV)
    code, out, _ = world.proj.land("serve", "--once", "--target", "main")
    assert code == 4 and "never touches main" in out


def test_second_runner_is_refused_while_the_claim_is_held(world):
    world.runner.once()
    rival = land_board.Runner(world.runner_session("other-lander"), world.proj.root, env=world.proj.env)
    assert rival.once()["status"] == "runner-held", "the process lock is held"
    world.runner.lock.release()
    held = rival.once()
    assert held["status"] == "runner-held" and held["owner"] == world.lead.name, "then the branch claim is"


def test_stuck_gate_is_aborted_reported_and_not_pushed(world):
    Project.write(world.proj.root, "scripts/test", SLOW_TEST)
    Project.commit(world.proj.root, "chore: slow gate")
    git(world.proj.root, "push", "-q", "origin", DEV)
    rid, _ = world.ready("slow", mod(11))
    before = world.origin_head()
    result = world.make_runner(stall_s=1.0).once()
    assert result["status"] == "stuck"
    assert world.status(rid) == "needs-human" and "no progress" in world.detail(rid)
    assert world.origin_head() == before
    assert any("stuck" in t for t in world.announced("blocked"))


def test_cli_request_and_queue(world):
    sha = world.branch("cli", mod(12))
    env = {"POOLHOUSE_DEV_BRANCH": DEV}
    done = world.node.cli("land-request", "cli", sha, "--test", "tests/test_m12.py", "--replaces", "nothing", "--json",
                          who=world.members["alice"], env=env, cwd=world.proj.root)
    assert done.returncode == 0, done.stderr
    shown = world.node.cli("land-queue", "--json", who=world.members["bob"], env=env, cwd=world.proj.root)
    assert "cli@" in shown.stdout and "needs-review" in shown.stdout
    refused = world.node.cli("land-request", "cli", sha, "--test", "t", "--json", who=world.members["haiku"], env=env,
                             cwd=world.proj.root)
    assert refused.returncode == 3 and "lowest model tier" in json.loads(refused.stdout)["error"]
    other = world.node.cli("land-review", json.loads(done.stdout)["id"], sha, "--json", who=world.members["carol"], env=env,
                           cwd=world.proj.root)
    assert other.returncode == 0, other.stderr
    assert world.status(json.loads(done.stdout)["id"]) == "queued"


def test_unset_development_branch_is_the_branch_the_primary_checkout_is_on(world, monkeypatch):
    monkeypatch.delenv("POOLHOUSE_DEV_BRANCH")
    git(world.proj.root, "checkout", "-q", "-b", "9.9dev")
    runner = land_board.Runner(world.lead, world.proj.root, env=world.proj.env)
    assert runner.target == "9.9dev"
    assert landing.runner_claim(runner.target) == ("branch", "9.9dev")
    sha = world.branch("onnine", mod(40))
    monkeypatch.chdir(world.proj.root)
    rid = landing.request(world.who["alice"], {"branch": "onnine", "sha": sha, "selectors": ["t"]})["id"]
    assert world.requests()[rid]["target"] == "9.9dev"


def test_the_runner_registers_itself_and_keeps_its_token_in_the_client_credentials(world):
    made = land_board.runner_session(world.proj.root, DEV)
    again = land_board.runner_session(world.proj.root, DEV)
    assert made.name == again.name and made.name not in {m.name for m in world.members.values()}
    assert (world.node.state / "client" / world.node.board / f"{made.name}.token").exists()
    assert session.find("land-runner", f"{world.proj.root}:{DEV}", client=world.node.client) == made.name
    assert made.whoami().harness == "land-runner"
