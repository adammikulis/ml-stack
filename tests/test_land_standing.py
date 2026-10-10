"""A worker that asks to land and then ends still lands; one ended for cause does not; a sibling reviewer counts.

A real node, a real runner over real git repositories and a bare origin. The lead is `bob`; the worker and the
reviewer are subagents the lead spawned.
"""

from __future__ import annotations

import json

import pytest
from land_support import DEV
from land_world import World, mod

from poolhouse.board.client import Denied
from poolhouse.workspace import landing

pytest_plugins = ["node_kit"]


@pytest.fixture
def world(monkeypatch, tmp_path, workspace_node):
    return World(monkeypatch, tmp_path, workspace_node)


def spawn(world, name: str, model: str = "claude-sonnet-5-5", parent: str = "bob"):
    """A subagent of ``parent`` and its session."""
    return world.node.session(world.node.member(name, model=model, parent=world.members[parent]))


def requested(world, worker, branch: str, n: int) -> tuple[str, str]:
    sha = world.branch(branch, mod(n))
    return landing.request(worker, {"branch": branch, "sha": sha, "selectors": ["tests/test_mod.py"]})["id"], sha


def test_a_worker_that_asked_and_then_ended_still_lands_and_so_does_its_reviewer_who_ended(world):
    worker, reviewer = spawn(world, "worker"), spawn(world, "reviewer", "claude-opus-5-5")
    rid, sha = requested(world, worker, "done", 20)
    landing.review(reviewer, rid, sha, "accept")
    assert world.status(rid) == "queued"
    worker.retire()
    reviewer.retire()
    before = world.origin_head()
    assert world.runner.once()["status"] == "landed"
    assert world.status(rid) == "landed" and world.origin_head() == world.local_head() != before


def test_a_worker_ended_for_cause_after_it_asked_is_refused_with_the_reason(world):
    for cause in ("revoked", "forged"):
        worker, reviewer = spawn(world, f"w-{cause}"), spawn(world, f"r-{cause}", "claude-opus-5-5")
        rid, sha = requested(world, worker, f"cause-{cause}", 21 if cause == "revoked" else 22)
        landing.review(reviewer, rid, sha, "accept")
        world.who["bob"].retire(worker.name, cause)
        before = world.origin_head()
        assert world.runner.once()["status"] == "idle"
        assert world.status(rid) == "refused" and f"was {cause} after it asked" in world.detail(rid)
        assert world.origin_head() == before


def test_a_reviewer_ended_for_cause_no_longer_counts_but_one_that_ended_normally_does(world):
    worker, bad = spawn(world, "worker"), spawn(world, "bad", "claude-opus-5-5")
    rid, sha = requested(world, worker, "gone", 23)
    landing.review(bad, rid, sha, "accept")
    world.who["bob"].retire(bad.name, "revoked")
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "needs-review" and "no independent accept that still stands" in world.detail(rid)
    good = spawn(world, "good", "claude-opus-5-5")
    landing.review(good, rid, sha, "accept")
    good.retire()
    assert world.runner.once()["status"] == "landed"


def test_a_sibling_reviews_but_the_lead_and_the_requesters_own_descendants_cannot(world):
    worker, sibling = spawn(world, "worker"), spawn(world, "sibling", "claude-opus-5-5")
    helper = world.node.session(world.node.member("helper", model="claude-opus-5-5", parent=world.node.members["worker"]))
    rid, sha = requested(world, worker, "kin", 24)
    for kin in (world.who["bob"], worker, helper):
        with pytest.raises(Denied, match="independent reviewer"):
            landing.review(kin, rid, sha, "accept")
    assert world.status(rid) == "needs-review"
    landing.review(sibling, rid, sha, "accept")
    assert world.status(rid) == "queued"
    assert world.runner.once()["status"] == "landed"


def test_the_land_review_command_accepts_a_sibling_and_refuses_the_lead(world):
    env = {"POOLHOUSE_DEV_BRANCH": DEV}
    worker = world.node.member("cliworker", parent=world.members["bob"])
    sibling = world.node.member("clisibling", model="claude-opus-5-5", parent=world.members["bob"])
    sha = world.branch("clireview", mod(25))
    asked = world.node.cli("land-request", "clireview", sha, "--test", "tests/test_m25.py", "--json", who=worker, env=env,
                           cwd=world.proj.root)
    assert asked.returncode == 0, asked.stderr
    rid = json.loads(asked.stdout)["id"]
    lead = world.node.cli("land-review", rid, sha, "--json", who=world.members["bob"], env=env, cwd=world.proj.root)
    assert lead.returncode != 0 and "independent reviewer" in lead.stdout + lead.stderr
    done = world.node.cli("land-review", rid, sha, "--json", who=sibling, env=env, cwd=world.proj.root)
    assert done.returncode == 0, done.stderr
    assert world.status(rid) == "queued"
