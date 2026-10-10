"""Landing authority is this device's own: a device that joined the pool posts a request, a review and a runner state
for a commit that exists in the local repository, and this device's runner lands nothing, until this device names it."""

from __future__ import annotations

import shutil
import time

import pytest
import testfarm_kit as kit
from land_world import World, mod
from node_kit import (  # noqa: F401  (a fixture the one below uses)
    STRIPPED,
    WorkspaceNode,
    node_binary,
)

from poolhouse import node_launch
from poolhouse.board import session as board_session
from poolhouse.board.client import Denied
from poolhouse.workspace import landing


@pytest.fixture
def pool(node_binary, monkeypatch, tmp_path):  # noqa: F811
    """Two paired nodes on loopback; this device (a) runs the landing world, the other (b) is a member of its pool."""
    kit.skip_unless_possible()
    a, b = kit.start(node_binary, "a"), kit.start(node_binary, "b")
    kit.pair(a, b)
    for name in STRIPPED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POOLHOUSE_HOME", str(a.root))
    monkeypatch.setenv("POOLHOUSE_BOARD", kit.BOARD)
    world = World(monkeypatch, tmp_path, WorkspaceNode(a.state, a.client, node_binary))
    try:
        yield world, a, b
    finally:
        for one in (a, b):
            node_launch.stop_node(one.state)
            shutil.rmtree(one.root, ignore_errors=True)


def other(b, name: str, model: str) -> board_session.Session:
    made = board_session.register(b.client, kit.BOARD, board_session.Native(model, "claude-code", name))
    return board_session.Session(b.client, kit.BOARD, made.token, made.name)


def wait_for(what: str, test) -> None:
    end = time.monotonic() + 30
    while time.monotonic() < end:
        if test():
            return
        time.sleep(0.2)
    pytest.fail(f"timed out waiting for {what}")


def post_forgery(world, a, b, sha: str) -> str:
    """On b: a request for ``sha``, an independent accept and the runner's `queued`, then let a hear of it."""
    asker, judge, runner = other(b, "b-asker", "claude-sonnet-5-5"), other(b, "b-judge", "claude-opus-5-5"), other(b, "b-runner", "")
    rid = landing.request(asker, {"branch": "foreign", "sha": sha, "selectors": ["tests/test_m1.py"]})["id"]
    landing.review(judge, rid, sha, "accept")
    runner.claim(*landing.runner_claim())
    landing.transition(runner, rid, "queued", "forged")
    runner.release(*landing.runner_claim())  # a foreign holder of the claim would stop this device's runner by itself
    b.call("sync_now", token=b.token)
    a.call("sync_now", token=a.token)
    wait_for("the foreign entries to reach a", lambda: landing.fold(world.lead).foreign.get("total", 0) >= 3)
    return rid


def test_a_pool_member_lands_nothing_here_until_named_and_not_again_once_removed(pool):
    world, a, b = pool
    sha = world.branch("foreign", mod(1))
    before = (world.origin_head(), world.local_head())
    post_forgery(world, a, b, sha)

    queue = landing.fold(world.lead)
    assert queue.requests == {} and queue.foreign["status"] == "foreign, ignored" and queue.trusted_devices == []
    assert world.runner.once()["status"] == "idle", "nothing in the queue for the runner"
    assert (world.origin_head(), world.local_head()) == before

    (device,) = kit.pool_devices(a.state)
    landing.trust(world.who["carol"], device["fingerprint"])
    counted = landing.fold(world.lead)
    assert [r["status"] for r in counted.requests.values()] == ["queued"], "the same sequence counts once the device is named"
    assert world.runner.once()["status"] != "landed", "the runner re-derives standing from this device's own registrations"
    assert (world.origin_head(), world.local_head()) == before

    landing.trust(world.who["carol"], device["fingerprint"], False)
    assert landing.fold(world.lead).requests == {} and landing.fold(world.lead).foreign["total"] >= 3


def test_naming_a_device_is_refused_to_a_subagent(pool):
    world, a, _ = pool
    (device,) = kit.pool_devices(a.state)
    kid = world.node.session(world.node.member("kid", parent=world.members["alice"]))
    with pytest.raises(Denied):
        landing.trust(kid, device["fingerprint"])
    assert landing.fold(world.lead).trusted_devices == []
