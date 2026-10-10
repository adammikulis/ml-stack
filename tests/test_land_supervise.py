"""The runner's supervisor: restarts a crashing runner, runs one at a time, and is started by one command."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
from land_support import DEV
from land_world import World, mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import land_supervise

from poolhouse.lock import only_one, pid_alive
from poolhouse.workspace import landing, limits

CRASH = [sys.executable, "-c", "import sys; sys.exit(3)"]
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


pytest_plugins = ["node_kit"]


@pytest.fixture
def world(monkeypatch, tmp_path, workspace_node):
    return World(monkeypatch, tmp_path, workspace_node)


def lines(world: World) -> str:
    return "\n".join(landing.status_lines(world.lead, limits.root()))


def wait_for(check, seconds: float = 30.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        found = check()
        if found:
            return found
        time.sleep(0.1)
    raise AssertionError("timed out waiting")


def run_loop(world: World, argv: list[str], delays=(0.05, 0.2, 600.0)):
    """The supervisor loop in a thread; its stop switch and the thread."""
    stopping = {"now": False}
    _, log, _ = land_supervise.files(limits.root())
    thread = threading.Thread(target=land_supervise.loop, args=(limits.root(), argv, log, stopping, delays), daemon=True)
    thread.start()
    return stopping, thread


def test_a_crashing_runner_is_started_again_with_a_growing_delay_and_counted(world):
    stopping, thread = run_loop(world, CRASH)
    now = wait_for(lambda: (s := land_supervise.status(limits.root())).get("restarts", 0) >= 3 and s)
    stopping["now"] = True
    thread.join(10)
    assert not thread.is_alive()
    assert now["last_exit"] == 3 and "restarting" in now["state"]
    assert land_supervise.status(limits.root())["state"] == "stopped"


def test_stopping_the_supervisor_stops_the_runner_it_started(world):
    stopping, thread = run_loop(world, SLEEP)
    child = wait_for(lambda: land_supervise.status(limits.root()).get("child"))
    stopping["now"] = True
    thread.join(40)
    assert not thread.is_alive() and not pid_alive(child)


def test_a_second_supervisor_is_refused(world):
    with only_one(land_supervise.files(limits.root())[2], wait=False):
        code, out, _ = world.proj.land("supervise", "--interval", "1")
    assert code == 1 and "already runs" in out


def test_up_starts_a_runner_that_lands_a_request_and_down_stops_it_cleanly(world):
    assert "NOT RUNNING" in lines(world)
    rid, _ = world.ready("viaup", {**mod(60)})
    code, out, done = world.proj.land("up", "--interval", "1")
    try:
        assert code == 0 and done["started"] and done["alive"], out
        again = world.proj.land("up", "--interval", "1")[2]
        assert again["started"] is False and again["supervisor"] == done["supervisor"]
        wait_for(lambda: world.status(rid) == "landed", 90)
        assert "Runner supervisor: up" in lines(world)
        assert world.origin_head() == world.local_head()
        assert world.proj.land("status")[2]["alive"] is True
    finally:
        stopped = world.proj.land("down")
    assert stopped[0] == 0 and stopped[2]["stopped"], stopped
    assert land_supervise.status(limits.root())["state"] == "stopped"
    assert not [c for c in world.lead.claims("branch") if c.key == DEV]
    assert "NOT RUNNING" in lines(world)
