"""The test runner against a real temporary workspace: agent identity on leases, board threads and
inbox notices, subscriptions, project scoping and runner entries as task evidence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from taskboard_kit import board  # noqa: F401  (fixture)
from workspace_kit import Kit, clean_env

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import testreuse_store as storage  # noqa: E402

from ml_stack.activity import reuse  # noqa: E402
from ml_stack.workspace import onboard, slots, testruns, tokens  # noqa: E402
from ml_stack.workspace.taskboard import TaskBoard  # noqa: E402

PROJECT = {"key": "git@example.org:me/widgets.git", "name": "Widgets"}
FILE = "tests/test_a.py"
KEY = "ab12cd34ef56" + "0" * 52


@pytest.fixture
def team(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.limits(sends_per_window=1000)
    tokens.prepare(kit.ws.base)
    kit.seats = {}
    for name in ("alice", "bob"):
        code = kit.ws.invites.create(name, 600.0, PROJECT)
        joined = onboard.join(kit.ws, code, name)
        kit.seats[name] = testruns.Acting(kit.ws, tokens.load(kit.ws.base, joined),
                                          {"id": joined, "label": "t", "parent": "", "source": "test"})
    other = kit.ws.invites.create("zed", 600.0, {"key": "git@example.org:other/gadgets.git", "name": "Gadgets"})
    name = onboard.join(kit.ws, other, "zed")
    kit.seats["zed"] = testruns.Acting(kit.ws, tokens.load(kit.ws.base, name),
                                       {"id": name, "label": "", "parent": "", "source": "test"})
    return kit


def inbox(kit, who, ack=True):
    return kit.ws.inbox(kit.seats[who].token, ack=ack, limit=50)


def test_a_subscriber_to_a_runs_thread_is_told_exactly_once_when_it_lands(team):
    owner, watcher = testruns.BoardEvents(team.seats["alice"]), team.seats["bob"]
    thread = owner.claimed(FILE, KEY)
    assert thread > 0
    assert testruns.follow_thread(watcher, thread)["type"] == "thread"
    owner.finished(FILE, KEY, "pass", "entry0123456789abcdef")
    [message] = inbox(team, "bob")
    assert "test-result" in message["text"] and "outcome=pass" in message["text"]
    assert "entry=entry0123456789abcdef" in message["text"] and message["from"] == "alice/test-runner"
    assert inbox(team, "bob") == []


def test_waiters_are_told_when_the_run_they_waited_on_failed(team):
    owner, watcher = testruns.BoardEvents(team.seats["alice"]), testruns.BoardEvents(team.seats["bob"])
    thread = owner.claimed(FILE, KEY)
    watcher.waiting(FILE, {"thread": thread, "agent": {"id": "alice"}})
    owner.finished(FILE, KEY, "fail", "entryfail")
    [message] = inbox(team, "bob")
    assert "outcome=fail" in message["text"] and "run it yourself" in message["text"]


def test_a_job_ending_reaches_the_submitter_a_thread_follower_and_a_task_watcher_once_each(team):
    lead = team.agent("lead", "lead")
    tasks = TaskBoard(team.ws)
    task = tasks.create(lead, {"title": "Inspect", "description": "Look.", "acceptance": ["ok"],
                               "source_key": "repo:demo/sim:issue:1"})
    tasks.subscribe(team.seats["bob"].token, task["id"])
    carol = onboard.join(team.ws, team.ws.invites.create("carol", 600.0, PROJECT), "carol")
    seat = testruns.Acting(team.ws, tokens.load(team.ws.base, carol), {"id": carol, "label": "", "parent": ""})
    events = testruns.BoardEvents(team.seats["alice"], task["id"])
    thread = events.job_started("j1", {"argv": ["all", FILE]})
    testruns.follow_thread(seat, thread)
    testruns.follow_thread(team.seats["bob"], thread)
    events.job_done("j1", {"argv": ["all", FILE]}, {"exit": 0, "state": "done",
                                                   "summary": {"ran": 1, "reused": 2}}, thread)
    for name in ("alice", "bob"):
        [message] = inbox(team, name)
        assert "outcome=pass" in message["text"] and "ran=1 reused=2" in message["text"]
        assert "result=scripts/test result j1" in message["text"]
    [carols] = team.ws.inbox(seat.token, ack=True, limit=50)
    assert "test-job job=j1" in carols["text"]
    for name in ("alice", "bob"):
        assert inbox(team, name) == []


def test_a_canary_mismatch_posts_one_announcement_line(team):
    testruns.BoardEvents(team.seats["alice"]).canary_mismatch(FILE, "cached pass from e1 but a fresh run failed")
    rows = [r for r in team.ws.bus.log.rows() if r.get("to") == "#announcements"]
    assert len(rows) == 1 and rows[0]["type"] == "blocked" and "canary mismatch" in rows[0]["body"]
    assert "\n" not in rows[0]["body"] and len(rows[0]["body"]) <= 200


def test_another_projects_agent_cannot_follow_a_runs_thread(team):
    thread = testruns.BoardEvents(team.seats["alice"]).claimed(FILE, KEY)
    with pytest.raises(Exception, match="no board"):
        testruns.follow_thread(team.seats["zed"], thread)
    assert testruns.BoardEvents(team.seats["zed"]).claimed(FILE, KEY) != thread


def test_a_poisoned_thread_number_changes_neither_the_hit_nor_the_run(team, tmp_path):
    events = testruns.BoardEvents(team.seats["zed"])
    events.waiting(FILE, {"thread": 10 ** 6, "agent": {"id": "alice"}})
    events.waiting(FILE, {"thread": "not a number"})
    store = storage.Store(tmp_path / "store")
    assert store.claim(KEY, {"agent": {"id": "x"}}) is None
    store.note_thread(KEY, 10 ** 6)
    assert store.claimed(KEY)["thread"] == 10 ** 6 and store.claimed(KEY)["pid"] > 0


def test_a_lease_names_its_agent_and_the_board_checks_the_identity(team, tmp_path):
    folder = tmp_path / "slots"
    folder.mkdir(exist_ok=True)
    for number, agent in enumerate(({"id": "alice", "label": "reuse", "job": "j1"}, {"id": "mallory"})):
        (folder / f"{number}.slot").write_text(json.dumps({
            "label": "tests\x1b[31m", "pid": os.getpid(), "want": 2, "minimum": 1, "granted": number, "agent": agent}))
    (folder / "2.slot").write_text(json.dumps({"label": "free text", "pid": os.getpid(), "want": 1, "granted": 1}))
    seen = slots.testslots(team.ws.registry.role_of)
    named = {item["agent"]["id"]: item for item in seen["running"] + seen["waiting"] if item["agent"]}
    assert named["alice"]["agent"]["registered"] is True and named["alice"]["label"].startswith("alice/reuse (tests")
    assert named["mallory"]["agent"]["registered"] is False
    assert "\x1b" not in named["alice"]["label"]
    assert [i["agent"] for i in seen["running"] if i["label"] == "free text"] == [None]


def put_entry(base: Path, scope: str, **over) -> str:
    store = storage.Store(base / scope)
    fields = {"lookup": KEY, "file": FILE, "outcome": "pass", "manifest": {"files": {}, "dirs": {}, "dists": [],
              "env": {}, "tree": ""}, "manifest_digest": "d" * 64, "command": ["pytest", FILE],
              "junit_sha256": "e" * 64, "counts": {"tests": 3, "failed": 0, "skipped": 0}, "tree": "t" * 40,
              "runner": {"pid": 1, "started": 1.0}, "agent": {"id": "alice"}, **over}
    return store.put(fields)


def test_a_runner_entry_is_verified_for_its_own_project_only(team, tmp_path):
    base = tmp_path / "reuse"
    entry = put_entry(base, "scope-a")
    facts = testruns.verified(entry, "scope-a", base)
    assert (facts["file"], facts["outcome"], facts["agent"], facts["junit_sha256"]) == (FILE, "pass", "alice", "e" * 64)
    with pytest.raises(ValueError, match="verifies"):
        testruns.verified(entry, "scope-b", base)
    path = base / "scope-a" / "entries" / f"{entry}.json"
    forged = json.loads(path.read_text())
    forged["counts"]["tests"] = 300
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="verifies"):
        testruns.verified(entry, "scope-a", base)
    assert reuse.find("../x", "scope-a", base) is None


def test_task_checkpoint_and_submit_carry_verified_runner_entries(board, tmp_path, monkeypatch):  # noqa: F811
    base = tmp_path / "reuse"
    monkeypatch.setenv("DEV_TEST_REUSE_DIR", str(base))
    entry = put_entry(base, testruns.scope(Path.cwd()))
    kit = board
    submission = {"artifacts": {"replay.json": "a" * 64}, "checks": [{"name": "Worker claims tests", "passed": True}],
                  "summary": "done", "provenance": {"commit": "abc", "environment": "x", "model": "qwen", "runtime": "r"}}
    kit.board.claim(kit.child, kit.task["id"], kit.allocation["allocation_id"])
    saved = kit.board.checkpoint(kit.child, kit.task["id"], {"summary": "tests pass", "test_entry": entry})
    [fact] = saved["test_evidence"]
    assert (fact["id"], fact["file"], fact["key"], fact["tree"], fact["agent"]) == (entry, FILE, KEY, "t" * 40, "alice")
    with pytest.raises(ValueError, match="verifies"):
        kit.board.checkpoint(kit.child, kit.task["id"], {"summary": "x", "test_entry": "f" * 20})
    with pytest.raises(ValueError, match="at most 16"):
        kit.board.submit(kit.child, kit.task["id"], {**submission, "test_entries": "nope"})
    proposal = kit.board.submit(kit.child, kit.task["id"], {**submission, "test_entries": [entry]})
    assert [f["id"] for f in proposal["test_evidence"]] == [entry]
