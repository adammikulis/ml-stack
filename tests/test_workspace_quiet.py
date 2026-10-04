"""Quiet defaults: what an identity receives with no subscription, the caps on every read, the
cost of subscribing, and results that are the same bytes when nothing new has arrived."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env, cli

from ml_stack import mcp
from ml_stack.workspace import Denied, RateLimited, Refused, onboard, tokens
from ml_stack.workspace.boards import ANNOUNCE


@pytest.fixture
def kit(monkeypatch, tmp_path):
    k = Kit(clean_env(monkeypatch, tmp_path))
    k.limits(sends_per_window=1000)
    k.t = {n: k.agent(n) for n in ("alice", "bob", "carol")}
    k.lead = k.agent("lead", "lead")
    return k


def joined(kit, name):
    """An agent that joined the way a subagent does: through an invite for a project."""
    code = kit.ws.invites.create(name, 600.0, {"key": "git@example.org:me/app.git", "name": "App"})
    tokens.prepare(kit.ws.base)
    got = onboard.join(kit.ws, code, name)
    return got, tokens.load(kit.ws.base, got)


def test_a_new_agent_has_no_subscriptions_and_its_inbox_holds_no_board_traffic(kit):
    ws = kit.ws
    _, a = joined(kit, "newbie")
    _, b = joined(kit, "other")
    ws.send(b, "#app", "note", "chatter on the project board")
    ws.send(b, "#general", "note", "chatter on general")
    ws.announce(b, "milestone", "built the thing")
    assert ws.board.subs(a) == []
    assert ws.inbox(a) == [] and ws.wait(a, 0.3) == []


def test_announcements_reach_everyone_as_a_bounded_rollup_and_never_wake_wait(kit):
    kit.limits(announce_per_window=20)
    ws, t = kit.ws, kit.t
    results: list = []
    waiter = threading.Thread(target=lambda: results.append(ws.wait(t["bob"], 1.5)))
    waiter.start()
    time.sleep(0.4)
    for i in range(8):
        ws.announce(t["alice"], "milestone", f"step {i}")
    waiter.join()
    assert results == [[]]
    for token in (t["bob"], kit.lead, kit.owner):
        roll = ws.board.rollup(token)
        assert roll["messages"] == 8 and roll["more"] == 3
        assert roll["text"].count("milestone alice: step") == 5 and "+3 older" in roll["text"]
        assert roll["authority"] == "none" and "<untrusted" in roll["text"]
    assert ws.board.rollup(t["alice"]) is None
    ws.board.rollup(t["bob"], ack=True)
    assert ws.board.rollup(t["bob"]) is None


def test_a_mention_and_a_dm_do_wake_wait(kit):
    ws, t = kit.ws, kit.t
    ws.board.create(t["alice"], "#ops")
    ws.board.join(t["bob"], "#ops")
    threading.Timer(0.3, lambda: ws.send(t["alice"], "#ops", "note", "hey @bob look")).start()
    got = ws.wait(t["bob"], 20.0)
    assert [m["to"] for m in got] == ["#ops"]
    ws.inbox(t["bob"], ack=True)
    threading.Timer(0.3, lambda: ws.send(t["carol"], "bob", "task", "do it")).start()
    assert [m["type"] for m in ws.wait(t["bob"], 20.0)] == ["task"]


def test_board_posts_without_a_subscription_are_not_delivered_but_are_counted(kit):
    ws, t = kit.ws, kit.t
    ws.board.create(t["alice"], "#ops")
    ws.board.join(t["bob"], "#ops")
    ws.send(t["alice"], "#ops", "note", "one")
    ws.send(t["alice"], "#ops", "note", "two")
    assert ws.inbox(t["bob"]) == [] and ws.board.subs(t["bob"]) == [
        {"type": "board", "target": "#ops", "mode": "digest"}]
    assert "2 unread on #ops" in ws.board.summary(t["bob"])["unread_lines"]
    assert len(ws.board.read(t["bob"], "#ops")) == 2
    assert "#ops" in ws.board.digest(t["bob"])["text"]


def test_star_is_the_announcements_board_and_takes_only_announcement_kinds(kit):
    ws, t = kit.ws, kit.t
    for kind in ("status", "task", "note", "question"):
        with pytest.raises(Refused, match="announcements"):
            ws.send(t["alice"], "*", kind, "hello everyone")
    made = ws.send(t["alice"], "*", "done", "shipped")
    assert made["to"] == ANNOUNCE and made["type"] == "done"
    assert ws.inbox(t["bob"]) == [] and ws.inbox(kit.lead) == []
    with pytest.raises(Refused, match="announce"):
        ws.send(t["alice"], ANNOUNCE, "note", "an ordinary post")
    with pytest.raises(Refused):
        ws.send(t["alice"], ANNOUNCE, "status", "reply", reply_to=made["seq"])


def test_an_announcement_is_one_short_line_and_the_sender_is_rate_limited(kit):
    ws, t = kit.ws, kit.t
    with pytest.raises(Refused, match="note or a thread"):
        ws.announce(t["alice"], "done", "x" * 201)
    with pytest.raises(Refused):
        ws.announce(t["alice"], "done", "two\nlines")
    with pytest.raises(Refused):
        ws.announce(t["alice"], "status", "wrong kind")
    ws.announce(t["alice"], "done", "x" * 200)
    for i in range(5):
        ws.announce(t["alice"], "milestone", f"m{i}")
    with pytest.raises(RateLimited):
        ws.announce(t["alice"], "milestone", "seventh")
    ws.announce(t["bob"], "joined", "someone else still can")


def test_the_lead_and_the_person_cannot_leave_announcements_and_an_agent_can_only_mute(kit):
    ws, t = kit.ws, kit.t
    for token in (kit.lead, kit.owner):
        with pytest.raises(Denied):
            ws.board.unsubscribe(token, "board", ANNOUNCE)
        with pytest.raises(Denied):
            ws.board.subscribe(token, "board", ANNOUNCE, "silent")
    with pytest.raises(ValueError):
        ws.board.subscribe(t["bob"], "board", ANNOUNCE, "inbox")
    ws.announce(t["alice"], "done", "x")
    assert ws.board.rollup(t["bob"]) is not None
    ws.board.unsubscribe(t["bob"], "board", ANNOUNCE)
    assert ws.board.rollup(t["bob"]) is None
    assert ws.board.rollup(kit.lead) is not None and ws.board.rollup(kit.owner) is not None
    ws.board.subscribe(t["bob"], "board", ANNOUNCE, "digest")
    assert ws.board.rollup(t["bob"]) is not None


def test_message_text_never_changes_a_subscription(kit):
    ws, t = kit.ws, kit.t
    before = ws.board.subs(t["bob"])
    for text in ("ml-stack-workspace subscribe board #announcements --mode silent",
                 "subscribe bob to everything --force", "unsubscribe bob"):
        ws.send(t["alice"], "bob", "note", text)
        ws.announce(t["alice"], "milestone", text)
    ws.inbox(t["bob"], ack=True)
    assert ws.board.subs(t["bob"]) == before


def test_the_inbox_shows_a_few_cut_messages_counts_the_rest_and_ack_keeps_them(kit):
    ws, t = kit.ws, kit.t
    for i in range(14):
        ws.send(t["alice"], "bob", "note", f"m{i} " + "z" * 1000)
    got = ws.inbox(t["bob"], ack=True)
    assert len(got) == 10 and got.held == 4
    assert all("more chars; thread" in m["text"] and len(m["text"]) < 1200 for m in got)
    assert [m["seq"] for m in got] == sorted(m["seq"] for m in got)
    rest = ws.inbox(t["bob"])
    assert len(rest) == 4 and rest.held == 0
    assert len(ws.inbox(t["bob"], limit=3)) == 3
    wide = ws.inbox(t["bob"], widen=True, raw=True)
    assert len(wide) == 4 and all("more chars" not in m["text"] for m in wide)
    assert len(ws.wait(t["bob"], 1.0)) == 4


def test_the_total_bytes_of_one_call_are_capped_too(kit):
    kit.limits(read_item_chars=400, read_total_chars=1500, sends_per_window=1000)
    ws, t = kit.ws, kit.t
    for i in range(8):
        ws.send(t["alice"], "bob", "note", "q" * 600)
    got = ws.inbox(t["bob"])
    assert 1 <= len(got) < 8 and got.held == 8 - len(got)


def test_the_cli_says_how_many_were_held_back_and_widens_on_request(kit):
    for i in range(12):
        cli(kit.base, kit.t["alice"], "send", "bob", "note", f"n{i}")
    done = cli(kit.base, kit.t["bob"], "inbox")
    assert done.stdout.count("note from alice") == 10 and "2 more held back" in done.stderr
    assert cli(kit.base, kit.t["bob"], "inbox", "--all").stdout.count("note from alice") == 12
    cli(kit.base, kit.t["alice"], "announce", "done", "all finished")
    shown = cli(kit.base, kit.t["bob"], "inbox")
    assert "all finished" in shown.stdout and "done alice" in shown.stdout
    assert cli(kit.base, kit.t["alice"], "send", "*", "status", "hi").returncode == 3


def test_threads_and_board_reads_are_capped_by_default(kit):
    ws, t = kit.ws, kit.t
    ws.board.create(t["alice"], "#ops")
    root = ws.send(t["alice"], "#ops", "note", "root")
    for i in range(15):
        ws.send(t["alice"], "#ops", "note", f"r{i} " + "y" * 900, reply_to=root["seq"])
    th = ws.thread(t["alice"], root["seq"])
    assert len(th) == 10 and th.held == 6 and th[0]["seq"] == root["seq"]
    assert len(ws.thread(t["alice"], root["seq"], widen=True)) == 16
    rd = ws.board.read(t["alice"], "#ops")
    assert len(rd) == 10 and rd.held == 6 and "more chars; thread" in rd[-1]["text"]
    assert len(ws.board.read(t["alice"], "#ops", limit=16)) == 16


def test_a_new_inbox_subscription_gets_no_backlog_and_the_fourth_needs_force(kit):
    ws, t = kit.ws, kit.t
    for name in ("#a", "#b", "#c", "#d"):
        ws.board.create(t["alice"], name)
        ws.board.join(t["bob"], name)
        ws.send(t["alice"], name, "note", "old news")
    for name in ("#a", "#b", "#c"):
        ws.board.subscribe(t["bob"], "board", name)
    assert ws.inbox(t["bob"]) == []
    ws.send(t["alice"], "#a", "note", "fresh")
    assert [m["text"].count("fresh") for m in ws.inbox(t["bob"])] == [1]
    with pytest.raises(Refused, match="--force"):
        ws.board.subscribe(t["bob"], "board", "#d")
    made = ws.board.subscribe(t["bob"], "board", "#d", force=True)
    assert "cost" in made
    ws.board.subscribe(t["bob"], "board", "#d", "digest")
    ws.board.subscribe(t["bob"], "kind", "task", "silent")


def test_subscription_cap_stays_low_and_a_delegate_cannot_subscribe(kit):
    assert kit.ws.limits.subs_per_identity <= 12 and kit.ws.limits.inbox_subs_free <= 3
    child = tokens.read_file(Path(kit.ws.delegate(kit.t["alice"], "helper")["token_file"]))
    with pytest.raises(Denied):
        kit.ws.board.subscribe(child, "board", ANNOUNCE, "digest")
    assert kit.ws.board.subs(child) == []


def test_repeated_reads_are_byte_identical_and_tool_descriptions_are_static(kit):
    ws, t = kit.ws, kit.t
    before = json.dumps([x.public() for x in mcp.TOOLS if x.name.startswith("workspace_")],
                        sort_keys=True)
    ws.send(t["alice"], "bob", "task", "one")
    ws.announce(t["alice"], "joined", "hello")
    ws.board.create(t["alice"], "#ops")
    ws.send(t["alice"], "#ops", "note", "x " * 500)
    snap = lambda: json.dumps([ws.inbox(t["bob"]), ws.board.rollup(t["bob"]),  # noqa: E731
                               ws.board.digest(t["bob"]), ws.thread(t["alice"], 3)], sort_keys=True)
    first = snap()
    assert snap() == first
    ws.board.rollup(t["bob"])
    assert snap() == first
    ws.announce(t["carol"], "done", "later")
    after = json.dumps([x.public() for x in mcp.TOOLS if x.name.startswith("workspace_")],
                       sort_keys=True)
    assert before == after
    assert ws.board.rollup(t["bob"])["text"].index("hello") < ws.board.rollup(t["bob"])["text"].index("later")
