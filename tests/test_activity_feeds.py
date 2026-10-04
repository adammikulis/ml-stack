"""Each feed into the activity log, from the real event point: what is recorded, and that no
body text reaches the log. Real chat, rules, workspace, ledger, broker, download and bench."""

from __future__ import annotations

from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack import bench, net, sentinel
from ml_stack.activity import feeds, writer
from ml_stack.reputation.store import Ledger
from ml_stack.sentinel.events import Event, Severity
from ml_stack.serve.broker import Ask
from tests.activity_support import CANARY, entries, person, ring
from tests.conftest import a_row
from tests.net_site import gguf_bytes
from tests.test_chat import call, session
from tests.test_net_download import SHA, pipe, pull, site
from tests.test_serve_broker import broker, holders, llama_binary, models

__all__ = ["broker", "holders", "llama_binary", "models", "person", "pipe", "ring", "site"]
BODY = "BODY-" + CANARY


def on_disk(directory: Path) -> bytes:
    return b"".join(p.read_bytes() for p in directory.rglob("*") if p.is_file())


def kinds():
    return [(e.kind, e.subject, e.outcome) for e in entries()]


# -- chat tool calls and confirmations ------------------------------------------------------------
def test_a_tool_call_is_recorded_with_its_outcome_and_argument_names_never_its_values(person):
    chat_, _m, seen, _out = session([call("serve_up", model=BODY, port=8080), "done"], "1\n")
    chat_.turn("start it")
    assert seen
    by_kind = {e.kind: e for e in entries()}
    tool = by_kind["agent.tool_call"]
    assert (tool.subject, tool.outcome, tool.refs["role"]) == ("serve_up", "ok", "operator")
    assert tool.meta["arg_names"] == "model,port" and tool.meta["result_chars"] > 0
    assert by_kind["approval.asked"].subject == "serve_up"
    assert by_kind["approval.answered"].outcome == "allow_once"
    assert CANARY.encode() not in on_disk(writer.directory())
    assert BODY not in " ".join(repr(e) for e in entries())


def test_a_call_the_role_denies_is_recorded_as_blocked_by_the_rail(person):
    chat_, _m, seen, _out = session([call("serve_up", model="m", port=1), "done"], role="reader")
    chat_.turn("start it")
    assert not seen
    [tool] = [e for e in entries() if e.kind == "agent.tool_call"]
    assert tool.outcome == "blocked" and tool.refs["rail"] == "role"


def test_a_call_to_a_tool_that_does_not_exist_is_recorded_as_such(person):
    chat_, _m, _s, _out = session([call("not_a_tool", x=1), "done"])
    chat_.turn("go")
    assert [e.outcome for e in entries() if e.kind == "agent.tool_call"] == ["blocked"] or \
        [e.outcome for e in entries() if e.kind == "agent.tool_call"] == ["no_such_tool"]


def test_never_allow_records_the_answer_the_rule_saved_and_the_rule_that_later_fired(person):
    args = {"model": "m.gguf", "port": 8080}
    chat_, _m, seen, _out = session([call("serve_up", **args), call("serve_up", **args), "done"], "3\n")
    chat_.turn("start it twice")
    got = kinds()
    assert ("approval.answered", "serve_up", "never") in got
    assert ("rule.added", "serve_up", "never") in got
    assert ("approval.rule_fired", "serve_up", "never") in got
    assert got.count(("approval.asked", "serve_up", "")) == 1 and not seen
    added = next(e for e in entries() if e.kind == "rule.added")
    assert added.actor == "person"


def test_removing_a_rule_is_recorded_as_the_persons_edit(person):
    from ml_stack import rules as saved
    rules = saved.Rules()
    rules.add("serve_up", {"model": "m", "port": 1}, "never", "")
    rules.flip(1)
    rules.remove(1)
    assert [(e.kind, e.outcome, e.actor) for e in entries()] == [
        ("rule.added", "never", "person"), ("rule.flipped", "always", "person"),
        ("rule.removed", "always", "person")]


def test_a_role_change_is_recorded_as_the_persons(person):
    chat_, _m, _s, _out = session([])
    chat_.use_role("reader")
    [e] = entries()
    assert (e.kind, e.actor, e.subject, e.meta["was"]) == ("role.changed", "person", "reader", "operator")


# -- sentinel events: keystore, quarantine, dialogs ------------------------------------------------------
def test_sentinel_events_are_mirrored_with_their_severity_and_nothing_loops(person):
    feeds.attach()
    node = sentinel.default()
    node.bus.emit(Event("keystore.refused", Severity.WARNING, "keystore", "purpose:memory",
                        {"outcome": "latched"}))
    node.bus.emit(Event("a.b.c.d.e", Severity.INFO, "x", ""))
    node.bus.emit(Event("net.download", Severity.INFO, "net", "artifact:x", {"url": "u"}))
    got = {(e.kind, e.subject, e.outcome): e for e in entries()}
    refused = got[("security.keystore.refused", "purpose:memory", "latched")]
    assert refused.meta["severity"] == "warning" and refused.actor == "system"
    assert ("security.event", "", "") in got
    assert not [k for k in got if k[0] == "security.net.download"]


def test_a_quarantine_reaches_the_log(person):
    feeds.attach()
    node = sentinel.default()
    node.store.quarantine(("peer", "10.9.9.9"), "forged", None)
    assert any(e.kind.startswith("security.quarantine") and e.subject == "peer:10.9.9.9" for e in entries())


def test_attaching_twice_does_not_double_every_event(person):
    feeds.attach()
    feeds.attach()
    sentinel.default().bus.emit(Event("sentinel.off", Severity.WARNING, "core", ""))
    assert [e.kind for e in entries()].count("security.sentinel.off") == 1


# -- workspace ------------------------------------------------------------------------------------------------
def test_workspace_messages_and_claims_are_recorded_by_metadata_without_the_body(person, monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    alpha, beta = kit.agent("alpha"), kit.agent("beta")
    kit.ws.send(alpha, "beta", "handoff", BODY)
    kit.ws.claim(beta, "branch", "feat/x")
    assert feeds.workspace_sync(kit.base) >= 3
    msg = next(e for e in entries() if e.kind == "workspace.message")
    assert msg.actor.endswith("alpha") or "alpha" in msg.actor
    assert msg.refs["to"] == "beta" and msg.meta["type"] == "handoff" and msg.meta["size"] == len(BODY)
    claim = next(e for e in entries() if e.kind == "workspace.claim")
    assert claim.outcome == "claimed" and claim.refs["claim"] == "branch:feat/x"
    assert BODY not in " ".join(repr(e) for e in entries())
    assert CANARY.encode() not in on_disk(writer.directory())


def test_a_second_sync_writes_only_what_is_new(person, monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    alpha = kit.agent("alpha")
    first = feeds.workspace_sync(kit.base)
    assert first >= 2 and feeds.workspace_sync(kit.base) == 0
    kit.ws.send(alpha, "*", "status", "x")
    assert feeds.workspace_sync(kit.base) == 1


# -- reputation, leases, downloads, bench -------------------------------------------------------------------
def test_a_reputation_change_is_recorded(person, tmp_path):
    ledger = Ledger(tmp_path / "rep" / "graph.enc", clock=lambda: 1_800_000_000.0, flush_s=0)
    ledger.observe("host", "bad.example", "hash_change")
    ledger.close()
    [e] = [e for e in entries() if e.kind == "reputation.observed"]
    assert e.subject == "host:bad.example" and e.meta["event"] == "hash_change" and e.outcome


def test_a_lease_records_the_model_who_asked_and_the_flags(person, broker, holders, models):
    one, two = holders(), holders()
    spec = {"context": 512, "parallel": 1, "mtp": False}
    grant = broker.lease(Ask(purpose="chat", models=(models[0],), pid=one.pid, label="pid one",
                             weight=3, spec=spec), timeout=60)
    shared = broker.lease(Ask(purpose="chat", models=(models[0],), pid=two.pid, label="pid two",
                              spec=spec), timeout=60)
    leases = [e for e in entries() if e.kind == "model.lease"]
    assert [e.outcome for e in leases] == ["granted", "shared"] and shared.shared
    first = leases[0]
    assert first.subject.startswith("model:") and first.subject.endswith(".gguf")
    assert first.refs["for"] == "pid one"
    assert first.refs["lease"] == grant.lease and first.meta["port"] == grant.port
    assert (first.meta["context"], first.meta["weight"], first.meta["mtp"]) == (512, 3, False)


def test_a_download_records_host_size_and_hash_and_a_held_one_says_so(person, tmp_path, site, pipe):
    body = gguf_bytes(extra=2048)
    site.add("/m.gguf", body)
    pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(sha256=SHA(body)))
    with pytest.raises(net.ChecksumMismatch):
        pull(pipe, site, "/m.gguf", tmp_path / "n.gguf", net.Want(sha256="0" * 64))
    ok, held = [e for e in entries() if e.kind == "net.download"]
    assert (ok.subject, ok.outcome, ok.refs["sha256"], ok.meta["size"]) == ("host:127.0.0.1", "ok", SHA(body), len(body))
    assert held.outcome == "held" and "sha256" in held.meta["reason"]
    assert not [e for e in entries() if e.kind == "security.net.download"]


def test_a_kept_bench_run_records_command_label_and_where_it_is_kept(person, tmp_path):
    rows = [a_row("who welds?", expected=["person:iris"], shown=[BODY], label="trial")]
    key = bench.save(tmp_path / "runs.ladybug", rows, server={"model": "/m/qwen-q4.gguf"})
    [e] = [e for e in entries() if e.kind == "bench.run"]
    assert (e.subject, e.outcome, e.refs["key"], e.meta["rows"], e.meta["model"]) == (
        "trial", "kept", key, 1, "qwen-q4.gguf")
    assert BODY not in repr(e) and CANARY.encode() not in on_disk(writer.directory())
