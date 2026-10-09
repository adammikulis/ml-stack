"""The model an agent runs is recorded, shown and never an authority: real registry files, real bus."""

from __future__ import annotations

import json
import re

import pytest
from workspace_kit import Kit, clean_env, cli

from poolhouse.person import HumanRequired
from poolhouse.requests import Origin
from poolhouse.workspace import Denied, onboard, tokens
from poolhouse.workspace.identity import Registry
from poolhouse.workspace.modelid import clean_harness, clean_model

pytest_plugins = ["node_kit"]
PERSON = {"terminal": (True, True), "env": {}}
HOSTILE = ["a\nb", "a\u202eb", "<b>x</b>", "x" * 81, "a b", "-lead", "a;rm", "x`y`", "a\x00b",
           "\uff41\uff42", "model\u2028x"]


@pytest.fixture
def kit(monkeypatch, tmp_path):
    k = Kit(clean_env(monkeypatch, tmp_path))
    k.limits(sends_per_window=1000, announce_per_window=1000)
    k.tokens = {n: k.agent(n) for n in ("alice", "bob")}
    return k


def test_an_old_registry_reads_as_model_unknown_and_is_rewritten_at_the_new_version(kit):
    path = kit.ws.registry.path
    old = json.loads(path.read_text())
    old["version"] = 1
    for entry in old["agents"].values():
        for key in ("model", "harness", "model_state", "models"):
            entry.pop(key, None)
    path.write_text(json.dumps(old))
    reg = Registry(kit.base)
    assert reg.model_of("alice") == ("", "")
    assert reg.info("alice")["models"] == []
    kit.ws.claim_model(kit.tokens["alice"], "claude-sonnet-5-5", "claude-code")
    assert json.loads(path.read_text())["version"] == 2
    assert kit.ws.model_of("alice") == ("claude-sonnet-5-5", "claimed")


@pytest.mark.parametrize("text", HOSTILE)
def test_a_hostile_model_or_harness_is_refused_everywhere(kit, text):
    with pytest.raises(ValueError):
        clean_model(text)
    with pytest.raises(ValueError):
        kit.ws.claim_model(kit.tokens["alice"], text)
    with pytest.raises(ValueError):
        kit.ws.set_model("alice", text, **PERSON)
    code = kit.ws.invites.create("zed", 600.0)
    with pytest.raises(ValueError):
        onboard.join(kit.ws, code, "zed", claim=(text, ""))
    assert kit.ws.model_of("alice") == ("", "")
    assert kit.ws.registry.role_of("zed") == ""
    if text:
        with pytest.raises(ValueError):
            clean_harness(text.upper() if text.isascii() else text)


def test_an_empty_model_is_not_a_model(kit):
    with pytest.raises(ValueError):
        kit.ws.claim_model(kit.tokens["alice"], "")


def test_safe_ids_are_accepted():
    for ok in ("claude-sonnet-5-5", "Qwen3.8-35B-A3B-UD-Q4_K_XL", "gpt-5.1:high", "org/m+x@1", "x" * 80):
        assert clean_model(ok) == ok


def test_join_records_a_claimed_model_and_the_code_is_kept_when_it_is_refused(kit):
    ws = kit.ws
    code = ws.invites.create("codex", 600.0)
    tokens.prepare(ws.base)
    with pytest.raises(ValueError):
        onboard.join(ws, code, "codex", claim=("bad model", ""))
    name = onboard.join(ws, code, "codex", claim=("gpt-5.1", "codex"))
    assert ws.model_of(name) == ("gpt-5.1", "claimed")
    assert ws.registry.info(name)["harness"] == "codex"
    other = ws.invites.create("zed", 600.0)
    assert ws.model_of(onboard.join(ws, other, "zed")) == ("", "")


def test_only_a_person_or_launcher_process_records_a_verified_model(kit):
    ws = kit.ws
    with pytest.raises(HumanRequired):
        ws.set_model("alice", "claude-sonnet-5-5", "claude-code", terminal=(True, True),
                     env={"CLAUDECODE": "1"})
    with pytest.raises(HumanRequired):
        ws.set_model("alice", "claude-sonnet-5-5", terminal=(False, False), env={})
    assert ws.model_of("alice") == ("", "")
    ws.set_model("alice", "claude-sonnet-5-5", "claude-code", **PERSON)
    assert ws.model_of("alice") == ("claude-sonnet-5-5", "verified")
    ws.set_model("bob", "qwen", verified=False, **PERSON)
    assert ws.model_of("bob") == ("qwen", "claimed")
    with pytest.raises(ValueError):
        ws.set_model("nobody", "qwen", **PERSON)


def test_an_agent_cannot_verify_itself_or_overwrite_a_verified_model_or_set_anothers(kit):
    ws = kit.ws
    ws.set_model("alice", "claude-sonnet-5-5", "claude-code", **PERSON)
    with pytest.raises(Denied):
        ws.claim_model(kit.tokens["alice"], "claude-fable-5-1")
    assert ws.model_of("alice") == ("claude-sonnet-5-5", "verified")
    ws.claim_model(kit.tokens["alice"], "claude-sonnet-5-5")
    assert ws.model_of("alice") == ("claude-sonnet-5-5", "verified")
    ws.claim_model(kit.tokens["bob"], "gpt-5.1")
    assert ws.model_of("bob") == ("gpt-5.1", "claimed")
    assert ws.model_of("alice") == ("claude-sonnet-5-5", "verified")


def test_a_claimed_model_changes_no_right(kit):
    ws = kit.ws
    before = ws.registry.info("alice")
    for claim in ("lead", "human", "owner", "claude-opus-admin@root"):
        ws.claim_model(kit.tokens["alice"], claim)
        who = ws.auth(kit.tokens["alice"])
        assert (who.role, who.can, who.parent) == ("agent", before["can"] and tuple(before["can"]), "")
        assert ws.registry.role_of("alice") == "agent" and who.trust == "agent-claimed"
        with pytest.raises(Denied):
            ws.mint(kit.tokens["alice"], "mallory", "lead", 60.0)
        with pytest.raises(Denied):
            ws.gc(kit.tokens["alice"])
    after = ws.registry.info("alice")
    assert {k: after[k] for k in ("role", "can", "parent", "expires", "revoked")} == {
        k: before[k] for k in ("role", "can", "parent", "expires", "revoked")}


def test_the_model_shows_in_headers_listings_board_json_and_the_activity_fields(kit):
    ws = kit.ws
    ws.claim_model(kit.tokens["alice"], "gpt-5.1", "codex")
    ws.set_model("bob", "claude-sonnet-5-5", "claude-code", **PERSON)
    sent = ws.send(kit.tokens["alice"], "bob", "question", "which tree?")
    assert sent["from_model"] == "gpt-5.1" and sent["from_model_state"] == "claimed"
    assert "(gpt-5.1, claimed)" in sent["text"].splitlines()[1]
    got = ws.inbox(kit.tokens["bob"])[0]
    assert got["from_model"] == "gpt-5.1"
    ws.send(kit.tokens["bob"], "alice", "answer", "this one")
    rows = {r["id"]: r for r in ws.registered()}
    assert (rows["alice"]["model"], rows["alice"]["model_state"]) == ("gpt-5.1", "claimed")
    assert (rows["bob"]["model"], rows["bob"]["model_state"]) == ("claude-sonnet-5-5", "verified")
    dm = ws.board.dm(kit.tokens["alice"], "bob")
    assert {m["from_model"] for m in dm if m["from"] == "alice"} == {"gpt-5.1"}
    page = ws.board.ui_dm(kit.owner, "alice", "bob")
    assert {(m["model"], m["model_state"]) for m in page if m["from"] == "alice"} == {("gpt-5.1", "claimed")}
    claimed = ws.who_owns("branch", "x") or ws.claim(kit.tokens["alice"], "branch", "x")
    assert ws.who_owns("branch", "x")["owner_model"] == "gpt-5.1" and claimed
    audit = [r for r in ws.audit_log.rows() if r["event"] == "message" and r["who"] == "alice"]
    assert audit[-1]["model"] == "gpt-5.1" and audit[-1]["verified"] is False


def test_a_node_session_claims_a_model_through_whoami_and_it_shows_in_the_inbox_and_the_listing(workspace_node):
    node = workspace_node
    alice, bob = node.member("alice", model="gpt-5.1", harness="codex"), node.member("bob")
    node.session(alice).post(bob.name, "question", "which tree?")
    done = node.cli("whoami", "--model", "gpt-5.2", "--json", who=bob)
    assert done.returncode == 0 and json.loads(done.stdout)["model_state"] == "claimed"
    assert node.cli("whoami", "--model", "x y", who=bob).returncode != 0
    assert re.search(r"question from .+ \(gpt-5\.1, claimed\)", node.cli("inbox", who=bob).stdout)
    listing = node.cli("agents", who=alice).stdout
    assert re.search(r"gpt-5\.1, claimed", listing) and re.search(r"gpt-5\.2, claimed", listing)


def test_a_message_keeps_the_model_it_was_sent_under_and_a_swap_is_announced_with_history(kit):
    ws = kit.ws
    ws.claim_model(kit.tokens["alice"], "claude-sonnet-5-5", "claude-code")
    first = ws.send(kit.tokens["alice"], "bob", "note", "one")
    ws.claim_model(kit.tokens["alice"], "claude-fable-5-1")
    ws.claim_model(kit.tokens["alice"], "claude-fable-5-1")
    second = ws.send(kit.tokens["alice"], "bob", "note", "two")
    assert (first["from_model"], second["from_model"]) == ("claude-sonnet-5-5", "claude-fable-5-1")
    assert ws.deliver(ws.bus.get(first["seq"]))["from_model"] == "claude-sonnet-5-5"
    history = ws.registry.info("alice")["models"]
    assert [(h["model"], h["verified"]) for h in history] == [
        ("claude-sonnet-5-5", False), ("claude-fable-5-1", False)]
    news = [m["text"] for m in ws.board.read(kit.owner, "#announcements", limit=50)]
    assert any("alice now runs claude-fable-5-1" in t for t in news)
    assert not any("alice now runs claude-sonnet-5-5" in t for t in news)


def test_a_subagent_has_its_own_model_or_inherits_the_parents(kit):
    ws = kit.ws
    ws.claim_model(kit.tokens["alice"], "claude-sonnet-5-5")
    plain = ws.spawn(kit.tokens["alice"], "claude-code", "speed-1")["id"]
    inherited = ws.send(tokens.load(kit.base, plain), "bob", "note", "x")
    assert (inherited["from_model"], inherited["from_model_state"]) == ("claude-sonnet-5-5", "inherited")
    own_name = ws.spawn(kit.tokens["alice"], "claude-code", "speed-2", "claude-haiku-5")["id"]
    own = ws.send(tokens.load(kit.base, own_name), "bob", "note", "y")
    assert (own["from_model"], own["from_model_state"]) == ("claude-haiku-5", "claimed")
    assert ws.model_of("alice") == ("claude-sonnet-5-5", "claimed")
    child = ws.delegate(kit.tokens["alice"], "kid")["id"]
    assert ws.model_of(child) == ("claude-sonnet-5-5", "inherited")


def test_the_output_is_byte_stable(kit):
    ws = kit.ws
    ws.claim_model(kit.tokens["alice"], "gpt-5.1", "codex")
    row = ws.send(kit.tokens["alice"], "bob", "note", "same")
    assert json.dumps(ws.deliver(ws.bus.get(row["seq"])), sort_keys=True) == json.dumps(
        ws.deliver(ws.bus.get(row["seq"])), sort_keys=True)
    one = cli(kit.base, kit.tokens["alice"], "agents").stdout
    assert one == cli(kit.base, kit.tokens["alice"], "agents").stdout


def test_a_request_names_the_model_beside_the_requester():
    assert Origin("claude", "p", "", "claude-sonnet-5-5", "claimed").who == \
        "claude (claude-sonnet-5-5, claimed)"
    assert Origin("claude").who == "claude (model unknown)"
    assert Origin().who == ""
