"""A subagent is its own board-named identity, and no caller-supplied field sets who is sending."""

from __future__ import annotations

import subprocess

import pytest
from test_workspace_remote import call, joined
from workspace_kit import Kit, clean_env, cli

from ml_stack import authority
from ml_stack.sentinel.human import HumanRequired
from ml_stack.workspace import agent_display, session_name, tokens
from ml_stack.workspace.identity import Denied

pytest_plugins = ["test_workspace_remote"]


@pytest.fixture(autouse=True)
def person_at_the_keyboard(monkeypatch):
    """The workspace is set up by a person, so no agent marker is set while the fixtures run."""
    monkeypatch.delenv("CLAUDECODE", raising=False)


@pytest.fixture
def kit(monkeypatch, tmp_path):
    made = Kit(clean_env(monkeypatch, tmp_path))
    made.lead_name = session_name.assign(made.base, "claude-sonnet-5-5", "claude-code", "main-1")
    made.lead = made.agent(made.lead_name)
    tokens.store(made.base, made.lead_name, made.lead)
    made.ws.register_session(made.lead, None, "claude-code")
    made.ws.set_model(made.lead_name, "claude-sonnet-5-5", terminal=(True, True), env={})
    made.bob = made.agent("bob")
    return made


def spawn(kit, parent_token, session, model=""):
    """Spawn a native session as the holder of ``parent_token``; returns the child's name and its stored token."""
    made = kit.ws.spawn(parent_token, "claude-code", session, model)
    return made["id"], tokens.load(kit.base, made["id"])


def test_a_main_session_and_two_subagents_one_nested_have_three_different_names(kit):
    first, first_token = spawn(kit, kit.lead, "agent-a", "claude-haiku-5-5")
    second, second_token = spawn(kit, first_token, "agent-b")
    assert len({kit.lead_name, first, second}) == 3
    assert all(name.startswith("claude-") for name in (first, second))
    registry = kit.ws.registry
    assert registry.info(first)["parent"] == kit.lead_name and registry.info(second)["parent"] == first
    assert kit.ws.auth(second_token).id == second and kit.ws.auth(second_token).parent == first
    assert kit.ws.model_of(first) == ("claude-haiku-5-5", "claimed")
    assert kit.ws.model_of(second) == ("claude-haiku-5-5", "inherited")


def test_the_board_shows_each_senders_parent_from_its_own_record(kit):
    first, first_token = spawn(kit, kit.lead, "agent-a")
    second, second_token = spawn(kit, first_token, "agent-b")
    kit.ws.send(second_token, "bob", "note", "from the grandchild")
    kit.ws.send(first_token, "bob", "note", "from the child")
    shown = {m["from"]: m for m in kit.ws.inbox(kit.bob)}
    assert shown[second]["from_name"] == f"{second} (spawned by {first})"
    assert shown[first]["from_name"] == f"{first} (spawned by {kit.lead_name})"
    assert shown[second]["from"] == second and shown[first]["from"] == first


def test_a_subagent_cannot_post_as_its_parent_or_as_another_subagent(kit):
    first, first_token = spawn(kit, kit.lead, "agent-a")
    second, _ = spawn(kit, kit.lead, "agent-b")
    for forged in ({"sender": kit.lead_name}, {"from_": second}, {"label": "x"}, {"parent": kit.lead_name},
                   {"name": second}):
        with pytest.raises(TypeError, match="unknown option"):
            kit.ws.send(first_token, "bob", "note", "hi", **forged)
    with pytest.raises(TypeError):
        kit.ws.announce(first_token, "milestone", "hi", label="x")
    kit.ws.send(first_token, "bob", "note", "real")
    assert [m["from"] for m in kit.ws.inbox(kit.bob)] == [first]


@pytest.mark.parametrize("flag", ["--label", "--owner", "--sender", "--from", "--parent"])
def test_the_cli_has_no_flag_that_names_a_sender_or_label(kit, flag):
    done = cli(kit.base, kit.lead, "send", "bob", "note", "hi", flag, "x")
    assert done.returncode == 2 and "unrecognized arguments" in done.stderr




def test_subagents_are_never_coordinator_eligible_at_any_depth(kit):
    first, first_token = spawn(kit, kit.lead, "agent-a")
    second, _ = spawn(kit, first_token, "agent-b")
    kit.ws.set_model(kit.lead_name, "claude-opus-5-5", terminal=(True, True), env={})
    assert agent_display.metadata(kit.ws.registry, kit.lead_name)["coordinator_eligible"] is True
    for name in (first, second):
        shown = agent_display.metadata(kit.ws.registry, name)
        assert shown["coordinator_eligible"] is False and shown["session_kind"] == "subagent"


def test_a_native_session_belongs_to_one_parent_and_a_repeat_is_idempotent(kit):
    first, _ = spawn(kit, kit.lead, "agent-a")
    again = kit.ws.spawn(kit.lead, "claude-code", "agent-a")
    assert again == {"id": first, "parent": kit.lead_name}
    with pytest.raises(Denied, match="another parent"):
        kit.ws.spawn(kit.bob, "claude-code", "agent-a")


def test_a_child_holds_no_more_rights_than_its_parent(kit):
    _, first_token = spawn(kit, kit.lead, "agent-a")
    registry = kit.ws.registry
    limited = kit.ws.auth(first_token)
    assert set(limited.can) <= set(kit.ws.auth(kit.lead).can)
    assert registry.info(limited.id)["depth"] == 1


def test_retire_ends_the_identity_and_releases_its_claims(kit):
    first, first_token = spawn(kit, kit.lead, "agent-a")
    kit.ws.claim(first_token, "branch", f"{first}/work")
    assert kit.ws.who_owns("branch", f"{first}/work")["owner"] == first
    kit.ws.retire(first_token)
    with pytest.raises(Denied):
        kit.ws.auth(first_token)
    assert kit.ws.who_owns("branch", f"{first}/work") is None
    with pytest.raises(Denied, match="only a spawned subagent"):
        kit.ws.retire(kit.lead)


def test_a_delegated_gate_refuses_a_spawned_subagent_but_not_its_lead(kit, monkeypatch, tmp_path):
    first, _ = spawn(kit, kit.lead, "agent-a")
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    monkeypatch.delenv(authority.FLOOR_ENV, raising=False)
    lead = {"CLAUDECODE": "1", "ML_STACK_WORKSPACE_AGENT": kit.lead_name}
    assert authority.require("sentinel.policy", "mode", (False, False), lead) == authority.DELEGATED
    with pytest.raises(HumanRequired, match="not a helper"):
        authority.require("sentinel.policy", "mode", (False, False), {**lead, "ML_STACK_WORKSPACE_AGENT": first})




def test_private_records_stay_lead_or_person_only(kit):
    spawn(kit, kit.lead, "agent-a")
    kit.ws.scratch_new(kit.lead, "mine")
    refused = cli(kit.base, kit.bob, "scratch-ls", "--for-agent", kit.lead_name, "--json")
    assert refused.returncode == 3 and "cannot reach" in refused.stdout


def test_a_board_call_with_a_sender_parent_name_or_label_is_refused_by_the_host(host):
    agent = joined(host, name="pc")
    for extra in ({"sender": "mac"}, {"from": "mac"}, {"label": "x"}, {"parent": "mac"}, {"name": "mac"}):
        code, reply = host.answer("a" * 32, "board", {"agent_token": agent["token"], "operation": "send",
                                                      "args": ["pc", "note", "hi"], "kwargs": {}, **extra})
        assert code == 400 and "come from the token" in reply["error"]
    for kwargs in ({"sender": "mac"}, {"label": "x"}, {"from_": "mac"}):
        code, _ = call(host, agent, "send", "pc", "note", "hi", **kwargs)
        assert code == 400


def test_the_project_route_refuses_a_header_that_names_a_sender(host):
    from types import SimpleNamespace

    from ml_stack.fleet import project_workers

    sent = []
    handler = SimpleNamespace(path="/workspace/v1/projects/" + "a" * 32 + "/board", headers={"X-ML-Stack-Sender": "mac"},
                              _sealing=lambda: None, _send=lambda code, body: sent.append((code, body)))
    assert project_workers.answer(handler, host, b"{}", None) is True
    assert sent[0][0] == 400 and "attributed to its token" in sent[0][1]["error"]


def test_the_hook_scripts_exist_without_a_label(kit):
    done = subprocess.run(["grep", "-rn", "label", "scripts/hooks/claude-subagent-start",
                           "scripts/hooks/claude-subagent-stop", "scripts/hooks/workspace_hook.py"],
                          capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert done.returncode == 1, done.stdout




def test_a_spawned_subagent_claims_every_kind_a_task_needs_and_its_parent_releases_them(kit, tmp_path):
    first, first_token = spawn(kit, kit.lead, "agent-a")
    tree = tmp_path / "tree"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    for kind, key in (("worktree", str(tree)), ("area", str(tree / "src")), ("branch", "feature/x"),
                      ("port", "9411"), ("server", "model-a")):
        assert kit.ws.claim(first_token, kind, key)["owner"] == first
    held = [row for row in kit.ws.claims.listing() if row["owner"] == first]
    assert len(held) == 5 and {row["parent"] for row in held} == {kit.lead_name}
    with pytest.raises(Exception, match="held by"):
        kit.ws.claim(kit.bob, "branch", "feature/x")
    with pytest.raises(Denied, match="belongs to"):
        kit.ws.release(kit.bob, "port", "9411")
    kit.ws.release(kit.lead, "port", "9411")
    assert kit.ws.who_owns("port", "9411") is None
    kit.ws.retire(first_token)
    assert [row for row in kit.ws.claims.listing() if row["owner"] == first] == []
    kit.ws.claim(kit.bob, "branch", "feature/x")


def test_a_spawned_subagent_still_cannot_invite_or_make_identities(kit):
    _, first_token = spawn(kit, kit.lead, "agent-a")
    with pytest.raises(Denied):
        kit.ws.delegate(first_token, "kid")
