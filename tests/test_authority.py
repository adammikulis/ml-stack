"""The delegable authority registry: its states, the gates that read it and the ones that never do."""

from __future__ import annotations

import argparse
import re

import pytest
from taskboard_kit import board as _board_fixture

from poolhouse import authority, chatpolicy
from poolhouse.net import cli as net_cli
from poolhouse.person import HumanRequired
from poolhouse.sentinel import human
from poolhouse.serve import wired, wired_apply
from poolhouse.workspace import authority_cli, enforcement
from poolhouse.workspace.identity import Denied

board = _board_fixture
LEAD = {"CLAUDECODE": "1", "POOLHOUSE_WORKSPACE_AGENT": "claude-code"}


@pytest.fixture(autouse=True)
def registry(monkeypatch, tmp_path):
    monkeypatch.delenv(authority.FLOOR_ENV, raising=False)
    for name in (*authority.DELEGATING, "POOLHOUSE_NONINTERACTIVE", "POOLHOUSE_WORKSPACE_AGENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))


def uses():
    return [row for row in authority.audit_rows() if row["event"] == "authority.use"]


def test_every_gate_names_a_group_and_is_delegated_until_set():
    assert authority.GATES["serve.wired-limit"] == "serve"
    assert {row["state"] for row in authority.show()["gates"]} == {authority.DELEGATED}


def test_set_selects_all_a_group_or_a_gate_and_persists():
    authority.set_state(["sentinel"], authority.PERSON, by="lead")
    assert authority.state_of("sentinel.quarantine") == authority.state_of("sentinel.policy") == authority.PERSON
    assert authority.state_of("serve.wired-limit") == authority.DELEGATED
    authority.set_state(["ALL"], authority.PERSON, by="lead")
    authority.set_state(["serve.wired-limit", "fleet.recovery"], authority.DELEGATED, by="lead")
    held = {row["gate"]: row["state"] for row in authority.show()["gates"]}
    assert held["serve.wired-limit"] == held["fleet.recovery"] == authority.DELEGATED
    assert held["sentinel.policy"] == authority.PERSON


def test_an_unknown_name_is_refused_and_changes_nothing():
    with pytest.raises(KeyError):
        authority.set_state(["ALL", "nonsense"], authority.PERSON, by="lead")
    assert authority.state_of("chat.policy") == authority.DELEGATED


def test_a_project_has_its_own_state_over_the_machine_s():
    authority.set_state(["chat.policy"], authority.PERSON, by="lead", project="alpha")
    assert authority.state_of("chat.policy", "alpha") == authority.PERSON
    assert authority.state_of("chat.policy", "beta") == authority.state_of("chat.policy") == authority.DELEGATED


def test_the_floor_variable_can_only_tighten(monkeypatch):
    monkeypatch.setenv(authority.FLOOR_ENV, "person")
    assert authority.state_of("serve.wired-limit") == authority.PERSON
    monkeypatch.setenv(authority.FLOOR_ENV, "delegated")
    authority.set_state(["ALL"], authority.PERSON, by="lead")
    assert authority.state_of("serve.wired-limit") == authority.PERSON


def test_a_helper_identity_cannot_flip_and_a_flip_is_audited():
    with pytest.raises(HumanRequired):
        authority.set_state(["ALL"], authority.PERSON, by="lead/helper")
    assert authority.state_of("chat.policy") == authority.DELEGATED
    authority.set_state(["chat.policy"], authority.PERSON, by="lead")
    row = authority.audit_rows()[-1]
    assert (row["by"], row["gates"], row["from"], row["to"]) == ("lead", ["chat.policy"], ["delegated"], "person")


def test_a_person_state_refuses_a_marked_process():
    authority.set_state(["ALL"], authority.PERSON, by="lead")
    with pytest.raises(HumanRequired, match="for a person"):
        authority.require("sentinel.policy", "mode", (True, True), LEAD)
    assert authority.require("sentinel.policy", "mode", (True, True), {}) == authority.PERSON
    assert not uses()


def test_a_delegated_gate_passes_a_lead_agent_and_audits_each_use():
    assert authority.require("sentinel.policy", "mode", (False, False), LEAD) == authority.DELEGATED
    authority.require("sentinel.policy", "mode", (False, False), LEAD)
    assert [(u["gate"], u["agent"]) for u in uses()] == [("sentinel.policy", "claude-code")] * 2


def test_a_delegated_gate_refuses_helpers_and_unattended_processes():
    for env in ({**LEAD, "POOLHOUSE_WORKSPACE_AGENT": "claude-code/reader"},):
        with pytest.raises(HumanRequired, match="not a helper"):
            authority.require("sentinel.policy", "mode", (False, False), env)
    for env in ({}, {"POOLHOUSE_NONINTERACTIVE": "1"}):
        with pytest.raises(HumanRequired):
            authority.require("sentinel.policy", "mode", (False, False), env)
    assert not uses()


def test_a_person_at_a_terminal_passes_a_delegated_gate_unaudited():
    assert authority.require("sentinel.policy", "mode", (True, True), {}) == authority.PERSON
    assert not uses()


def test_a_delegated_quarantine_gate_grants_without_the_typed_confirmation(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")

    def typed(prompt):
        raise AssertionError("asked a person to type")

    grant = human.mint_gated("sentinel.quarantine", "release", "q1", typed=typed, terminal=(False, False))
    grant.check("release", "q1")
    authority.set_state(["sentinel.quarantine"], authority.PERSON, by="lead")
    with pytest.raises(HumanRequired):
        human.mint_gated("sentinel.quarantine", "release", "q1", typed=typed, terminal=(False, False))


def wired_call(env, via):
    runs = []

    def runner(argv, capture):
        runs.append(argv)
        return 0, "", ""

    hooks = wired.Hooks(total=64 * 1024**3, current=0, others=0, env=env, runner=runner,
                        system="Darwin", terminal=(False, False), read=lambda: 40960)
    return wired_apply.set_limit(40960, via=via, hooks=hooks), runs


def test_the_wired_limit_goes_through_the_administrator_dialog_for_a_delegated_lead():
    done, runs = wired_call(LEAD, "osascript")
    assert done.ok and runs[0][0] == "osascript" and "with administrator privileges" in runs[0][-1]
    with pytest.raises(HumanRequired, match="needs a terminal"):
        wired_call(LEAD, "sudo")
    authority.set_state(["serve.wired-limit"], authority.PERSON, by="lead")
    with pytest.raises(HumanRequired, match="for a person"):
        wired_call(LEAD, "osascript")


def test_chat_policy_refuses_human_only_wording_only_while_it_is_a_person_s():
    text = "release the thing from quarantine"
    assert chatpolicy.refusal_for(text) is None
    authority.set_state(["chat.policy"], authority.PERSON, by="lead")
    assert chatpolicy.refusal_for(text)[0].startswith("release or purge")


PERSON_ONLY = ("keystore", "passphrase", "sudoers", "signing", "cluster-token", "init", "review", "os-prompt")


def test_secrets_and_the_os_prompt_have_no_gate_in_the_registry():
    assert not [g for g in authority.GATES if any(word in g for word in PERSON_ONLY)]


def test_the_keystore_stays_a_person_s_whatever_the_registry_says(monkeypatch, capsys):
    authority.set_state(["ALL"], authority.DELEGATED, by="lead")
    monkeypatch.setenv("CLAUDECODE", "1")
    assert net_cli.unlock(argparse.Namespace(json=False)) == 2
    assert "for a person" in capsys.readouterr().err


def test_the_cluster_passphrase_stays_a_person_s(monkeypatch, capsys):
    from poolhouse.fleet import recovery
    authority.set_state(["ALL"], authority.DELEGATED, by="lead")
    monkeypatch.setenv("CLAUDECODE", "1")
    args = argparse.Namespace(cmd="passphrase", group="", cluster_key=None)
    assert recovery.run(args) == 2
    assert "for a person" in capsys.readouterr().err


def test_the_gates_of_no_registry_name_survive_in_the_source():
    from pathlib import Path
    root = Path(authority.__file__).parent
    pinned = {"net/cli.py": 'human.require_person("unlock")', "fleet/recovery.py": 'human.require_person("passphrase")',
              "fleet/runtime_paths.py": 'person.require_person("display the cluster token")',
              "sentinel/review.py": 'human.require_person("review")'}
    for name, call in pinned.items():
        assert call in (root / name).read_text()
    assert not re.search(r"authority\.require\([^)]*(keystore|passphrase|unlock)", (root / "net/cli.py").read_text())


def args(action, *words, project=""):
    return argparse.Namespace(action=action, words=list(words), project=project)


def events(kit, name):
    return [row for row in kit.ws.audit_log.rows() if row["event"] == name]


def test_show_set_and_preset_through_the_workspace_command(board):
    shown = authority_cli.run(args("show"), board.ws, board.parent)
    assert {row["state"] for row in shown["gates"]} == {authority.DELEGATED}
    done = authority_cli.run(args("set", "person", "sentinel", "serve.wired-limit"), board.ws, board.parent)
    assert {c["gate"] for c in done["changes"]} == {"sentinel.quarantine", "sentinel.policy", "serve.wired-limit"}
    assert authority.state_of("serve.wired-limit") == authority.PERSON
    flip = events(board, "authority.set")[-1]
    assert (flip["who"], flip["from"], flip["to"]) == ("lead", ["delegated"], "person")


def test_prod_makes_the_sensitive_gates_a_person_s_and_enforcement_strict_and_dev_undoes_both(board):
    prod = authority_cli.run(args("preset", "prod", project="alpha"), board.ws, board.parent)
    assert prod["enforcement"]["mode"] == "strict" and enforcement.mode(board.ws, {"key": "alpha"}) == "strict"
    assert authority.state_of("sentinel.quarantine", "alpha") == authority.PERSON
    assert authority.state_of("workspace.setup", "alpha") == authority.DELEGATED
    dev = authority_cli.run(args("preset", "dev", project="alpha"), board.ws, board.parent)
    assert dev["enforcement"]["mode"] == "open"
    assert {row["state"] for row in authority.show("alpha")["gates"]} == {authority.DELEGATED}
    assert authority.show()["preset"] == "dev"


def test_a_helper_identity_cannot_set_or_preset(board):
    for call in (args("set", "person", "ALL"), args("preset", "prod", project="alpha")):
        with pytest.raises(Denied):
            authority_cli.run(call, board.ws, board.child)
    assert authority.state_of("sentinel.quarantine") == authority.DELEGATED
    assert enforcement.mode(board.ws, {"key": "alpha"}) == "open"
    assert not events(board, "authority.set")


def test_a_bad_command_is_refused_with_a_message(board):
    for call in (args("set", "maybe", "ALL"), args("set", "person"), args("set", "person", "nope"), args("preset", "x"),
                 args("preset")):
        with pytest.raises(ValueError):
            authority_cli.run(call, board.ws, board.parent)
