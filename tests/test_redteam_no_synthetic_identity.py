"""Red team: no agent can obtain an identity for another actor or one that reads as a person.

Real workspaces in temporary state, real child processes for the command line, no mocks.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import Denied, agent_display, onboard, tokens
from ml_stack.workspace.cli import TABLE
from ml_stack.workspace.identity import AGENT, HUMAN, LEAD, RESERVED, valid_name

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CREATING = {"adopt", "mint", "enroll_project", "bootstrap_agent", "_add", "init", "delegate"}

ALLOWED = {
    "src/ml_stack/workspace/agent_invites.py": {"adopt"},
    "src/ml_stack/workspace/cli.py": {"init", "mint"},
    "src/ml_stack/workspace/device_agent.py": {"_add", "mint"},
    "src/ml_stack/workspace/guide.py": {"bootstrap_agent", "init"},
    "src/ml_stack/workspace/identity.py": {"_add"},
    "src/ml_stack/workspace/localstart.py": {"mint"},
    "src/ml_stack/workspace/onboard.py": {"adopt", "init", "mint"},
    "src/ml_stack/workspace/remote_host.py": {"enroll_project", "init"},
    "src/ml_stack/workspace/service.py": {"init", "mint"},
}
PENDING_REMOVAL = {
    "src/ml_stack/harnessid.py": {"delegate"},
    "src/ml_stack/workspace/automatic_connection.py": {"delegate"},
    "src/ml_stack/workspace/localstart.py": {"delegate"},
    "src/ml_stack/workspace/remote_host.py": {"delegate"},
    "src/ml_stack/workspace/service.py": {"delegate"},
}
IDENTITY_VERBS = {"init", "mint", "join", "invite", "connect", "setup", "agent"}


def call_sites() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    paths = [*(SRC / "ml_stack" / "workspace").rglob("*.py"), SRC / "ml_stack" / "harnessid.py"]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CREATING:
                found.setdefault(path.relative_to(ROOT).as_posix(), set()).add(node.func.attr)
    return found


def merged(*tables: dict[str, set[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for table in tables:
        for path, names in table.items():
            out.setdefault(path, set()).update(names)
    return out


def test_identity_creating_call_sites_are_the_reviewed_ones():
    assert call_sites() == merged(ALLOWED, PENDING_REMOVAL)


def test_the_command_line_has_no_verb_that_mints_for_another_actor():
    verbs = {row[0]: row[1] for row in TABLE}
    assert "delegate" not in verbs
    risky = {name for name, text in verbs.items()
             if any(word in text.lower() for word in ("mint", "child identity", "delegate", "new identity"))}
    assert risky <= IDENTITY_VERBS


def test_the_removed_verb_is_refused_by_the_real_command(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    worker = kit.agent("worker")
    done = subprocess.run([sys.executable, "-m", "ml_stack.workspace.cli", "delegate", "kid", "--agent", "worker"],
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC), "CLAUDECODE": "1",
                               "ML_STACK_WORKSPACE_HOME": str(kit.base)},
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, check=False)
    assert done.returncode != 0
    assert worker not in done.stdout
    assert "worker/kid" not in kit.ws.registry.ids()


@pytest.mark.parametrize("role", [HUMAN, LEAD, AGENT])
def test_no_agent_or_lead_token_mints_any_role(tmp_path, monkeypatch, role):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    holders = {"agent": kit.agent("worker"), "lead": kit.agent("lead-1", "lead")}
    for who, token in holders.items():
        with pytest.raises(Denied):
            kit.ws.mint(token, f"made-{who}-{role}", role)
    assert not any(name.startswith("made-") for name in kit.ws.registry.ids())


def test_an_invite_always_yields_a_standard_agent_whoever_issued_it(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.limits(sends_per_window=1000, announce_per_window=1000, agent_invite_ask="plan-and-go",
               agent_invites_open=10, max_children=10)
    monkeypatch.setenv("CLAUDECODE", "1")
    for issuer_role, name in ((AGENT, "worker"), (LEAD, "lead-1")):
        token = kit.agent(name, issuer_role)
        made = kit.ws.invite(token, "peer", 1800.0, 1)
        record, = (r for r in kit.ws.invites._load()["invites"].values() if r["issuer"] == name)
        assert not {"role", "roles"} & set(record)
        joined = onboard.join(kit.ws, made["code"], f"peer-{name}")
        info = kit.ws.registry.info(joined)
        assert info["role"] == AGENT and info["parent"] == name
        assert kit.ws.auth(tokens.load(kit.base, joined)).trust == "agent-claimed"


def test_no_registered_identity_other_than_a_person_renders_as_a_person(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    worker = kit.agent("worker")
    kit.agent("lead-1", "lead")
    kid = kit.ws.delegate(worker, "kid")["id"]
    for name in ("worker", "lead-1", kid):
        shown = agent_display.metadata(kit.ws.registry, name)
        assert shown["session_kind"] != "person"
        assert kit.ws.registry.info(name)["role"] != HUMAN
    assert kit.ws.registry.info("owner")["role"] == HUMAN


@pytest.mark.parametrize("name", [*sorted(RESERVED), "Human", "OWNER"])
def test_a_person_looking_identity_name_cannot_be_registered(tmp_path, monkeypatch, name):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    assert not valid_name(name)
    with pytest.raises(ValueError):
        kit.ws.mint(kit.owner, name, "agent")


def test_a_second_person_cannot_be_created_from_an_agent_process(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(Denied):
        kit.ws.registry.init("another-person")
    roles = [kit.ws.registry.info(name)["role"] for name in kit.ws.registry.ids()]
    assert roles.count(HUMAN) == 1
