"""A trusted machine: trust-machine and untrust-machine, the join without a code, setup --yes
without a terminal, the WSL marker check and the SessionStart hook."""

from __future__ import annotations

import importlib.machinery
import io
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from workspace_kit import SRC, STRIPPED, clean_env

from ml_stack import person
from ml_stack.person import HumanRequired
from ml_stack.workspace import Workspace, onboard, tokens, trusted
from ml_stack.workspace.identity import AGENT, Denied

REPO = Path(__file__).resolve().parents[1]
AT_TERMINAL = (True, True)
HERE = {"key": "k-alpha", "name": "alpha"}


@pytest.fixture
def ws(monkeypatch, tmp_path):
    return Workspace(clean_env(monkeypatch, tmp_path))


def trust(ws, project=None):
    return trusted.trust(ws, project or {}, terminal=AT_TERMINAL, env={})


def join(ws, name="claude-code", here=HERE, **env):
    return trusted.join(ws, name, ("claude-opus-5-5", "claude-code"), here=here, env=env)


def child(argv, base, **env):
    full = {**{k: v for k, v in os.environ.items() if k not in STRIPPED},
            "ML_STACK_WORKSPACE_HOME": str(base), "PYTHONPATH": SRC, **env}
    return subprocess.run([sys.executable, "-m", "ml_stack.workspace.cli", *argv], env=full,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          timeout=60, check=False)


def owner_token(ws):
    return (ws.base / "tokens" / tokens.OWNER_FILE).read_text().strip()


def announced(ws):
    return [r["body"] for r in ws.bus.log.rows() if r.get("to") == "#announcements"]


def audited(ws, event):
    return [r for r in ws.audit_log.rows() if r["event"] == event]


@pytest.mark.parametrize("marker", ["CLAUDECODE", "ML_STACK_NONINTERACTIVE"])
def test_trust_and_untrust_are_refused_to_an_agent(ws, marker):
    with pytest.raises(HumanRequired, match=marker):
        trusted.trust(ws, {}, terminal=AT_TERMINAL, env={marker: "1"})
    with pytest.raises(HumanRequired, match=marker):
        trusted.untrust(ws, terminal=AT_TERMINAL, env={marker: "1"})
    assert not trusted.path(ws.base).exists()


def test_trust_machine_from_an_agent_shell_writes_nothing(ws):
    done = child(["trust-machine"], ws.base, CLAUDECODE="1")
    assert done.returncode == 3 and "agent" in done.stderr
    assert not trusted.path(ws.base).exists()


def test_trust_is_refused_while_tainted(ws):
    with pytest.raises(Denied, match="ML_STACK_TAINTED"):
        trusted.trust(ws, {}, terminal=AT_TERMINAL, env={"ML_STACK_TAINTED": "1"})
    assert not trusted.path(ws.base).exists()


def test_the_trust_file_is_private_and_initialises_the_workspace(ws):
    target = trust(ws)
    assert tokens.problem(target) == ""
    assert tokens.problem(ws.base / "tokens" / tokens.OWNER_FILE) == ""
    assert ws.registry.role_of("owner") == "human"
    assert len(audited(ws, "trusted_machine.on")) == 1


def test_a_join_without_trust_is_refused_and_names_both_ways_in(ws):
    with pytest.raises(Denied) as err:
        join(ws)
    assert "connect" in str(err.value) and "trust-machine" in str(err.value)
    assert not ws.registry.role_of("claude-code")
    done = child(["join", "--name", "codex"], ws.base)
    assert done.returncode == 3 and "trust-machine" in done.stderr


def test_a_trusted_join_is_an_agent_audited_announced_and_placed(ws):
    trust(ws)
    name, reused = join(ws)
    assert (name, reused) == ("claude-code", False)
    who = ws.auth(tokens.load(ws.base, name))
    assert who.role == AGENT and not who.parent
    assert tokens.problem(ws.base / "tokens" / name) == ""
    info = ws.registry.info(name)
    assert info["project"] == HERE and info["model"] == "claude-opus-5-5"
    assert info["harness"] == "claude-code"
    rows = audited(ws, "trusted_machine.join")
    assert [r["who"] for r in rows] == [name] and rows[0]["project"] == "alpha"
    assert any(name in body and "trusted machine" in body for body in announced(ws))


def test_the_cli_joins_without_a_code_then_says_it_is_already_joined(ws):
    trust(ws)
    first = child(["join", "--name", "codex", "--model", "gpt-9", "--harness", "codex"], ws.base)
    assert first.returncode == 0 and first.stdout.strip() == "joined as codex"
    again = child(["join", "--name", "codex", "--harness", "codex"], ws.base)
    assert again.returncode == 0 and again.stdout.strip() == "already joined as codex"


def test_a_working_token_is_reused_and_its_model_updated(ws):
    trust(ws)
    join(ws)
    before = (ws.base / "tokens" / "claude-code").read_text()
    name, reused = trusted.join(ws, "claude-code", ("claude-sonnet-5-5", "claude-code"), here=HERE, env={})
    assert (name, reused) == ("claude-code", True)
    assert (ws.base / "tokens" / "claude-code").read_text() == before
    assert ws.registry.info("claude-code")["model"] == "claude-sonnet-5-5"
    assert len(audited(ws, "trusted_machine.join")) == 1


def test_a_taken_name_whose_token_is_not_here_gets_a_suffix(ws):
    trust(ws)
    ws.mint(owner_token(ws), "codex")
    name, reused = join(ws, "codex")
    assert name.startswith("codex-") and not reused


@pytest.mark.parametrize("wanted", ["admin", "lead", "trusted-machine", "ml-stack-x", "human", "Bad Name"])
def test_reserved_and_invalid_names_are_refused(ws, wanted):
    trust(ws)
    with pytest.raises(ValueError):
        join(ws, wanted)
    assert not [n for n in ws.registry.ids() if ws.registry.info(n)["role"] == AGENT]


def test_a_tainted_session_cannot_join_without_a_code(ws):
    trust(ws)
    with pytest.raises(Denied, match="ML_STACK_TAINTED"):
        join(ws, ML_STACK_TAINTED="1")
    assert not ws.registry.role_of("claude-code")


def test_untrust_stops_new_joins_and_revoke_still_stops_one_agent(ws):
    trust(ws)
    join(ws, "codex")
    assert trusted.untrust(ws, terminal=AT_TERMINAL, env={})
    with pytest.raises(Denied, match="not trusted"):
        join(ws, "claude-code")
    assert ws.auth(tokens.load(ws.base, "codex")).id == "codex"
    assert len(audited(ws, "trusted_machine.off")) == 1
    assert not trusted.untrust(ws, terminal=AT_TERMINAL, env={})


def test_a_revoked_agent_does_not_come_back_through_trust(ws):
    trust(ws)
    join(ws)
    ws.revoke(owner_token(ws), "claude-code")
    with pytest.raises(Denied, match="revoked"):
        join(ws)
    assert not ws.registry.role_of("claude-code")


def test_trust_for_one_project_refuses_a_join_from_another(ws):
    trust(ws, HERE)
    with pytest.raises(Denied, match="alpha"):
        join(ws, here={"key": "k-beta", "name": "beta"})
    assert join(ws)[0] == "claude-code"


def test_the_live_identity_cap_holds_for_trusted_joins(ws):
    (ws.base / "limits.json").write_text(json.dumps({"version": 1, "mints_per_identity": 1}))
    ws = Workspace(ws.base)
    trust(ws)
    join(ws, "one")
    with pytest.raises(Denied, match="limits"):
        join(ws, "two")


def test_a_trust_file_others_can_read_or_a_damaged_one_is_refused(ws):
    target = trust(ws)
    target.write_text("{}")
    with pytest.raises(Denied, match="damaged|lets others"):
        join(ws)


def test_setup_yes_without_a_terminal_works_and_is_still_refused_to_an_agent(ws):
    done = child(["setup", "--yes", "codex"], ws.base)
    assert "created: codex" in done.stdout, done.stderr
    assert ws.auth(tokens.load(ws.base, "codex")).id == "codex"
    refused = child(["setup", "--yes", "other"], ws.base, CLAUDECODE="1")
    assert refused.returncode == 3 and "agent" in refused.stderr
    assert not (ws.base / "tokens" / "other").exists()


def test_setup_refuses_an_agent_in_process(ws, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(HumanRequired, match="CLAUDECODE"):
        onboard.setup(ws, ["codex"], [], 3600.0)
    assert not ws.registry.ids()


def test_a_windows_process_started_from_wsl_counts_as_an_agent_unless_the_wrapper_forwarded(monkeypatch):
    monkeypatch.setattr(person, "from_wsl", lambda: True)
    for name in (*person.AGENT_MARKERS, person.WSL_FORWARDED):
        monkeypatch.delenv(name, raising=False)
    assert "WSL" in person.marked()
    with pytest.raises(HumanRequired, match="WSL"):
        person.require_unmarked("workspace setup")
    monkeypatch.setenv(person.WSL_FORWARDED, "1")
    assert person.marked() == ""
    monkeypatch.setenv("CLAUDECODE", "1")
    assert person.marked() == "CLAUDECODE is set"
    assert person.marked({}) == ""


def hook():
    path = REPO / "scripts" / "hooks" / "claude-session-join"
    loader = importlib.machinery.SourceFileLoader("claude_session_join", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize(("said", "code", "context"), [
    ("joined as claude-code", 0, True), ("already joined as claude-code", 0, False),
    ("", 3, False)])
def test_the_session_hook_speaks_only_after_a_new_join(monkeypatch, tmp_path, capsys, said, code, context):
    fake = tmp_path / "fake.py"
    fake.write_text(f"import sys, json\nopen(r'{tmp_path / 'argv.json'}', 'w').write(json.dumps(sys.argv[1:]))\n"
                    f"print({said!r})\nsys.exit({code})\n")
    module = hook()
    monkeypatch.setattr(module, "command", lambda: [sys.executable, str(fake)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"model": "claude-opus-5-5", "cwd": str(tmp_path)})))
    assert module.main() == 0
    out = capsys.readouterr().out
    assert bool(out) == context
    if context:
        assert "claude-code" in json.loads(out)["hookSpecificOutput"]["additionalContext"]
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv == ["join", "--name", "claude-code", "--harness", "claude-code", "--model", "claude-opus-5-5"]


def test_the_session_hook_is_silent_without_the_command(monkeypatch, capsys):
    module = hook()
    monkeypatch.setattr(module, "command", list)
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    assert module.main() == 0 and capsys.readouterr().out == ""


@pytest.mark.skipif(os.name == "nt", reason="the wrapper is a POSIX shell script")
def test_the_wsl_wrapper_refuses_outside_wsl():
    release = Path("/proc/sys/kernel/osrelease")
    if os.environ.get("WSL_DISTRO_NAME") or (release.exists() and "microsoft" in release.read_text().lower()):
        pytest.skip("this machine is WSL")
    done = subprocess.run([str(REPO / "scripts" / "ml-stack-workspace"), "--help"], capture_output=True,
                          text=True, timeout=30, check=False)
    assert done.returncode == 2 and "WSL" in done.stderr
