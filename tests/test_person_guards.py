"""What consumes the person record (the pre-push hook), what must not write it, and the hard rules of the Bash and edit guards."""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys

import pytest
from person_support import (DEV, HOOKS, PROPOSAL_TEXT, ROOT, SESSION, assistant, environment, git, repository, rows,
                            say)

from ml_stack.workspace import person_store
from ml_stack.workspace.graphlog import GraphLog

PROPOSAL = assistant(PROPOSAL_TEXT)
ZERO = "0" * 40


@pytest.fixture
def world(tmp_path):
    return tmp_path, tmp_path / "state", repository(tmp_path / "repo")


def push(repo, state, *refs, sha="", session=SESSION, **env):
    sha = sha or git(repo, "rev-parse", "HEAD")
    source = repo / ".git" / "push-input"
    source.write_text("".join(f"refs/heads/{r} {sha} refs/heads/{r} {ZERO}\n" for r in refs))
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\nexec sh " + shlex.quote((HOOKS / "pre-push").as_posix()) + ' "$@"\n')
    hook.chmod(0o755)
    return subprocess.run(["git", "hook", "run", "--to-stdin=" + str(source), "pre-push", "--", "origin",
                           "https://example.invalid/x.git"], cwd=repo, text=True, capture_output=True,
                          env=environment(state, PYTHON=sys.executable, ML_STACK_SESSION_ID=session, **env))


def test_the_development_push_consumes_the_authorization_once(world):
    tmp, state, repo = world
    say(tmp, state, repo, "yes", before=(PROPOSAL,))
    first = push(repo, state, DEV, CLAUDECODE="1")
    assert first.returncode == 0, first.stderr
    assert "authorized by the person's statement" in first.stderr
    second = push(repo, state, DEV, CLAUDECODE="1")
    assert second.returncode == 0 and "authorized by the person" not in second.stderr
    assert [r["state"] for r in rows(state) if r["type"] == "transition"] == ["used"]


def test_the_existing_policy_still_lets_an_agent_push_the_development_branch_unauthorized(world):
    _, state, repo = world
    done = push(repo, state, DEV, CLAUDECODE="1")
    assert done.returncode == 0 and done.stderr == "" and rows(state) == []


def test_another_sessions_authorization_is_not_consumed(world):
    tmp, state, repo = world
    say(tmp, state, repo, "yes", before=(PROPOSAL,))
    done = push(repo, state, DEV, session="someone-else", CLAUDECODE="1")
    assert done.returncode == 0 and "authorized by the person" not in done.stderr


@pytest.mark.parametrize("refs,sha", [(("main",), ""), ((DEV, "main"), ""), ((DEV,), ZERO), (("topic",), "")])
def test_a_live_authorization_never_opens_main_a_deletion_or_another_branch(world, refs, sha):
    tmp, state, repo = world
    say(tmp, state, repo, "yes", before=(PROPOSAL,))
    done = push(repo, state, *refs, sha=sha, CLAUDECODE="1")
    assert done.returncode != 0 and "refused" in done.stderr
    assert [r for r in rows(state) if r["type"] == "transition"] == []


def test_a_person_pushing_from_their_own_terminal_meets_nothing(world):
    tmp, state, repo = world
    say(tmp, state, repo, "yes", before=(PROPOSAL,))
    done = push(repo, state, "main", CLAUDECODE="")
    assert done.returncode == 0 and done.stderr == ""
    assert [r for r in rows(state) if r["type"] == "transition"] == []


def guard(name, event, **env):
    return subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(event), text=True,
                          capture_output=True, timeout=30, check=False, env={**environment(ROOT / ".none"), **env})


def bash(command, **env):
    return guard("claude-bash-guard", {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(ROOT)},
                 **env)


HARD = ["git push --force origin 0.9dev", "git push origin main", "git push origin --tags", "git push origin --delete x",
        "ml-stack-workspace mint someone --role human", "cat ~/.ml-stack/person/statements.log",
        "python3 -c 'from ml_stack.workspace.person_record import mark_used'"]


@pytest.mark.parametrize("command", HARD)
@pytest.mark.parametrize("marker", [{"CLAUDECODE": "1"}, {"ML_STACK_AGENT": "x"}, {"CODEX_THREAD_ID": "t"}])
def test_a_hard_rule_holds_when_the_guard_is_switched_off_under_an_agent(command, marker):
    done = bash(command, MLSTACK_GUARD="off", **marker)
    assert done.returncode == 2, command
    assert "No switch opens this rule" in done.stderr


@pytest.mark.parametrize("command", HARD)
def test_the_hard_rules_also_hold_with_the_guard_on(command):
    assert bash(command).returncode == 2


def test_the_switch_still_turns_off_a_soft_rule_and_a_person_keeps_it():
    assert bash("git add -A").returncode == 2
    assert bash("git add -A", MLSTACK_GUARD="off", CLAUDECODE="1").returncode == 0
    assert bash("git push --force origin x", MLSTACK_GUARD="off").returncode == 0


def edit(path, **env):
    return guard("claude-edit-guard", {"tool_name": "Write", "tool_input": {"file_path": path, "content": "x\n"},
                                       "cwd": str(ROOT)}, **env)


@pytest.mark.parametrize("path", ["/home/u/.ml-stack/person/statements.log", "/home/u/.ml-stack/person/statements.log.head.key",
                                  "/home/u/.ml-stack/person"])
def test_nothing_writes_the_person_store_even_with_the_guard_off(path):
    for env in ({}, {"MLSTACK_GUARD": "off", "CLAUDECODE": "1"}):
        done = edit(path, **env)
        assert done.returncode == 2 and "no switch opens this rule" in done.stderr


def test_the_edit_guard_switch_still_works_for_a_soft_rule(tmp_path):
    assert edit(str(tmp_path / "x.txt"), MLSTACK_GUARD="off", CLAUDECODE="1").returncode == 0


def importers(module):
    """The src/ml_stack files that import a module called ``module``."""
    found = set()
    for path in sorted((ROOT / "src" / "ml_stack").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *((node.module or "") + "." + a.name for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            if any(n.split(".")[-1] == module for n in names):
                found.add(path.relative_to(ROOT / "src" / "ml_stack").as_posix())
    return sorted(found)


def test_only_the_hook_logic_and_the_consumer_import_the_writers():
    assert importers("person_record") == ["workspace/person_auth.py", "workspace/person_hook.py"]


def test_no_agent_reachable_module_imports_the_hook_logic_or_the_consumer():
    assert importers("person_hook") == []
    assert importers("person_auth") == []
    assert importers("person_transcript") == ["workspace/person_hook.py"]


def test_the_scripts_that_reach_the_person_record_are_the_hooks_only():
    reaching = sorted(p.name for p in HOOKS.iterdir()
                      if p.is_file() and re.search(r"person_(hook|auth|record)", p.read_text(errors="replace")))
    assert reaching == ["claude-bash-guard", "person-consume", "workspace_hook.py"]


def test_no_console_script_or_module_main_writes_the_record():
    project = (ROOT / "pyproject.toml").read_text()
    assert not re.search(r"person_(record|hook|auth)", project)


def test_the_board_append_refuses_a_person_attestation_from_every_caller(tmp_path):
    log = GraphLog(tmp_path, "boards")
    for body in ({"kind": "person-attestation"}, {"kind": "msg", "type": "person-attestation"},
                 {"kind": "msg", "attestation": "person-attestation"}):
        with pytest.raises(PermissionError):
            log.append(body)


def test_the_board_view_shows_the_record_as_attested_by_the_hook_and_never_as_a_person(world, monkeypatch):
    tmp, state, repo = world
    say(tmp, state, repo, "yes", before=(PROPOSAL,))
    monkeypatch.setenv("ML_STACK_HOME", str(state))
    from ml_stack.workspace import person_view
    shown = person_view.listing()
    assert {r["kind"] for r in shown} == {person_store.ATTESTATION}
    assert all(r["text"] == f"attested by the harness hook for session {SESSION}" for r in shown)
    assert all("author" not in r and "role" not in r for r in shown)
    tampered = state / "person" / "statements.log"
    tampered.write_text(tampered.read_text().replace('"uses": 1', '"uses": 5'))
    assert person_view.listing()[0]["ok"] is False


def test_no_configured_hook_can_pre_answer_an_ask_user_question():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    for matcher in settings["hooks"].get("PreToolUse", []):
        if re.fullmatch(matcher.get("matcher", ".*"), "AskUserQuestion"):
            for hook in matcher["hooks"]:
                script = hook["command"].replace("$CLAUDE_PROJECT_DIR", str(ROOT))
                done = subprocess.run([sys.executable, script], input=json.dumps(
                    {"hook_event_name": "PreToolUse", "tool_name": "AskUserQuestion", "tool_input": {"questions": []}}),
                    text=True, capture_output=True, timeout=30, check=False)
                assert "updatedInput" not in done.stdout
    for path in HOOKS.iterdir():
        if path.is_file():
            assert "updatedInput" not in path.read_text(errors="replace"), path.name


def test_the_person_hooks_are_wired_for_prompts_answers_and_session_end():
    hooks = json.loads((ROOT / ".claude" / "settings.json").read_text())["hooks"]
    assert "claude-user-prompt" in json.dumps(hooks["UserPromptSubmit"])
    assert hooks["PostToolUse"][0]["matcher"] == "AskUserQuestion"
    assert "claude-user-prompt" in json.dumps(hooks["SessionEnd"])
