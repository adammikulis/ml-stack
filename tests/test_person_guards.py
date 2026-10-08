"""What consumes the person record (the pre-push hook and the Bash guard), what must not write it, and the hard rules of the guards."""

from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys

import pytest
from person_support import (
    HOOKS,
    ROOT,
    SESSION,
    ZERO,
    approve,
    commit,
    environment,
    git,
    repository,
    rows,
    say,
    tip,
)

from ml_stack.workspace import person_ancestry, person_store
from ml_stack.workspace.graphlog import GraphLog

AGENT = {"CLAUDECODE": "1"}


@pytest.fixture
def world(tmp_path):
    repo = repository(tmp_path / "repo")
    git(repo, "checkout", "-q", "-b", "0.9dev")
    return tmp_path, tmp_path / "state", repo


def line(local, remote, ref="main", target_ref=""):
    return f"refs/heads/{ref} {local} refs/heads/{target_ref or ref} {remote}\n"


def release(repo):
    return line(tip(repo), tip(repo, "origin/main"))


def push(repo, state, text, **env):
    source = repo / ".git" / "push-input"
    source.write_text(text)
    hook = repo / ".git" / "hooks" / "pre-push"
    hook.write_text("#!/bin/sh\nexec sh " + shlex.quote((HOOKS / "pre-push").as_posix()) + ' "$@"\n')
    hook.chmod(0o755)
    return subprocess.run(["git", "hook", "run", "--to-stdin=" + str(source), "pre-push", "--", "origin",
                           "https://example.invalid/x.git"], cwd=repo, text=True, capture_output=True,
                          env=environment(state, PYTHON=sys.executable, **env))


def test_the_approved_release_goes_through_once_and_is_recorded(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    first = push(repo, state, release(repo), **AGENT)
    assert first.returncode == 0, first.stderr
    assert "main released with the person's approval" in first.stderr
    second = push(repo, state, release(repo), **AGENT)
    assert second.returncode != 0 and "no live release-main authorization" in second.stderr
    assert [r["state"] for r in rows(state) if r["type"] == "transition"] == ["used"]


def test_a_push_of_main_without_an_approval_is_refused_and_says_how_to_ask(world):
    _tmp, state, repo = world
    done = push(repo, state, release(repo), **AGENT)
    assert done.returncode != 0 and "refused:" in done.stderr and "person-consume propose" in done.stderr


def test_the_opener_variable_no_longer_opens_main_for_an_agent(world):
    _, state, repo = world
    assert push(repo, state, release(repo), ML_STACK_PUSH_MAIN="yes", **AGENT).returncode != 0


def test_the_development_branch_needs_no_approval_and_consumes_nothing(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    done = push(repo, state, line(tip(repo, "HEAD"), ZERO, "0.9dev"), **AGENT)
    assert done.returncode == 0 and done.stderr == ""
    assert [r for r in rows(state) if r["type"] == "transition"] == []


def refused(repo, state, text, **env):
    done = push(repo, state, text, **{**AGENT, **env})
    return done.returncode != 0


def test_an_approval_covers_exactly_one_fast_forward_to_the_approved_commit(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    approved, remote = tip(repo), tip(repo, "origin/main")
    git(repo, "checkout", "-q", "main")
    newer = commit(repo, "later.txt")
    git(repo, "checkout", "-q", "0.9dev")
    git(repo, "branch", "-q", "side", remote)
    git(repo, "checkout", "-q", "side")
    divergent = commit(repo, "side.txt")
    git(repo, "checkout", "-q", "0.9dev")
    refs_ok = line(approved, remote)
    bad = {
        "a commit made after the approval": line(newer, remote),
        "not a fast-forward": line(approved, divergent),
        "HEAD:main resolving to the base": line(remote, remote, "HEAD", "main"),
        "a deletion": line(ZERO, remote),
        "a new main": line(approved, ZERO),
        "a tag": line(approved, ZERO, "v1").replace("refs/heads/v1", "refs/tags/v1"),
        "two refs": refs_ok + line(approved, ZERO, "0.9dev"),
        "another branch name": line(approved, remote, "main", "release"),
    }
    for label, text in bad.items():
        assert refused(repo, state, text), label
    assert [r for r in rows(state) if r["type"] == "transition"] == []
    assert push(repo, state, refs_ok, **AGENT).returncode == 0


def test_an_approval_in_one_repository_does_not_cover_another_with_the_same_names(world):
    tmp, state, repo = world
    approve(tmp, state, repo)
    other = repository(tmp / "second")
    assert refused(other, state, release(other))
    assert not refused(repo, state, release(repo))


def test_a_person_pushing_from_their_own_terminal_meets_nothing(world):
    _, state, repo = world
    if person_ancestry.under_harness():
        pytest.skip("this run is itself under an agent harness")
    done = push(repo, state, release(repo), CLAUDECODE="")
    assert done.returncode == 0 and done.stderr == ""


def test_dropping_the_marker_variable_does_not_make_an_agent_a_person(world):
    if person_ancestry.under_harness():
        pytest.skip("this run is itself under an agent harness")
    tmp, state, repo = world
    say(tmp, state, repo, "hello", prompt_id="bind")
    assert refused(repo, state, release(repo), CLAUDECODE="")
    assert refused(repo, state, line(tip(repo), ZERO, "v1").replace("refs/heads/v1", "refs/tags/v1"), CLAUDECODE="")


def guard(name, event, state, **env):
    return subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(event), text=True,
                          capture_output=True, timeout=60, check=False, env=environment(state, **env))


def bash(command, state, cwd=ROOT, **env):
    return guard("claude-bash-guard", {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd)},
                 state, **env)


FORCES = ["git push --force origin main", "git push origin main --force-with-lease", "git push -f origin 0.9dev",
          "git push origin --tags", "git push origin --delete work", "git push --all origin", "git push --mirror"]
PREFIXES = ["", "env -u CLAUDECODE ", "env -i ", "env FOO=1 ", "unset CLAUDECODE; ", "sh -c 'X' ", "bash -c \"X\" ",
            "sudo ", "(", "git -C /tmp/x "]
OTHERS = ["ml-stack-workspace mint someone --role human", "cat ~/.ml-stack/person/statements.log",
          "python3 -c 'from ml_stack.workspace.person_record import mark_used'",
          "python3 -c 'from ml_stack.workspace import person_store'", "python3 -m ml_stack.workspace.person_hook",
          f"python3 {HOOKS}/claude-user-prompt < /tmp/event.json", f"{HOOKS}/person-consume consume origin",
          "CLAUDE_CONFIG_DIR=/tmp/x python3 -c 1"]


def wrap(prefix, command):
    if prefix.endswith("-C /tmp/x "):
        return command.replace("git ", "git -C /tmp/x ", 1)
    if "X" in prefix:
        return prefix.replace("X", command)
    return prefix + command


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("command", FORCES)
def test_force_tag_delete_and_mirror_pushes_are_refused_through_every_prefix(tmp_path, prefix, command):
    for env in ({}, {"MLSTACK_GUARD": "off"}, {"MLSTACK_GUARD": "off", **AGENT}):
        done = bash(wrap(prefix, command), tmp_path / "state", **env)
        assert done.returncode == 2, (prefix, command, env)
        assert "No switch opens this rule" in done.stderr


@pytest.mark.parametrize("prefix", PREFIXES)
def test_a_push_of_main_without_an_approval_is_refused_through_every_prefix(tmp_path, prefix):
    for env in ({}, {"MLSTACK_GUARD": "off"}):
        done = bash(wrap(prefix, "git push origin main"), tmp_path / "state", **env)
        assert done.returncode == 2 and "person-consume propose" in done.stderr


@pytest.mark.parametrize("command", OTHERS)
def test_the_person_record_minting_and_hook_rules_hold_with_the_guard_switched_off(tmp_path, command):
    for env in ({}, {"MLSTACK_GUARD": "off"}, {"MLSTACK_GUARD": "off", **AGENT}):
        assert bash(command, tmp_path / "state", **env).returncode == 2, (command, env)


def test_the_guard_lets_an_agent_ask_the_approval_question_and_write_about_the_record(tmp_path):
    state = tmp_path / "state"
    assert bash(f"{HOOKS}/person-consume propose", state).returncode == 0
    for text in ('git commit -m "docs: mention ~/.ml-stack/person/statements.log and person_record"',
                 "git add scripts/hooks/person-consume scripts/hooks/claude-user-prompt",
                 "scripts/test all tests/test_person_hooks.py", "grep -rn 'git push origin main' docs/"):
        assert bash(text, state).returncode == 0, text


def test_a_push_of_main_passes_the_guard_only_while_this_session_holds_an_approval(world):
    tmp, state, repo = world
    assert bash("git push origin main", state, cwd=repo).returncode == 2
    approve(tmp, state, repo)
    assert bash("git push origin main", state, cwd=repo).returncode == 0
    assert bash("git push --force origin main", state, cwd=repo).returncode == 2
    assert bash("git push origin --tags", state, cwd=repo).returncode == 2
    assert bash("git push origin main", state, cwd=tmp).returncode == 2


def test_the_switch_still_turns_off_a_soft_rule_and_nothing_else(tmp_path):
    state = tmp_path / "state"
    assert bash("git add -A", state).returncode == 2
    assert bash("git add -A", state, MLSTACK_GUARD="off").returncode == 0
    assert bash("git add -A", state, MLSTACK_GUARD="off", **AGENT).returncode == 0


def edit(path, state, **env):
    return guard("claude-edit-guard", {"tool_name": "Write", "tool_input": {"file_path": str(path), "content": "x\n"},
                                       "cwd": str(ROOT)}, state, **env)


def test_nothing_writes_the_person_store_through_a_path_a_link_or_a_moved_root(tmp_path):
    state = tmp_path / "state"
    (state / "person").mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(state / "person")
    targets = [state / "person" / "statements.log", link / "x", "/home/u/.ml-stack/person/statements.log.head.key"]
    for path in targets:
        for env in ({}, {"MLSTACK_GUARD": "off", **AGENT}):
            done = edit(path, state, **env)
            assert done.returncode == 2 and "no switch opens this rule" in done.stderr, (path, env)


def test_an_ordinary_file_and_the_soft_switch_still_work(tmp_path):
    assert edit(tmp_path / "x.txt", tmp_path / "state").returncode == 0
    assert edit(tmp_path / "x.txt", tmp_path / "state", MLSTACK_GUARD="off", **AGENT).returncode == 0


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


def test_no_agent_reachable_module_imports_the_hook_logic_or_the_consumers():
    assert importers("person_hook") == []
    assert importers("person_release") == ["workspace/person_hook.py"]
    assert importers("person_auth") == ["workspace/person_release.py"]
    assert importers("person_transcript") == ["workspace/person_hook.py"]


def test_the_scripts_that_reach_the_person_record_are_the_hooks_only():
    reaching = sorted(p.name for p in HOOKS.iterdir()
                      if p.is_file() and re.search(r"person_(hook|auth|record|release)", p.read_text(errors="replace")))
    assert reaching == ["person-consume", "workspace_hook.py"]


def test_no_console_script_names_the_record_writers():
    assert not re.search(r"person_(record|hook|auth|release)", (ROOT / "pyproject.toml").read_text())


def test_the_board_append_refuses_a_person_attestation_from_every_caller(tmp_path):
    log = GraphLog(tmp_path, "boards")
    for body in ({"kind": "person-attestation"}, {"kind": "msg", "type": "person-attestation"},
                 {"kind": "msg", "attestation": "person-attestation"}):
        with pytest.raises(PermissionError):
            log.append(body)


def test_the_board_view_shows_the_record_as_attested_by_the_hook_and_never_as_a_person(world, monkeypatch):
    tmp, state, repo = world
    approve(tmp, state, repo)
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
