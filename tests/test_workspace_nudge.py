"""The inbox nudge and the Claude Code hooks built on it, against a real workspace and real hook processes."""

from __future__ import annotations

import json
import re
import time

import pytest
from workspace_kit import Kit, clean_env, cli

from ml_stack.sentinel import human
from ml_stack.workspace import onboard

LONG_AGO = 3 * 3600 + 12 * 60 + 20


@pytest.fixture
def kit(monkeypatch, tmp_path):
    k = Kit(clean_env(monkeypatch, tmp_path), clock=lambda: time.time() - LONG_AGO)
    k.limits(sends_per_window=1000)
    k.t = {n: k.agent(n) for n in ("alice", "bob", "carol")}
    return k


def hook(kit, event, stdin="", tmp="state"):
    tmpdir = kit.base.parent / tmp
    tmpdir.mkdir(exist_ok=True)
    return cli(kit.base, kit.t["bob"], "nudge", "--hook", event, env_extra={"TMPDIR": str(tmpdir)},
               input=stdin)


def test_the_line_counts_kinds_names_senders_and_gives_the_age_without_a_body(kit):
    ws, t = kit.ws, kit.t
    ws.send(t["alice"], "bob", "question", "SECRET-BODY which port?", label="local-qwen")
    ws.send(t["alice"], "bob", "question", "SECRET-BODY and the lease?")
    ws.send(t["carol"], "bob", "task", "SECRET-BODY build it")
    for _ in range(3):
        ws.send(t["carol"], "bob", "status", "SECRET-BODY progress")
    out = cli(kit.base, t["bob"], "nudge").stdout
    assert out.startswith("workspace: 6 waiting for you (2 questions, 1 task, 3 status; "
                          "from alice/local-qwen, alice, carol; oldest 3h12m). A direct question or "
                          "task is waiting on you: run ml-stack-workspace inbox now and answer it")
    assert "SECRET" not in out


def test_routine_kinds_keep_the_short_form(kit):
    kit.ws.send(kit.t["alice"], "bob", "status", "SECRET-BODY on it")
    kit.ws.send(kit.t["alice"], "bob", "note", "SECRET-BODY fyi")
    out = cli(kit.base, kit.t["bob"], "nudge").stdout
    assert out == "workspace: 2 waiting for you (1 note, 1 status; from alice; oldest 3h12m); run inbox\n"


def test_prompt_hook_injects_the_line_as_context_and_stays_silent_when_empty(kit):
    assert hook(kit, "prompt").stdout == ""
    kit.ws.send(kit.t["alice"], "bob", "status", "x")
    shape = json.loads(hook(kit, "prompt").stdout)
    text = shape["hookSpecificOutput"]["additionalContext"]
    assert shape == {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text}}
    assert text.startswith("workspace: 1 waiting for you")


def test_post_hook_is_rate_limited_to_one_check_in_twenty_seconds(kit):
    kit.ws.send(kit.t["alice"], "bob", "status", "x")
    first = hook(kit, "post")
    assert json.loads(first.stdout)["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert hook(kit, "post").stdout == ""


def test_stop_hook_blocks_once_per_set_of_unread_urgent_messages(kit):
    ws, t = kit.ws, kit.t
    assert hook(kit, "stop", "{}").stdout == ""
    ws.send(t["alice"], "bob", "question", "SECRET-BODY ready?")
    first = hook(kit, "stop", "{}")
    verdict = json.loads(first.stdout)
    assert verdict["decision"] == "block" and "1 question" in verdict["reason"]
    assert "SECRET" not in first.stdout
    assert hook(kit, "stop", "{}").stdout == ""
    ws.send(t["carol"], "bob", "handoff", "SECRET-BODY yours")
    again = json.loads(hook(kit, "stop", "{}").stdout)
    assert again["decision"] == "block" and "1 handoff" in again["reason"]


def test_stop_hook_allows_when_continuing_or_when_only_routine_kinds_wait(kit):
    ws, t = kit.ws, kit.t
    ws.send(t["alice"], "bob", "status", "x")
    ws.send(t["alice"], "bob", "note", "x")
    assert hook(kit, "stop", "{}").stdout == ""
    ws.send(t["alice"], "bob", "question", "x")
    assert hook(kit, "stop", '{"stop_hook_active": true}').stdout == ""
    assert json.loads(hook(kit, "stop", "{}").stdout)["decision"] == "block"


def test_stop_hook_allows_a_question_younger_than_two_minutes(monkeypatch, tmp_path):
    fresh = Kit(clean_env(monkeypatch, tmp_path))
    bob, dan = fresh.agent("bob"), fresh.agent("dan")
    fresh.ws.send(dan, "bob", "question", "x")
    done = cli(fresh.base, bob, "nudge", "--hook", "stop", env_extra={"TMPDIR": str(tmp_path)}, input="{}")
    assert done.returncode == 0 and done.stdout == ""


def test_hooks_print_nothing_and_exit_zero_when_the_workspace_is_unreachable(tmp_path):
    out = cli(tmp_path / "nowhere", "", "nudge", "--hook", "stop", input="{}")
    assert out.returncode == 0 and out.stdout == "" and out.stderr == ""


def test_the_installer_adds_the_three_hooks_idempotently_and_keeps_other_hooks(monkeypatch, tmp_path):
    clean_env(monkeypatch, tmp_path)
    real = human.require_person
    monkeypatch.setattr(human, "require_person", lambda a, terminal=None, env=None: real(a, (True, True), env))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": "x", "hooks": {
        "PostToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "other-tool"}]},
            {"matcher": "*", "hooks": [{"type": "command", "command": "sh ~/.ml-stack/hooks/claude-nudge.sh"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "keep-me"}]}]}}))
    assert onboard.install_hooks(path, "claude-code") == ["PostToolUse", "Stop", "UserPromptSubmit"]
    first = path.read_text()
    onboard.install_hooks(path, "claude-code")
    assert path.read_text() == first
    data = json.loads(first)
    assert data["model"] == "x"
    commands = {e: [h["command"] for g in groups for h in g["hooks"]] for e, groups in data["hooks"].items()}
    assert commands["PostToolUse"] == ["other-tool", "ml-stack-workspace nudge --agent claude-code --hook post"]
    assert commands["Stop"] == ["keep-me", "ml-stack-workspace nudge --agent claude-code --hook stop"]
    assert commands["UserPromptSubmit"] == ["ml-stack-workspace nudge --agent claude-code --hook prompt"]
    assert re.search(r'"matcher": "\*"', first)


def test_the_installer_is_for_a_person(tmp_path):
    out = cli(tmp_path / "ws", "", "install-hooks", "--settings", str(tmp_path / "s.json"),
              env_extra={"CLAUDECODE": "1"})
    assert out.returncode == 3 and not (tmp_path / "s.json").exists()
