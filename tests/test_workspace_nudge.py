"""The inbox nudge and the Claude Code hooks built on it, against a real node and real hook processes."""

from __future__ import annotations

import json
import os
import re

import pytest
from workspace_kit import cli as old_cli

from ml_stack.workspace import board_cli, nudge

pytest_plugins = ["node_kit"]
LONG_AGO = 3 * 3600 + 12 * 60 + 20


@pytest.fixture
def kit(workspace_node, monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(nudge.tempfile, "gettempdir", lambda: str(state))
    workspace_node.alice, workspace_node.bob, workspace_node.carol = (workspace_node.member(n) for n in ("alice", "bob", "carol"))
    return workspace_node


def say(kit, who, to, kind, text):
    return kit.session(who).post(to.name, kind, text)


def waiting(kit, aged=True):
    """What waits for bob, as the hook sees it; ``aged`` moves the clock on so the messages are old."""
    summary = board_cli.waiting(kit.session(kit.bob))
    return nudge.Waiting.of({**summary, "now": summary["now"] + (LONG_AGO if aged else 0)})


def hook(kit, event, stdin=""):
    return nudge.output(event, waiting(kit), stdin)


def test_the_plain_line_counts_kinds_names_senders_and_gives_the_age_without_a_body(kit):
    for text in ("which port?", "and the lease?"):
        say(kit, kit.alice, kit.bob, "question", f"SECRET-BODY {text}")
    say(kit, kit.carol, kit.bob, "task", "SECRET-BODY build it")
    for step in range(3):
        say(kit, kit.carol, kit.bob, "status", f"SECRET-BODY progress {step}")
    out = waiting(kit).line()
    assert out.startswith(f"workspace: 6 waiting for you (2 questions, 1 task, 3 status; from {kit.alice.name}, {kit.carol.name}; "
                          "oldest 3h12m). A direct question or task is waiting on you: run ml-stack-workspace inbox now and answer it")
    assert "SECRET" not in out
    cli_line = kit.cli("nudge", who=kit.bob).stdout
    assert re.match(r"workspace: 6 waiting for you \(2 questions, 1 task, 3 status; from .*; oldest \d+s\)\. A direct", cli_line)
    assert "SECRET" not in cli_line


def test_routine_kinds_keep_the_short_form(kit):
    say(kit, kit.alice, kit.bob, "status", "SECRET-BODY on it")
    say(kit, kit.alice, kit.bob, "note", "SECRET-BODY fyi")
    assert waiting(kit).line() == f"workspace: 2 waiting for you (1 note, 1 status; from {kit.alice.name}; oldest 3h12m); run inbox"


def test_prompt_hook_injects_the_line_as_context_and_stays_silent_when_empty(kit):
    assert hook(kit, "prompt") == ""
    say(kit, kit.alice, kit.bob, "status", "x")
    shape = json.loads(hook(kit, "prompt"))
    text = shape["hookSpecificOutput"]["additionalContext"]
    assert shape == {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text}, "systemMessage": text}
    assert text.startswith("workspace: 1 waiting for you")
    assert hook(kit, "prompt") == ""


def test_the_cli_hook_prints_the_same_json_as_the_library(kit, tmp_path):
    say(kit, kit.alice, kit.bob, "question", "SECRET-BODY ready?")
    done = kit.cli("nudge", "--hook", "prompt", who=kit.bob, env={"TMPDIR": str(tmp_path / "state")})
    shape = json.loads(done.stdout)
    assert shape["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit" and "SECRET-BODY ready?" in shape["systemMessage"]


def test_post_hook_is_rate_limited_to_one_check_in_twenty_seconds(kit):
    say(kit, kit.alice, kit.bob, "status", "x")
    assert json.loads(hook(kit, "post"))["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert hook(kit, "post") == ""


def test_stop_hook_blocks_once_per_set_of_unread_urgent_messages(kit):
    assert hook(kit, "stop", "{}") == ""
    say(kit, kit.alice, kit.bob, "question", "SECRET-BODY ready?")
    verdict = json.loads(hook(kit, "stop", "{}"))
    assert verdict["decision"] == "block" and "1 question" in verdict["reason"]
    assert "SECRET-BODY ready?" in verdict["reason"] and verdict["systemMessage"] == verdict["reason"]
    assert hook(kit, "stop", "{}") == ""
    say(kit, kit.carol, kit.bob, "handoff", "SECRET-BODY yours")
    again = json.loads(hook(kit, "stop", "{}"))
    assert again["decision"] == "block" and "2 urgent" not in again["reason"]
    assert "SECRET-BODY yours" in again["reason"] and "SECRET-BODY ready?" not in again["reason"]


def test_stop_hook_allows_when_continuing_or_when_only_routine_kinds_wait(kit):
    say(kit, kit.alice, kit.bob, "status", "x")
    say(kit, kit.alice, kit.bob, "note", "x")
    assert hook(kit, "stop", "{}") == ""
    say(kit, kit.alice, kit.bob, "question", "x")
    assert hook(kit, "stop", '{"stop_hook_active": true}') == ""
    assert json.loads(hook(kit, "stop", "{}"))["decision"] == "block"


def test_stop_hook_allows_a_question_younger_than_two_minutes(kit):
    say(kit, kit.alice, kit.bob, "question", "x")
    assert nudge.output("stop", waiting(kit, aged=False), "{}") == ""


def test_hooks_print_nothing_and_exit_zero_when_the_node_cannot_be_reached(kit):
    gone = {"ML_STACK_NODE_BIN": "/nonexistent", "ML_STACK_NODE_DIR": "/tmp/ml-nowhere-node"}
    out = kit.cli("nudge", "--hook", "stop", stdin="{}", env=gone, who=kit.bob)
    assert out.returncode == 0 and out.stdout == "" and out.stderr == ""
    plain = kit.cli("nudge", env=gone, who=kit.bob)
    assert plain.returncode == 3


def test_the_installer_is_for_a_person(tmp_path):
    out = old_cli(tmp_path / "ws", "", "install-hooks", "--settings", str(tmp_path / "s.json"), env_extra={"CLAUDECODE": "1"})
    assert out.returncode == 3 and not (tmp_path / "s.json").exists()


def test_the_stamp_file_name_needs_no_posix_uid(monkeypatch):
    monkeypatch.delattr(os, "getuid", raising=False)
    assert nudge._stamp("ml-stack-nudge").name.startswith("ml-stack-nudge.")
