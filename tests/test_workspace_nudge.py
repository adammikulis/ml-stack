"""The inbox nudge and the Claude Code hooks built on it, against a real workspace and real hook processes."""

from __future__ import annotations

import json
import time

import pytest
from workspace_kit import Kit, clean_env, cli

from ml_stack.workspace import nudge
from ml_stack.workspace.remote_protocol import METHODS

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


def test_the_plain_line_counts_kinds_names_senders_and_gives_the_age_without_a_body(kit):
    ws, t = kit.ws, kit.t
    ws.send(t["alice"], "bob", "question", "SECRET-BODY which port?", label="local-qwen")
    ws.send(t["alice"], "bob", "question", "SECRET-BODY and the lease?")
    ws.send(t["carol"], "bob", "task", "SECRET-BODY build it")
    for step in range(3):
        ws.send(t["carol"], "bob", "status", f"SECRET-BODY progress {step}")
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
    assert shape == {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text},
                     "systemMessage": text}
    assert text.startswith("workspace: 1 waiting for you")
    assert hook(kit, "prompt").stdout == ""


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
    assert "SECRET-BODY ready?" in verdict["reason"] and verdict["systemMessage"] == verdict["reason"]
    assert hook(kit, "stop", "{}").stdout == ""
    ws.send(t["carol"], "bob", "handoff", "SECRET-BODY yours")
    again = json.loads(hook(kit, "stop", "{}").stdout)
    assert again["decision"] == "block" and "2 urgent" not in again["reason"]
    assert "SECRET-BODY yours" in again["reason"] and "SECRET-BODY ready?" not in again["reason"]


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


def test_the_installer_is_for_a_person(tmp_path):
    out = cli(tmp_path / "ws", "", "install-hooks", "--settings", str(tmp_path / "s.json"),
              env_extra={"CLAUDECODE": "1"})
    assert out.returncode == 3 and not (tmp_path / "s.json").exists()


class _Board:
    """A board that answers each operation from a real workspace."""

    host = "http://127.0.0.1:8770"
    project_id = "a" * 32
    cluster_key = ""

    def __init__(self, kit):
        self.kit = kit
        self.calls = []

    def token(self, **kwargs):
        return "capability"

    def call(self, operation, token, *args, **kwargs):
        self.calls.append(operation)
        if operation == "whoami":
            return {"id": "bob", "role": "agent", "can": ["read"], "project": {"key": self.project_id}}
        assert operation in METHODS
        return getattr(self.kit.ws, operation)(self.kit.t["bob"], *args, **kwargs)


def test_prompt_hook_on_a_board_injects_the_waiting_line(kit, monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from ml_stack.workspace import cli as ws_cli, project_connection as connection
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(nudge.tempfile, "gettempdir", lambda: str(tmp_path))
    root = tmp_path / "project"
    root.mkdir()
    board = _Board(kit)
    connection.bind(board, root, "bob", "default")
    monkeypatch.chdir(root)
    monkeypatch.setattr(connection, "RemoteWorkspace", lambda *a, **k: board)
    kit.ws.send(kit.t["alice"], "bob", "question", "SECRET-BODY which port?")
    args = SimpleNamespace(cmd="nudge", hook="prompt", agent="", token_file="", label="", json=False)
    assert ws_cli._nudging(args) == 0
    shape = json.loads(capsys.readouterr().out)
    text = shape["hookSpecificOutput"]["additionalContext"]
    assert shape["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert text.startswith("workspace: 1 waiting for you (1 question; from alice;")
    assert "SECRET-BODY which port?" in text and shape["systemMessage"] == text
    assert "waiting_summary" in board.calls


def test_the_stamp_file_name_needs_no_posix_uid(monkeypatch):
    import os

    from ml_stack.workspace import nudge

    monkeypatch.delattr(os, "getuid", raising=False)
    assert nudge._stamp("ml-stack-nudge").name.startswith("ml-stack-nudge.")
