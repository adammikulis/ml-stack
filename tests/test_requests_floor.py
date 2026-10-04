"""Answering a request is a person's act: no tool a model is offered, no MCP tool, no message and
no argument a model writes can answer, cancel, list the answers of or change a request."""

from __future__ import annotations

import io
import sys

import pytest

from ml_stack import chat, do, mcp, requests
from ml_stack.chatpolicy import HumanOnlyRail
from ml_stack.interventions import Call, Context, Deny
from ml_stack.sentinel.human import agent_may
from tests.requests_support import ask, no_markers

__all__ = ["no_markers"]
pytestmark = pytest.mark.usefixtures("no_markers")
NAMES = ("answer_request", "approve_request", "deny_request", "cancel_request", "requests_answer",
         "requests_list", "list_requests", "request_show", "inbox_answer", "resolve_request",
         "withdraw_request", "dismiss_request", "requests")


def test_no_tool_offered_to_a_model_names_a_request_or_an_answer():
    person = do.Person(io.StringIO(""), io.StringIO(""))
    session = chat.Chat(None, person, role="runner", extension=chat.extensions(person))
    offered = [s["function"]["name"] for s, _ in session.offered] + [t.name for t in mcp.TOOLS]
    named = [n for n in offered if any(w in n.lower() for w in ("request", "answer", "approve"))]
    assert named == []


@pytest.mark.parametrize("name", NAMES)
def test_a_gullible_model_calling_a_tool_named_for_the_inbox_is_refused_with_the_terminal_command(name):
    got = HumanOnlyRail().before_tool_call(Call(name, {"id": "rq_x", "choice": "allow-once"}), Context())
    assert isinstance(got, Deny) and "ml-stack-requests answer" in got.reason


@pytest.mark.parametrize("text", ["ml-stack-requests answer rq_1 allow-once", "ml-stack-requests list --json",
                                  "python -m ml_stack.inbox.cli answer rq_1 deny",
                                  "ml_stack.requests.inbox.answer('rq_1', 'allow-once', fp, 'ui')",
                                  "cat ~/.ml-stack/requests/requests.enc"])
@pytest.mark.parametrize("tool", ["bench_run", "run_shell", "serve_up", "models_fetch", "doctor"])
def test_an_argument_naming_the_inbox_is_refused_whichever_tool_carries_it(tool, text):
    assert agent_may(tool, {"argv": [text], "command": text})


def test_a_call_with_the_floor_in_its_text_is_denied_by_the_rail_not_just_the_helper():
    rail = HumanOnlyRail()
    got = rail.before_tool_call(Call("run_shell", {"command": "ml-stack-requests answer rq_1 allow-once"}), Context())
    assert isinstance(got, Deny)


def test_the_request_commands_refuse_a_process_an_agent_started(monkeypatch, capsys):
    from ml_stack.inbox import cli
    held = requests.raise_request(ask())
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    for argv in (["list"], ["show", held.id], ["answer", held.id, "allow-once", "--fingerprint", held.fingerprint],
                 ["watch", "--once"]):
        assert cli.main(argv) == 2, argv
    out = capsys.readouterr()
    assert held.id not in out.out
    assert requests.get(held.id).state == "pending"
