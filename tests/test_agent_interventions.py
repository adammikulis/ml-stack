"""Interventions on an agent: a refusal reaches the model as a tool error, a question waits
for the person, guidance rides on the next turn, and a hook that fails refuses."""

from __future__ import annotations

import asyncio
import json

import pytest

from ml_stack import guard
from ml_stack.agent import (
    Agent,
    Confirm,
    ConfirmRequest,
    Denied,
    Deny,
    Done,
    FunctionTools,
    Guide,
    Proceed,
    ToolCall,
    ToolResult,
)
from ml_stack.client import Client
from ml_stack.testing.tool_server import ToolCallingServer, Turn

SPEC = {"name": "wipe", "description": "Delete a path.", "inputSchema": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
ran: list[str] = []


def wipe(path: str) -> str:
    ran.append(path)
    return f"wiped {path}"


class Rule:
    """An intervention that answers with whatever it was built with."""

    def __init__(self, tool=None, model=None, invocation=None) -> None:
        self.answers = {"before_tool_call": tool, "before_model_call": model,
                        "before_invocation": invocation}
        self.seen: list = []

    def before_tool_call(self, call, context):
        self.seen.append(("tool", call.name, call.arguments, context.step))
        return self._answer("before_tool_call", call)

    def before_model_call(self, context):
        self.seen.append(("model", context.step, len(context.messages)))
        return self._answer("before_model_call", context)

    def before_invocation(self, context):
        self.seen.append(("invocation", len(context.messages)))
        return self._answer("before_invocation", context)

    def _answer(self, name, arg):
        said = self.answers[name]
        if callable(said):
            return said(arg)
        return said if said is not None else Proceed()


@pytest.fixture
def served():
    started: list[ToolCallingServer] = []

    def start(*turns: Turn) -> ToolCallingServer:
        fake = ToolCallingServer(list(turns))
        started.append(fake)
        return fake

    yield start
    for fake in started:
        fake.close()


@pytest.fixture(autouse=True)
def nothing_ran():
    ran.clear()


def agent_for(fake, *hooks) -> Agent:
    return Agent(Client(fake.base_url), FunctionTools([SPEC], {"wipe": wipe}),
                 interventions=list(hooks) or guard.off("these tests are about the hooks"))


def collect(agent: Agent, task: str = "clean up") -> list:
    async def go() -> list:
        return [e async for e in agent.run(task)]

    return asyncio.run(go())


def wiping(path: str = "/tmp/x") -> Turn:
    return Turn(calls=(("wipe", json.dumps({"path": path})),))


def test_a_call_nothing_objects_to_runs(served) -> None:
    rule = Rule()
    events = collect(agent_for(served(wiping(), Turn(text=("ok",))), rule))
    assert ran == ["/tmp/x"]
    assert [type(e) for e in events] == [ToolCall, ToolResult, type(events[2]), Done]
    assert rule.seen[0] == ("invocation", 1)
    assert ("tool", "wipe", {"path": "/tmp/x"}, 1) in rule.seen


def test_a_denied_call_is_not_run_and_the_model_is_told_why(served) -> None:
    fake = served(wiping("/"), Turn(text=("I will not.",)))
    events = collect(agent_for(fake, Rule(tool=Deny("never the root"))))
    assert ran == []
    assert any(e == Denied("call_0", "wipe", "never the root") for e in events)
    told = json.loads(fake.bodies[1]["messages"][-1]["content"])
    assert told == {"ok": False, "tool": "wipe", "denied": "never the root"}
    assert events[-1].reason == "answer"


def test_the_first_denial_stops_the_later_hooks(served) -> None:
    later = Rule()
    collect(agent_for(served(wiping(), Turn(text=("ok",))), Rule(tool=Deny("no")), later))
    assert ran == [] and not [s for s in later.seen if s[0] == "tool"]


def test_a_hook_that_raises_refuses(served) -> None:
    def boom(call):
        raise RuntimeError("detector down")

    fake = served(wiping(), Turn(text=("ok",)))
    events = collect(agent_for(fake, Rule(tool=boom)))
    assert ran == []
    denied = next(e for e in events if isinstance(e, Denied))
    assert "raised RuntimeError: detector down" in denied.reason


def test_a_hook_that_answers_nonsense_refuses(served) -> None:
    events = collect(agent_for(served(wiping(), Turn(text=("ok",))), Rule(tool="yes")))
    assert ran == [] and any(isinstance(e, Denied) for e in events)


def test_a_confirm_without_a_handler_is_a_refusal(served) -> None:
    events = collect(agent_for(served(wiping(), Turn(text=("ok",))),
                               Rule(tool=Confirm("Delete /tmp/x?", {"path": "/tmp/x"}))))
    ask = next(e for e in events if isinstance(e, ConfirmRequest))
    assert ask.as_dict() == {"type": "confirm", "id": "call_0", "name": "wipe",
                             "question": "Delete /tmp/x?", "details": {"path": "/tmp/x"}}
    assert ran == []
    assert next(e for e in events if isinstance(e, Denied)).reason.startswith(
        "the person declined")


@pytest.mark.parametrize("approve", [True, False])
def test_a_confirm_waits_for_the_person(served, approve) -> None:
    agent = agent_for(served(wiping(), Turn(text=("ok",))), Rule(tool=Confirm("Delete?")))
    asked: list = []

    async def person(decision, call):
        asked.append((decision.question, call.name))
        await asyncio.sleep(0.05)
        return approve

    agent.confirm = person
    events = collect(agent)
    assert asked == [("Delete?", "wipe")]
    assert ran == (["/tmp/x"] if approve else [])
    assert sum(isinstance(e, ConfirmRequest) for e in events) == 1


def test_guidance_rides_on_the_next_model_turn(served) -> None:
    fake = served(wiping(), Turn(text=("careful then",)))
    collect(agent_for(fake, Rule(tool=Guide("That path is shared; ask first."))))
    assert ran == ["/tmp/x"]
    last = fake.bodies[1]["messages"][-1]
    assert last["role"] == "user" and last["content"] == (
        "[Guidance]\nThat path is shared; ask first.")


def test_guidance_before_a_user_turn_joins_it(served) -> None:
    fake = served(Turn(text=("ok",)))
    collect(agent_for(fake, Rule(model=Guide("be brief"))), "clean up")
    sent = fake.bodies[0]["messages"]
    assert [m["role"] for m in sent] == ["user"]
    assert sent[0]["content"] == "clean up\n\n[Guidance]\nbe brief"


def test_a_refused_model_call_ends_the_run_before_the_request(served) -> None:
    fake = served(Turn(text=("never sent",)))
    events = collect(agent_for(fake, Rule(model=Deny("over budget"))))
    assert isinstance(events[-1], Done) and events[-1].reason == "denied"
    assert events[-1].text == "over budget" and fake.bodies == []


def test_a_refused_invocation_ends_the_run_at_once(served) -> None:
    fake = served(Turn(text=("never sent",)))
    events = collect(agent_for(fake, Rule(invocation=Deny("not today"))))
    assert [type(e) for e in events] == [Done] and fake.bodies == []


def test_an_async_hook_is_awaited(served) -> None:
    class Late:
        async def before_tool_call(self, call, context):
            await asyncio.sleep(0.01)
            return Deny("slow no")

    events = collect(agent_for(served(wiping(), Turn(text=("ok",))), Late()))
    assert ran == [] and any(isinstance(e, Denied) and e.reason == "slow no" for e in events)


def test_an_agent_given_no_interventions_runs_the_builtin_rails(served) -> None:
    fake = served(wiping(), Turn(text=("ok",)))
    collect(Agent(Client(fake.base_url), FunctionTools([SPEC], {"wipe": wipe})))
    told = fake.bodies[1]["messages"][-1]["content"]
    assert told.startswith("<untrusted") and "wiped /tmp/x" in told


def test_a_credential_in_a_call_is_refused_by_the_default_rails(served) -> None:
    token = "hf_" + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3z"
    fake = served(wiping(f"/tmp/{token}"), Turn(text=("ok",)))
    events = collect(Agent(Client(fake.base_url), FunctionTools([SPEC], {"wipe": wipe})))
    assert ran == [] and any(isinstance(e, Denied) for e in events)


def test_an_empty_interventions_list_is_refused_and_the_marker_is_accepted(served) -> None:
    tools = FunctionTools([SPEC], {"wipe": wipe})
    for empty in ((), []):
        with pytest.raises(ValueError, match=r"guard\.off"):
            Agent(Client("http://127.0.0.1:9"), tools, interventions=empty)
    fake = served(wiping(), Turn(text=("ok",)))
    collect(Agent(Client(fake.base_url), tools, interventions=guard.off("a test of the marker")))
    assert not fake.bodies[1]["messages"][-1]["content"].startswith("<untrusted")
