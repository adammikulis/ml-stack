"""The tool-calling loop against a llama-server on a real socket that follows a script, with
tools that are Python callables and, in the last test, an MCP server over stdio."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from ml_stack.agent import (
    Agent,
    Budget,
    Done,
    FunctionTools,
    McpTools,
    Repair,
    Text,
    ToolCall,
    ToolResult,
    from_mcp,
    parse_arguments,
    validate,
)
from ml_stack.client import Client
from ml_stack.testing.tool_server import ToolCallingServer, Turn

SERVER = Path(__file__).with_name("toy_mcp_server.py")
ADD = {"name": "add", "description": "Add.", "inputSchema": {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"]}}


def add(a: int, b: int) -> int:
    return a + b


def call(name: str, **arguments: object) -> tuple[str, str]:
    return name, json.dumps(arguments)


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


def collect(agent: Agent, task: str = "go") -> list:
    async def go() -> list:
        return [event async for event in agent.run(task)]

    return asyncio.run(go())


def agent_for(fake: ToolCallingServer, **kwargs: object) -> Agent:
    return Agent(Client(fake.base_url), FunctionTools([ADD], {"add": add}), **kwargs)


def test_arguments_are_repaired_when_they_can_be() -> None:
    assert parse_arguments('{"a": 1}') == ({"a": 1}, "")
    assert parse_arguments("") == ({}, "")
    assert parse_arguments('```json\n{"a": 1,}\n```')[0] == {"a": 1}
    assert parse_arguments("{'a': True, 'b': None}")[0] == {"a": True, "b": None}
    assert parse_arguments('Sure! {"a": 1} hope that helps')[0] == {"a": 1}
    assert parse_arguments('{"a": 1, "b": "tru')[0] == {"a": 1, "b": "tru"}
    assert parse_arguments("[1, 2]")[0] is None
    assert parse_arguments("not json at all")[1].startswith("arguments are not a JSON object")


def test_validate_names_what_is_wrong() -> None:
    schema = {"type": "object", "required": ["a"], "additionalProperties": False,
              "properties": {"a": {"type": "integer", "minimum": 0},
                             "side": {"enum": ["F", "B"]}}}
    assert validate({"a": 1}, schema) == []
    assert validate({}, schema) == ["<root>: missing required property 'a'"]
    assert validate({"a": "1"}, schema) == ["a: expected integer, got str"]
    assert validate({"a": True}, schema) == ["a: expected integer, got bool"]
    assert validate({"a": -1, "side": "Z", "x": 1}, schema) == [
        "<root>: unexpected property 'x'", "a: -1 violates minimum 0",
        "side: 'Z' not in ['F', 'B']"]


def test_mcp_tools_become_openai_schemas_and_lean_trims_them() -> None:
    tool = {"name": "t", "description": "d" * 500, "inputSchema": {
        "type": "object", "properties": {"x": {"type": "string", "description": "y" * 500}}}}
    full, lean = from_mcp([tool])[0]["function"], from_mcp([tool], "lean")[0]["function"]
    assert len(full["description"]) == 500
    assert len(lean["description"]) == 200
    assert lean["parameters"]["properties"]["x"]["description"].endswith("...")
    assert from_mcp([{"name": "bare"}])[0]["function"]["parameters"] == {
        "type": "object", "properties": {}}
    with pytest.raises(ValueError, match="profile"):
        from_mcp([tool], "huge")


def test_a_tool_call_is_run_and_its_answer_goes_back(served) -> None:
    fake = served(Turn(calls=(call("add", a=2, b=3),)), Turn(text=("the sum ", "is 5")))
    events = collect(agent_for(fake))
    assert [type(e) for e in events] == [ToolCall, ToolResult, Text, Text, Done]
    assert events[1] == ToolResult("call_0", "add", "5")
    done = events[-1]
    assert (done.reason, done.text, done.steps, done.tool_calls) == ("answer", "the sum is 5", 2, 1)
    second = fake.bodies[1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "call_0", "name": "add", "content": "5"}
    assert second[-2]["tool_calls"][0]["function"]["arguments"] == '{"a": 2, "b": 3}'
    assert fake.bodies[0]["tools"][0]["function"]["name"] == "add"


def test_calls_in_one_turn_run_in_parallel(served) -> None:
    gate = threading.Barrier(2, timeout=5)

    def meet(n: int) -> int:
        gate.wait()
        return n

    spec = {"name": "meet", "description": "", "inputSchema": {
        "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}
    fake = served(Turn(calls=(call("meet", n=1), call("meet", n=2))), Turn(text=("ok",)))
    agent = Agent(Client(fake.base_url), FunctionTools([spec], {"meet": meet}))
    results = [e for e in collect(agent) if isinstance(e, ToolResult)]
    assert [r.text for r in results] == ["1", "2"]


def test_a_malformed_call_is_repaired_and_a_bad_one_is_sent_back(served) -> None:
    fake = served(
        Turn(calls=(("add", '{"a": 1, "b": 2,}'),)),
        Turn(calls=call("add", a="x", b=2) and (call("add", a="x", b=2),)),
        Turn(text=("fine",)))
    events = collect(agent_for(fake))
    assert events[0] == ToolCall("call_0", "add", {"a": 1, "b": 2})
    assert events[1].text == "3"
    repair = next(e for e in events if isinstance(e, Repair))
    assert repair.errors == ["a: expected integer, got str"]
    told = json.loads(fake.bodies[2]["messages"][-1]["content"])
    assert told["ok"] is False and told["errors"] == repair.errors
    assert events[-1].reason == "answer"


def test_an_unknown_tool_is_named_back(served) -> None:
    fake = served(Turn(calls=(call("nope"),)), Turn(text=("ok",)))
    repair = next(e for e in collect(agent_for(fake)) if isinstance(e, Repair))
    assert "unknown tool 'nope'" in repair.errors[0] and "'add'" in repair.errors[0]


def test_a_call_written_in_the_text_is_run(served) -> None:
    fake = served(Turn(text=('{"name": "add", "arguments": {"a": 4, "b": 4}}',)),
                  Turn(text=("8",)))
    events = collect(agent_for(fake))
    assert any(isinstance(e, ToolResult) and e.text == "8" for e in events)


def test_repeated_rejections_stop_the_run(served) -> None:
    fake = served(*[Turn(calls=(call("nope"),))] * 5)
    done = collect(agent_for(fake, budget=Budget(max_repairs=1)))[-1]
    assert (done.reason, done.steps) == ("repairs_exhausted", 2)


def test_the_budgets_stop_the_run(served) -> None:
    loop = [Turn(calls=(call("add", a=1, b=1),), completion_tokens=40)] * 10
    steps = collect(agent_for(served(*loop), budget=Budget(max_steps=3)))[-1]
    assert (steps.reason, steps.steps, steps.tool_calls) == ("max_steps", 3, 3)
    calls = collect(agent_for(served(*loop), budget=Budget(max_tool_calls=2)))[-1]
    assert (calls.reason, calls.tool_calls) == ("max_tool_calls", 2)
    tokens = collect(agent_for(served(*loop), budget=Budget(max_tokens=70)))[-1]
    assert (tokens.reason, tokens.steps, tokens.tokens) == ("max_tokens", 2, 80)


def test_results_are_truncated_and_a_hook_may_rewrite_them(served) -> None:
    wide = FunctionTools([ADD], {"add": lambda a, b: "z" * 50})
    fake = served(Turn(calls=(call("add", a=1, b=1),)), Turn(text=("ok",)))
    agent = Agent(Client(fake.base_url), wide, budget=Budget(max_result_chars=10))
    result = next(e for e in collect(agent) if isinstance(e, ToolResult))
    assert result.text == "z" * 10 + "... [40 more characters cut]"
    fake = served(Turn(calls=(call("add", a=1, b=1),)), Turn(text=("ok",)))
    hooked = Agent(Client(fake.base_url), wide, summarise=lambda n, out: f"{n}: {len(out.text)}")
    assert next(e for e in collect(hooked) if isinstance(e, ToolResult)).text == "add: 50"


def test_a_tool_that_raises_is_an_error_answer(served) -> None:
    def broken(a: int, b: int) -> int:
        raise ValueError("no")

    fake = served(Turn(calls=(call("add", a=1, b=1),)), Turn(text=("ok",)))
    agent = Agent(Client(fake.base_url), FunctionTools([ADD], {"add": broken}))
    result = next(e for e in collect(agent) if isinstance(e, ToolResult))
    assert (result.text, result.is_error) == ("ValueError: no", True)


def test_text_deltas_arrive_before_the_run_ends(served) -> None:
    fake = served(Turn(text=("a", "b", "c")))

    async def first() -> tuple[str, bool]:
        stream = agent_for(fake).run("hi")
        got = await anext(stream)
        await stream.aclose()
        return got.delta, fake.turns == []

    assert asyncio.run(first()) == ("a", True)


def test_cancelling_the_task_stops_a_slow_tool(served) -> None:
    started = threading.Event()

    def slow(a: int, b: int) -> int:
        started.set()
        time.sleep(0.5)
        return 0

    fake = served(Turn(calls=(call("add", a=1, b=1),)), Turn(text=("late",)))
    agent = Agent(Client(fake.base_url), FunctionTools([ADD], {"add": slow}))

    async def run_then_cancel() -> list:
        seen: list = []

        async def drive() -> None:
            async for event in agent.run("go"):
                seen.append(event)

        task = asyncio.ensure_future(drive())
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return seen

    seen = asyncio.run(run_then_cancel())
    assert not any(isinstance(e, Done) for e in seen)
    assert len(fake.bodies) == 1


def test_tools_from_an_mcp_server_over_stdio(served) -> None:
    pytest.importorskip("mcp")
    fake = served(Turn(calls=(call("add", a=20, b=22), call("shout", text="hi"))),
                  Turn(text=("done",)))

    async def go() -> list:
        async with McpTools.stdio(sys.executable, [str(SERVER)]) as tools:
            listed = {t["name"] for t in await tools.list_tools()}
            failed = await tools.call("fail", {})
            assert listed == {"add", "shout", "fail"} and failed.is_error
            return [e async for e in Agent(Client(fake.base_url), tools).run("go")]

    results = [e.text for e in asyncio.run(go()) if isinstance(e, ToolResult)]
    assert results == ["42", "HI"]
