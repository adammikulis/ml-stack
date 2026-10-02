"""Fitting a conversation into its context: each stage on its own, the invariants every
result keeps, and an agent that compacts before a request would overflow -- against a
llama-server on a real socket whose ``/tokenize`` counts words."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from ml_stack.agent import (
    Agent,
    Budget,
    Compacted,
    Compacting,
    Compaction,
    Context,
    Counter,
    Done,
    FunctionTools,
    Spill,
    ToolResult,
    Transcript,
    compact,
    context_usage,
    has_open_calls,
    model_summarizer,
)
from ml_stack.agent.compact import ELIDED, SUMMARY_PREFIX
from ml_stack.client import Client
from ml_stack.testing import FakeLlamaServer, Served
from ml_stack.testing.tool_server import ToolCallingServer, Turn


def words(n: int, stem: str = "w") -> str:
    return " ".join(f"{stem}{i}" for i in range(n))


def count_words(text: str) -> int:
    return len(text.split())


def tool_turn(n: int, name: str = "look", arguments: str | None = None, size: int = 300
              ) -> list[dict]:
    args = arguments if arguments is not None else json.dumps({"q": n})
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{n}", "type": "function", "function": {"name": name,
                                                                 "arguments": args}}]},
            {"role": "tool", "tool_call_id": f"c{n}", "name": name,
             "content": words(size, f"r{n}_")}]


def conversation(turns: int = 6, size: int = 300) -> list[dict]:
    out = [{"role": "system", "content": "You are helpful."},
           {"role": "user", "content": "find the part numbers"}]
    for n in range(turns):
        out += tool_turn(n, size=size)
    return [*out, {"role": "assistant", "content": "done so far"},
            {"role": "user", "content": "now the prices"}]


def assert_valid(messages: list[dict]) -> None:
    """Roles a chat template accepts: system first, no orphaned tool results, every call
    answered, no two user messages in a row."""
    calls: set[str] = set()
    for i, m in enumerate(messages):
        if m["role"] == "system":
            assert all(x["role"] == "system" for x in messages[:i]), "system after the start"
        if m["role"] == "assistant":
            calls = {c["id"] for c in m.get("tool_calls") or []}
        if m["role"] == "tool":
            assert m["tool_call_id"] in calls, f"orphaned result {m['tool_call_id']}"
        if i and m["role"] == "user":
            assert messages[i - 1]["role"] != "user", "two user messages in a row"
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    asked = {c["id"] for m in messages for c in m.get("tool_calls") or []}
    assert asked == answered


def summarizer(messages, prior):
    return f"{prior} | {len(messages)} messages: " + ", ".join(
        sorted({m.get("name", m["role"]) for m in messages}))


def total(messages: list[dict]) -> int:
    return context_usage(messages, None, 1, count=count_words).used


def test_usage_is_counted_by_the_server_when_it_can() -> None:
    fake = FakeLlamaServer(Served(context=1000))
    try:
        messages = [{"role": "user", "content": words(10)}]
        tools = [{"type": "function", "function": {"name": "t"}}]
        seen = context_usage(messages, tools, 1000, count=Counter(Client(fake.base_url)))
        offline = context_usage(messages, tools, 1000, count=Counter())
    finally:
        fake.close()
    assert seen.exact and seen.limit == 1000
    assert seen.used == (10 + 4) + 5          # ten words and role markers, five words of schema
    assert not offline.exact and offline.used > seen.used
    assert seen.as_dict()["fraction"] == round(seen.used / 1000, 4)


def test_an_unreachable_server_falls_back_to_the_estimator() -> None:
    fake = FakeLlamaServer(Served())
    client = Client(fake.base_url)
    fake.close()
    count = Counter(client)
    assert count(words(30)) > 30
    assert not count.exact


def test_long_tool_results_are_cut_and_kept_in_the_transcript(tmp_path) -> None:
    log = Transcript(tmp_path / "t.jsonl")
    messages = conversation(turns=3, size=500)
    result = compact(messages, budget=900, strategy="elide", count=count_words,
                     using=Compaction(keep_last=4, spill=Spill(log, above=100)))
    cut = [m for m in result.messages if m["role"] == "tool" and ELIDED in m["content"]]
    assert len(cut) == 2 and result.strategy_used == "elide"
    ref = cut[0]["content"].split("full text: ")[1].split(" ...]")[0]
    assert log.fetch(ref)["text"] == messages[3]["content"]
    assert cut[0]["content"].startswith("r0_0 r0_1") and cut[0]["content"].endswith("r0_499")
    assert result.messages[-4:] == messages[-4:]
    assert_valid(result.messages)
    again = compact(result.messages, budget=10_000, count=count_words,
                    using=Compaction(spill=Spill(log, above=100)))
    assert again.strategy_used == "none" and again.messages == result.messages


def test_a_call_a_later_call_repeats_is_dropped_with_its_result() -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
                *tool_turn(1, arguments='{"q": "same"}', size=50),
                *tool_turn(2, name="other", size=50),
                *tool_turn(3, arguments='{"q":"same"}', size=50),
                {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
    result = compact(messages, budget=10, strategy="prune", count=count_words,
                     using=Compaction(keep_last=3))
    ids = [m.get("tool_call_id") for m in result.messages if m["role"] == "tool"]
    assert ids == ["c2", "c3"] and result.dropped_count == 2
    assert_valid(result.messages)


def test_the_oldest_messages_become_one_summary(tmp_path) -> None:
    log = Transcript(tmp_path / "t.jsonl")
    messages = conversation(turns=6, size=200)
    result = compact(messages, budget=700, count=count_words,
                     using=Compaction(keep_last=4, summarizer=summarizer,
                                      spill=Spill(log, above=10_000)))
    assert result.strategy_used == "summarise"
    assert result.messages[0] == messages[0]
    assert result.messages[1] is result.summary_message
    assert result.summary_message["content"].startswith(SUMMARY_PREFIX)
    assert result.messages[-4:] == messages[-4:]
    assert result.tokens_after <= 700 < result.tokens_before
    assert result.dropped_count == len(messages) - 1 - 4 - 1 or result.dropped_count > 0
    assert_valid(result.messages)
    first = log.fetch(result.notes[-1].split(": ")[-1]) if ": " in result.notes[-1] else None
    assert first is None or first["messages"]
    recorded = [r for r in json.loads("[" + ",".join(
        (tmp_path / "t.jsonl").read_text().splitlines()) + "]") if r["kind"] == "summarised"]
    assert sum(len(r["messages"]) for r in recorded) == result.dropped_count


def test_the_same_conversation_compacts_the_same_way() -> None:
    messages = conversation()
    runs = [compact(messages, budget=700, count=count_words,
                    using=Compaction(keep_last=4, summarizer=summarizer)) for _ in range(2)]
    assert runs[0].messages == runs[1].messages


def test_compacting_again_does_not_grow() -> None:
    first = compact(conversation(), budget=700, count=count_words,
                    using=Compaction(keep_last=4, summarizer=summarizer))
    second = compact(first.messages, budget=700, count=count_words,
                     using=Compaction(keep_last=4, summarizer=summarizer))
    assert second.strategy_used == "none" and second.messages == first.messages
    more = first.messages + conversation(turns=4, size=200)[2:]
    third = compact(more, budget=700, count=count_words,
                    using=Compaction(keep_last=4, summarizer=summarizer))
    assert sum(m["content"].startswith(SUMMARY_PREFIX) for m in third.messages
               if m["role"] == "user") == 1
    assert third.tokens_after <= 700
    assert_valid(third.messages)


def test_a_failing_summarizer_falls_back_to_removal() -> None:
    def broken(messages, prior):
        raise ValueError("model is down")

    result = compact(conversation(), budget=700, count=count_words,
                     using=Compaction(keep_last=4, summarizer=broken))
    assert result.strategy_used == "truncate"
    assert any("summary failed (ValueError: model is down)" in n for n in result.notes)
    assert "were removed to fit the context" in result.summary_message["content"]
    assert result.tokens_after <= 700
    assert_valid(result.messages)


def test_without_a_summarizer_it_goes_straight_to_removal() -> None:
    result = compact(conversation(), budget=700, count=count_words,
                     using=Compaction(keep_last=4, summarize=False, summarizer=summarizer))
    assert result.strategy_used == "truncate"


def test_preserved_messages_survive_every_stage() -> None:
    messages = conversation()
    messages[3]["content"] = "ORDER-77 " + messages[3]["content"]
    result = compact(messages, budget=600, count=count_words,
                     using=Compaction(keep_last=4, summarizer=summarizer,
                                      preserve=("ORDER-77",)))
    assert messages[3] in result.messages
    assert messages[2] in result.messages        # its call travels with it
    assert_valid(result.messages)


def test_the_kept_messages_alone_over_budget_is_said() -> None:
    result = compact(conversation(turns=2, size=400), budget=50, count=count_words,
                     using=Compaction(keep_last=4, summarizer=summarizer))
    assert any("kept messages alone exceed" in n for n in result.notes)


def test_a_stage_may_be_asked_for_by_name_and_a_bad_name_is_refused() -> None:
    assert compact(conversation(), budget=10**6, strategy="truncate",
                   count=count_words).strategy_used == "none"
    with pytest.raises(ValueError, match="not auto"):
        compact(conversation(), budget=10, strategy="shrink")


def test_open_calls_are_seen() -> None:
    done = conversation(turns=1)
    assert not has_open_calls(done)
    assert has_open_calls(done[:3])


@pytest.fixture
def served():
    started: list[ToolCallingServer] = []

    def start(*turns: Turn, context: int = 2000) -> ToolCallingServer:
        fake = ToolCallingServer(list(turns), context=context)
        started.append(fake)
        return fake

    yield start
    for fake in started:
        fake.close()


def collect(agent: Agent, messages: list[dict]) -> list:
    async def go() -> list:
        return [event async for event in agent.run(messages)]

    return asyncio.run(go())


def an_agent(fake, tmp_path, **config) -> Agent:
    keep = Compaction(keep_last=4, summarizer=summarizer,
                      spill=Spill(Transcript(tmp_path / "t.jsonl"), above=10_000), **config)
    return Agent(Client(fake.base_url), FunctionTools([]), auto_compact=keep)


def test_an_agent_compacts_before_the_request_that_would_overflow(served, tmp_path) -> None:
    fake = served(Turn(text=("the prices are in",)), context=1000)
    messages = conversation(turns=6, size=200)
    events = collect(an_agent(fake, tmp_path), messages)
    seen = [e for e in events if isinstance(e, Context)]
    made = [e for e in events if isinstance(e, Compacted)]
    assert seen[0].limit == 1000 and seen[0].fraction > 0.8
    assert len(made) == 1 and made[0].before > 0.8 and made[0].after <= 0.5
    assert made[0].as_dict()["type"] == "compact" and made[0].dropped > 0
    sent = fake.bodies[0]["messages"]
    assert len(sent) < len(conversation(turns=6, size=200))
    assert sent[0]["role"] == "system" and sent[1]["content"].startswith(SUMMARY_PREFIX)
    assert sent[-1]["content"] == "now the prices"
    assert isinstance(events[-1], Done) and events[-1].reason == "answer"
    assert_valid(sent)


def test_nothing_is_compacted_while_the_context_has_room(served, tmp_path) -> None:
    fake = served(Turn(text=("ok",)), context=100_000)
    events = collect(an_agent(fake, tmp_path), conversation(turns=2, size=20))
    assert not any(isinstance(e, Compacted) for e in events)
    assert any(isinstance(e, Context) for e in events)


def test_a_refusal_for_length_compacts_and_asks_once_more(served, tmp_path) -> None:
    fake = served(Turn(text=("ok",)), context=100_000)
    fake.overflow = 1
    messages = conversation(turns=6, size=200)
    events = collect(an_agent(fake, tmp_path, threshold=0.99, target=0.01), messages)
    made = [e for e in events if isinstance(e, Compacted)]
    assert len(made) == 1 and events[-1].reason == "answer"
    assert len(fake.bodies) == 2
    assert len(fake.bodies[1]["messages"]) < len(fake.bodies[0]["messages"])


def test_a_refusal_that_is_not_about_length_is_raised(served, tmp_path) -> None:
    fake = served(Turn(text=("ok",)))
    fake.refuse["/v1/chat/completions"] = 400
    with pytest.raises(Exception, match="400"):
        collect(an_agent(fake, tmp_path), conversation(turns=1))


def test_compact_now_runs_whatever_the_context_holds(served, tmp_path) -> None:
    fake = served(context=100_000)
    messages = conversation(turns=6, size=200)
    result = asyncio.run(an_agent(fake, tmp_path, context_size=1000).compact_now(messages))
    assert result.strategy_used == "summarise" and messages == result.messages
    assert fake.bodies == []


def test_cancelling_during_a_summary_leaves_the_conversation_alone(served, tmp_path) -> None:
    started = threading.Event()

    def slow(messages, prior):
        started.set()
        time.sleep(0.4)
        return "late"

    fake = served(Turn(text=("ok",)), context=1000)
    keep = Compaction(keep_last=4, summarizer=slow)
    agent = Agent(Client(fake.base_url), FunctionTools([]), auto_compact=keep)
    messages = conversation(turns=6, size=200)
    before = list(messages)

    async def cancelled() -> None:
        async def drive() -> None:
            async for _ in agent.run(messages):
                pass

        task = asyncio.ensure_future(drive())
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancelled())
    assert messages == before and fake.bodies == []


def test_tool_results_inside_a_run_are_not_orphaned_by_a_compaction(served, tmp_path) -> None:
    spec = {"name": "look", "description": "", "inputSchema": {
        "type": "object", "properties": {"q": {"type": "integer"}}, "required": ["q"]},
            "annotations": {"readOnlyHint": True}}
    tools = FunctionTools([spec], {"look": lambda q: words(150, f"hit{q}_")})
    fake = served(*[Turn(calls=(("look", json.dumps({"q": n})),)) for n in range(5)],
                  Turn(text=("finished",)), context=900)
    keep = Compaction(keep_last=3, summarizer=summarizer,
                      spill=Spill(Transcript(tmp_path / "t.jsonl"), above=10_000))
    agent = Agent(Client(fake.base_url), tools, auto_compact=keep, budget=Budget(max_steps=9))
    events = collect(agent, [{"role": "user", "content": "start"}])
    assert events[-1].reason == "answer"
    assert any(isinstance(e, Compacted) for e in events)
    assert sum(isinstance(e, ToolResult) for e in events) == 5
    for body in fake.bodies:
        assert_valid(body["messages"])


def test_the_model_summarizer_asks_for_the_sections_and_the_identifiers(served) -> None:
    fake = served(Turn(text=("Goals: x",)))
    write = model_summarizer(Client(fake.base_url), words=120)
    got = write([{"role": "user", "content": "order ZX-9981 please"},
                 *tool_turn(1, size=5)], "earlier: ZX-1")
    assert got == "Goals: x"
    system, user = fake.bodies[0]["messages"]
    assert "Goals:" in system["content"] and "Open tasks:" in system["content"]
    assert "under 120 words" in system["content"]
    assert "earlier: ZX-1" in user["content"] and "ZX-9981" in user["content"]
    assert "assistant called look({\"q\": 1})" in user["content"]


def test_a_wrapped_client_compacts_the_messages_it_is_handed(served, tmp_path) -> None:
    fake = served(Turn(text=("fine",)), context=1000)
    seen: list = []
    config = Compaction(keep_last=4, summarizer=summarizer,
                        spill=Spill(Transcript(tmp_path / "t.jsonl"), above=10_000))
    client = Compacting(Client(fake.base_url), config, on_event=seen.append)
    messages = conversation(turns=6, size=200)
    reply = client.chat(messages)
    assert reply.content == "fine"
    assert [type(e) for e in seen] == [Context, Compacted]
    assert messages[1]["content"].startswith(SUMMARY_PREFIX) and fake.bodies[0]["messages"] == messages
    assert client.base_url == fake.base_url


def test_a_repeated_call_inside_the_kept_messages_is_kept() -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
                *tool_turn(1, arguments='{"q": "same"}', size=50),
                *tool_turn(2, arguments='{"q": "same"}', size=50),
                {"role": "user", "content": "next"}]
    result = compact(messages, budget=10, strategy="prune", count=count_words,
                     using=Compaction(keep_last=5))
    assert result.messages == messages and result.dropped_count == 0


def test_a_conversation_waiting_on_a_tool_result_is_not_compacted(served, tmp_path) -> None:
    from ml_stack.agent import AutoCompact

    fake = served(context=500)
    auto = AutoCompact(Client(fake.base_url), Compaction(keep_last=2, summarizer=summarizer))
    waiting = [*conversation(turns=4, size=200)[:-2],
               {"role": "assistant", "content": None, "tool_calls": [
                   {"id": "open", "type": "function",
                    "function": {"name": "look", "arguments": "{}"}}]}]
    before = list(waiting)
    events = asyncio.run(auto.before(waiting, []))
    assert [type(e) for e in events] == [Context] and events[0].fraction > 0.8
    assert waiting == before
