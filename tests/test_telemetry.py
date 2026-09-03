"""One call, written down once -- and a `Spent` that is the sum of its calls.

Every reply here is invented, built in the test, and no server is started: the whole point
of `telemetry.Call` is that it is a record and its arithmetic, so a fake reply with a
``raw`` dict on it is the real input.

What these guard is drift. Three records counted the same reply -- the page's `Spent`, the
bench's per-call trace, and what a scraper reads -- and every timing added to one of them
had to be added by hand to the others. So: the record reads a reply, the totals are the
records added up, the bench's dict shape maps onto the record, and `Spent.public()` still
has exactly the keys the page and the conversation store already hold.
"""

from __future__ import annotations

import json

from ml_stack.client.chat import Reply
from ml_stack.client.spent import Spent
from ml_stack.telemetry import ARGS_CAP, Call, args_summary


def reply(*, content="ok", model="tiny-Q4.gguf", prompt_n=300, cache_n=600, predicted_n=40,
          prompt_ms=100.0, predicted_ms=400.0, draft_n=0, draft_taken=0, prompt_tokens=900,
          completion_tokens=40, finish="stop", thinking="", tool=None, args="{}"):
    """A reply shaped exactly as llama.cpp's ``/v1/chat/completions`` sends one."""
    calls = ([{"id": "call_1", "type": "function",
               "function": {"name": tool, "arguments": args}}] if tool else None)
    return Reply(content=content, tool_calls=calls, finish_reason=finish,
                 thinking=thinking or None,
                 raw={"model": model,
                      "usage": {"prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens},
                      "timings": {"prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                                  "prompt_n": prompt_n, "cache_n": cache_n,
                                  "predicted_n": predicted_n,
                                  "draft_n": draft_n, "draft_n_accepted": draft_taken}})


# -- the record ---------------------------------------------------------------------------
class TestOneCall:
    def test_it_reads_every_number_the_server_reported(self):
        one = Call.from_reply(reply(draft_n=20, draft_taken=15), 0.8,
                              host="127.0.0.1", port=8099)
        assert one.model == "tiny-Q4.gguf" and one.host == "127.0.0.1" and one.port == 8099
        assert (one.prompt_n, one.cache_n, one.predicted_n) == (300, 600, 40)
        assert (one.prompt_ms, one.predicted_ms) == (100.0, 400.0)
        assert (one.draft_n, one.draft_n_accepted) == (20, 15)
        assert (one.prompt_tokens, one.completion_tokens) == (900, 40)
        assert one.seconds == 0.8 and one.finish == "stop" and one.result_chars == 2
        assert one.when > 0, "a call that does not say when is not telemetry"

    def test_the_slot_held_the_prefix_the_prompt_and_the_answer(self):
        assert Call.from_reply(reply(), 0.5).held == 940
        # a server that reports no timings at all still says how big the slot got
        bare = Reply(content="ok", raw={"usage": {"prompt_tokens": 120, "completion_tokens": 8}})
        assert Call.from_reply(bare, 0.2).held == 128

    def test_the_first_token_is_the_wall_clock_less_the_writing(self):
        """Nothing streams, so it is never seen arriving; what is known is that reading the
        prompt and waiting for a slot both happened before it."""
        assert Call.from_reply(reply(predicted_ms=400.0), 0.8).first_token == 0.4

    def test_the_time_the_server_did_not_account_for_is_the_wait_for_a_slot(self):
        one = Call.from_reply(reply(prompt_ms=100.0, predicted_ms=400.0), 2.0)
        assert one.generating_ms == 500.0 and one.waited_ms == 1500.0

    def test_a_cached_prefix_reported_the_openai_way_is_still_counted(self):
        """llama.cpp says ``timings.cache_n``; an OpenAI-shaped server says it under usage,
        and a page that read only one of them undercounted the cheapest tokens there are."""
        other = Reply(content="ok", raw={"usage": {"prompt_tokens": 900,
                                                   "prompt_tokens_details": {"cached_tokens": 700}}})
        assert Call.from_reply(other, 0.3).cache_n == 700

    def test_it_keeps_what_the_model_asked_to_call_and_with_what(self):
        one = Call.from_reply(reply(tool="look_up", args='{"word": "surveying"}'), 0.4)
        assert one.tool == "look_up" and one.args == {"word": "surveying"}
        assert one.asked == [{"name": "look_up", "args": {"word": "surveying"}}]

    def test_arguments_that_are_not_json_are_kept_verbatim_rather_than_dropped(self):
        """The model's syntax failing and the schema failing look identical afterwards
        unless what it actually wrote was kept."""
        one = Call.from_reply(reply(tool="look_up", args='{"word": '), 0.4)
        assert one.args["_unparsed"] == '{"word": '

    def test_a_long_argument_is_cut_so_a_ring_of_calls_stays_small(self):
        got = args_summary({"question": "x" * (ARGS_CAP + 500), "id": "person:iris"})
        assert got["id"] == "person:iris"
        assert len(got["question"]) == ARGS_CAP + 3 and got["question"].endswith("...")
        assert args_summary(["not", "a", "mapping"])["_value"].startswith("['not'")

    def test_several_tool_calls_in_one_reply_are_all_kept_and_named_together(self):
        many = Reply(content="", tool_calls=[
            {"function": {"name": "look_up", "arguments": '{"word": "surveying"}'}},
            {"function": {"name": "look_at", "arguments": '{"id": "person:iris"}'}}], raw={})
        one = Call.from_reply(many, 0.2)
        assert one.tool == "look_up + look_at" and len(one.asked) == 2

    def test_the_caller_says_what_the_reply_cannot(self):
        """How many graph ids a tool result named is known after the tool ran, not by the
        reply that asked for it."""
        one = Call.from_reply(reply(), 0.4, tool="look_up", result_ids=7, result_chars=1200,
                              offered=[{"function": {"name": "look_up"}}, "look_at"])
        assert one.result_ids == 7 and one.result_chars == 1200
        assert one.offered == ["look_up", "look_at"] and one.tool == "look_up"

    def test_public_is_json_ready_and_carries_the_derived_numbers(self):
        got = Call.from_reply(reply(), 0.8).public()
        assert json.dumps(got)
        assert got["first_token"] == 0.4 and got["held"] == 940
        assert got["generating_ms"] == 500.0 and got["waited_ms"] == 300.0


# -- the bench's trace, read back ----------------------------------------------------------
class TestFromTheBenchsTrace:
    def bench_entry(self):
        """Exactly the dict `graph.bench.measure.Counting._reply` appends, by hand: the
        point of the mapping is that it works on what the bench already wrote, and a run
        store full of these is what it has to keep working on."""
        return {"role": "assistant", "call": 2, "model": "tiny-Q4.gguf",
                "content": "Iris surveys land.", "chars": 18, "thinking_chars": 340,
                "finish": "stop", "seconds": 0.8, "offered": ["look_up", "look_at"],
                "tool_calls": [{"name": "look_up", "args": {"word": "surveying"}}],
                "tokens": {"prompt": 900, "completion": 40},
                "timings": {"prompt_ms": 100.0, "predicted_ms": 400.0, "prompt_n": 300,
                            "cache_n": 600, "predicted_n": 40, "draft_n": 20,
                            "draft_n_accepted": 15}}

    def test_an_assistant_entry_is_the_same_record_the_reply_made(self):
        """The two records counted the same reply and had to be kept in step by hand. They
        are one type now, and this is the proof: the bench's dict and the reply itself
        arrive at the same `Call`, field for field."""
        from_trace = Call.from_trace(self.bench_entry())
        from_reply = Call.from_reply(
            reply(content="Iris surveys land.", thinking="t" * 340, draft_n=20, draft_taken=15,
                  tool="look_up", args='{"word": "surveying"}'),
            0.8, offered=["look_up", "look_at"], when=0.0)
        assert from_trace.public() == from_reply.public()

    def test_a_tool_entry_carries_what_came_back_and_no_timings(self):
        got = Call.from_trace({"role": "tool", "name": "look_up", "content": "{...}",
                               "chars": 2400, "ids": 12})
        assert got.tool == "look_up" and got.result_chars == 2400 and got.result_ids == 12
        assert got.seconds == 0.0 and got.prompt_n == 0

    def test_the_rest_of_a_transcript_maps_to_nothing_rather_than_raising(self):
        """A trace begins with the tools offered and carries the system and user messages;
        a caller maps the whole list and drops the empties."""
        for entry in ({"role": "tools", "tools": [{"function": {"name": "look_up"}}]},
                      {"role": "user", "content": "who surveys?"}, {}):
            assert Call.from_trace(entry) == Call()


# -- Spent, as the sum of its calls ---------------------------------------------------------
class TestSpentIsTheSumOfItsCalls:
    def test_the_totals_are_the_calls_added_up(self):
        s = Spent()
        s.note(reply(prompt_n=300, cache_n=600, completion_tokens=40, draft_n=20,
                     draft_taken=15), 0.8)
        s.note(reply(prompt_n=100, cache_n=900, completion_tokens=20, draft_n=10,
                     draft_taken=5, predicted_ms=200.0), 0.4)
        assert s.calls == 2 and round(s.seconds, 3) == 1.2
        assert s.read_tokens == 400 and s.cached_tokens == 1500
        assert s.completion_tokens == 60 and s.draft_tokens == 30 and s.draft_taken == 20
        assert s.model == "tiny-Q4.gguf" and s.acceptance == 20 / 30

    def test_the_peak_is_the_largest_single_call_and_not_the_last(self):
        """With a rolling window a conversation's length says nothing about what a slot
        held; the peak is what bounds how many users fit."""
        s = Spent()
        s.note(reply(prompt_n=300, cache_n=6000, predicted_n=40), 0.8)     # 6340
        s.note(reply(prompt_n=100, cache_n=900, predicted_n=20), 0.4)      # 1020
        assert s.context_peak == 6340 and s.context_last == 1020

    def test_a_reply_cut_by_the_ceiling_is_recorded_as_truncated(self):
        s = Spent()
        s.note(reply(finish="length"), 0.4)
        s.note(reply(finish="stop"), 0.4)
        assert s.finish == "stop" and s.truncated is True

    def test_the_calls_are_not_kept_unless_they_were_asked_for(self):
        """The page wants a footer, and a hundred-turn conversation with every call kept is
        megabytes written into the conversation store for nobody."""
        s = Spent()
        s.note(reply(), 0.4)
        assert s.calls_detail == [] and "calls_detail" not in s.public()

    def test_with_keep_calls_every_call_is_there_as_well_as_summed(self):
        s = Spent(keep_calls=True)
        s.note(reply(tool="look_up", args='{"word": "surveying"}'), 0.4, host="127.0.0.1",
               port=8099)
        s.note(reply(content="Iris surveys land."), 0.8)
        assert [c.tool for c in s.calls_detail] == ["look_up", ""]
        assert s.calls_detail[0].host == "127.0.0.1" and s.calls_detail[0].port == 8099
        got = s.public()["calls_detail"]
        assert len(got) == 2 and got[0]["args"] == {"word": "surveying"}
        assert json.dumps(s.public())

    def test_a_call_folded_in_directly_counts_the_same_as_a_reply_noted(self):
        """`add` is the only place the arithmetic lives, so a caller that already has a
        `Call` -- read back out of a bench trace, say -- does not need a fake reply.

        Everything but ``parts``: those are estimated from the text with a token counter,
        not counted off the reply, so they need the words and not the record of them.
        """
        noted, folded = Spent(), Spent()
        noted.note(reply(draft_n=20, draft_taken=15), 0.8)
        folded.add(Call.from_reply(reply(draft_n=20, draft_taken=15), 0.8))
        assert {k: v for k, v in noted.public().items() if k != "parts"} == \
               {k: v for k, v in folded.public().items() if k != "parts"}
        assert noted.public()["parts"] == {"answer": 1} and folded.public()["parts"] == {}

    def test_the_public_record_still_has_exactly_the_keys_the_page_holds(self):
        """The conversation store is full of these under ``meta.spent`` and the page reads
        them by name. A key that quietly went missing is an old answer that stops showing
        what it cost."""
        s = Spent()
        s.note(reply(), 0.8)
        s.part("system", "you are a graph")
        assert set(s.public()) == {
            "model", "calls", "seconds", "generating_ms", "prompt_ms", "predicted_ms",
            "first_token", "prompt_tokens", "completion_tokens", "read_tokens",
            "cached_tokens", "draft_tokens", "draft_taken", "finish", "truncated",
            "thinking_chars", "answer_chars", "tool_calls", "context_peak", "context_last",
            "parts", "drafted", "acceptance", "tokens_per_second",
            "decode_tokens_per_second", "prompt_tokens_per_second"}

    def test_the_totals_over_a_session_read_the_same_records(self):
        a, b = Spent(), Spent()
        a.note(reply(prompt_n=300, cache_n=600, completion_tokens=40, predicted_ms=400.0), 0.8)
        b.note(reply(prompt_n=100, cache_n=900, completion_tokens=20, predicted_ms=200.0), 0.4)
        got = Spent.totals([a.public(), b.public()])
        assert got["answers"] == 2 and got["calls"] == 2
        assert got["read_tokens"] == 400 and got["completion_tokens"] == 60
        assert got["models"] == ["tiny-Q4.gguf"]
