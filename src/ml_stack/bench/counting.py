"""One question's bill, and the transcript the bill is a total of.

`Counting` wraps a client and keeps what each call cost -- tokens, timings, the prefix the
server kept -- and, with ``trace``, every message sent and every reply with its tool calls
and its timings. `wants_trace` says whether a run of a given size keeps one, `TRACE_CAP`
and `TRACE_TEXT_CAP` how much of one message it keeps, and `PER_QUESTION` how long one
question may take before `QuestionTimedOut`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from ml_stack.bench.backends import timings_of
from ml_stack.bench.keep import SHORT
from ml_stack.bench.record import prompt_digest

TRACE_ENV = "MLSTACK_BENCH_TRACE"
"""Set to 1 to trace a run whatever its size, 0 to trace none of it. `wants_trace`."""

TRACE_CAP = 2000
"""How much of one tool result a trace keeps, in characters.

A tool result is the largest thing in a conversation and the least of what a fine-tune
learns -- what is being taught is the call, not the answer the graph gave back. The whole
length is kept beside the cut text (``chars``), so nothing about the cost is lost by not
keeping the bytes.
"""

TRACE_TEXT_CAP = 16000
"""And of anything else -- a system prompt, a question, an answer. High enough that nothing
a model actually writes is cut; there so that one runaway message cannot fill a store."""


def wants_trace(questions: int, told: bool | None = None) -> bool:
    """Whether a run of this many questions keeps its transcript.

    On for a run of `SHORT` questions or fewer -- a sampled run, a smoke run, the shape most
    runs actually have -- and off for the hundred, where the traces would be tens of
    megabytes in a store nothing backs up. ``told`` is a person saying so either way
    (``--trace`` / ``--no-trace``), and ``MLSTACK_BENCH_TRACE`` says so for a run whose
    command line cannot.

    The default is that way round because a trace is only useful if it exists before you
    know you wanted it: today's sweep scored thousands of tool calls and kept none of them,
    and there is no way to get them back except by spending the GPU again.
    """
    if told is not None:
        return bool(told)
    asked = os.environ.get(TRACE_ENV, "").strip().lower()
    if asked:
        return asked not in ("0", "no", "off", "false")
    return 0 < int(questions) <= SHORT


def _cut(text: Any, cap: int) -> tuple[str, int, bool]:
    """``(text as kept, its whole length, whether it was cut)``."""
    whole = str(text or "")
    return (whole[:cap] if len(whole) > cap else whole), len(whole), len(whole) > cap


def _ids_in(text: str) -> int:
    """How many graph entries a tool result named, counted as the ``id`` keys in it.

    What a tool call was worth in one number: `look_up` that found nothing and `look_up`
    that found eleven are the same number of calls and the same number of characters
    apart, and only this tells them apart in a transcript.
    """
    try:
        found = json.loads(text or "")
    except ValueError:
        return 0

    def count(value: Any) -> int:
        if isinstance(value, Mapping):
            return (1 if isinstance(value.get("id"), str) else 0) + sum(
                count(v) for k, v in value.items() if k != "id")
        if isinstance(value, (list, tuple)):
            return sum(count(v) for v in value)
        return 0

    return count(found)


class Counting:
    """A client that answers exactly as the real one does, and keeps the bill.

    Wrapping is the only way to count honestly: the tokens are on the reply the server sent,
    and nothing between here and there is going to add them up for you.

    With ``trace``, it also keeps what was *said*: every message sent that it has not
    already seen, and every reply -- its tool calls with their arguments, what it wrote,
    what it thought, why it stopped, which model answered, and the timings `Spent.note`
    reads, per call, so the transcript and the totals are the same numbers added up two
    ways. Off for a long run: the totals are numbers and this is kilobytes.
    """

    def __init__(self, client: Any, *, deadline: float | None = None,
                 trace: bool = False) -> None:
        self.client = client
        # When this question must be over, as `time.time()` reads it. Each call is given the
        # time left, so a question of three calls gets one cap and not three; a call that
        # ends at the deadline with an error is the timeout, whatever it was wrapped as.
        self.deadline = deadline
        self.timed_out = False
        self.calls = 0
        self.prompt_tokens = 0
        self.processed_tokens = 0
        self.completion_tokens = 0
        # None until a call reports the figure: a program that never says what it cached,
        # drafted or spent generating leaves None, and None is not 0.
        self.cached_tokens: int | None = None
        self.draft_tokens: int | None = None
        self.draft_taken: int | None = None
        self.draft_ms: float | None = None
        self.verify_ms: float | None = None
        self.verify_n: int | None = None
        # Per call, ``(cached, processed)`` as the server reported them: the totals above
        # cannot say whether the prefix survived from one call to the next, and that --
        # see `prefix_kept` -- is the cheapest speed lever there is.
        self.per_call: list[tuple[int | None, int | None]] = []
        # What the server itself spent reading and generating, so that the difference
        # between it and the wall clock -- time spent waiting for a slot -- is a number.
        self.generating_ms: float | None = None
        self.first_token: float | None = None
        # What was said, in order: the tools offered, then every message and every reply.
        # See `wants_trace` for when it is filled, `_message` and `_reply` for what one
        # entry holds, and `bench.transcript` for it read back out.
        self.tracing = bool(trace)
        self.trace: list[dict[str, Any]] = []
        self._traced = 0        # how many of `messages` are already in it
        # A digest of the system prompt and the tool schemas the model was shown, taken on
        # the first call -- see `bench.record.prompt_digest`.
        self.prompts = ""

    def _message(self, one: Mapping[str, Any]) -> dict[str, Any]:
        """One message the model was sent, as the trace keeps it."""
        role = str(one.get("role") or "")
        cap = TRACE_CAP if role == "tool" else TRACE_TEXT_CAP
        text, whole, cut = _cut(one.get("content"), cap)
        entry: dict[str, Any] = {"role": role, "content": text, "chars": whole}
        if cut:
            entry["cut"] = True
        if role == "tool":
            entry["name"] = str(one.get("name") or "")
            entry["ids"] = _ids_in(str(one.get("content") or ""))
        return entry

    def _sent(self, messages: Any, tools: Any) -> None:
        """Everything sent this call that was not sent last call, into the trace.

        A conversation is one list that grows, so what is new is what is past the end of
        what was recorded. The assistant turns in it are skipped: the reply itself was
        recorded when it arrived, with its timings on it, and the copy `ask` appends to
        the conversation carries none of that.
        """
        rows = list(messages or ())
        if not self.trace and tools:
            self.trace.append({"role": "tools", "tools": [dict(t) for t in tools]})
        fresh = rows[self._traced:] if len(rows) >= self._traced else rows
        self._traced = len(rows)
        for one in fresh:
            if isinstance(one, Mapping) and str(one.get("role") or "") != "assistant":
                self.trace.append(self._message(one))

    def _reply(self, reply: Any, took: float, tools: Any) -> None:
        """One reply into the trace: what it called, what it wrote, and what it cost.

        Everything `Spent.note` reads, per call, so the per-call record and the per-answer
        totals are the same numbers added up two ways.
        """
        raw = getattr(reply, "raw", None) or {}
        usage = raw.get("usage") or {}
        timings = raw.get("timings") or {}
        calls = []
        for call in getattr(reply, "tool_calls", None) or ():
            fn = (call.get("function") or {}) if isinstance(call, Mapping) else {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"_unparsed": str(fn.get("arguments") or "")[:TRACE_CAP]}
            calls.append({"name": str(fn.get("name") or ""),
                          "args": args if isinstance(args, dict) else {"_value": args}})
        text, whole, cut = _cut(getattr(reply, "content", "") or "", TRACE_TEXT_CAP)
        entry: dict[str, Any] = {
            "role": "assistant", "call": self.calls,
            "model": str(raw.get("model") or ""),
            "content": text, "chars": whole,
            "thinking_chars": len(getattr(reply, "thinking", None) or ""),
            "finish": str(getattr(reply, "finish_reason", None) or ""),
            "seconds": round(float(took), 3),
            "offered": [str((t.get("function") or {}).get("name")) for t in (tools or ())],
            "tool_calls": calls,
            "tokens": {"prompt": int(usage.get("prompt_tokens") or 0),
                       "completion": int(usage.get("completion_tokens") or 0)},
            "timings": {k: (float(timings[k]) if k.endswith("_ms") else int(timings[k]))
                        for k in ("prompt_ms", "predicted_ms", "prompt_n", "cache_n",
                                  "predicted_n", "draft_n", "draft_n_accepted",
                                  "draft_n_verified", "draft_ms", "verify_ms", "verify_n")
                        if timings.get(k) is not None},
        }
        if cut:
            entry["cut"] = True
        self.trace.append(entry)

    def chat(self, messages: Any, **kw: Any) -> Any:
        self.calls += 1
        sent = time.time()
        if not self.prompts:
            system = next((str(m.get("content") or "") for m in (messages or ())
                           if isinstance(m, Mapping) and m.get("role") == "system"), "")
            self.prompts = prompt_digest(system, kw.get("tools") or ())
        if self.tracing:
            self._sent(messages, kw.get("tools"))
        if self.deadline is not None:
            left = self.deadline - sent
            if left <= 0:
                self.timed_out = True
                raise QuestionTimedOut(f"no time left for call {self.calls}")
            # The real client takes `timeout` per call and closes the connection when it
            # expires -- urllib closes the socket on the exception it raises, and llama.cpp's
            # server polls `is_connection_closed` while a non-streamed result is pending and
            # cancels the slot's tasks when it is, so the slot stops generating rather than
            # finishing a reply nobody is waiting for. A client without a `transport` of its
            # own is only held to the deadline between calls.
            if hasattr(self.client, "transport"):
                kw.setdefault("timeout", max(0.1, left))
        try:
            reply = self.client.chat(messages, **kw)
        except Exception as exc:
            if self.deadline is not None and time.time() >= self.deadline - 0.5:
                self.timed_out = True
                raise QuestionTimedOut(f"call {self.calls}: {exc}") from exc
            raise
        took = time.time() - sent
        if self.tracing:
            self._reply(reply, took, kw.get("tools"))
        raw = getattr(reply, "raw", None) or {}
        usage = raw.get("usage") or {}
        timings = timings_of(reply)
        prompt_ms, predicted_ms = timings["prompt_ms"], timings["predicted_ms"]
        if prompt_ms is not None or predicted_ms is not None:
            self.generating_ms = (self.generating_ms or 0.0) + float(prompt_ms or 0) \
                + float(predicted_ms or 0)
        if self.first_token is None and predicted_ms is not None:
            # Nothing here streams, so the first token is not seen arriving. What is known
            # is how long the server spent generating; everything before that -- waiting
            # for a slot, then reading the prompt -- is what the first token waited for.
            self.first_token = round(max(0.0, took - float(predicted_ms) / 1000), 3)
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        # A conversation re-sends everything every turn, so the prompt total counts the same
        # words over and over. What the machine actually pays for is what it had to read:
        # `timings.prompt_n`, with `cache_n` the part it kept from the turn before.
        cached, processed = timings["cache_n"], timings["prompt_n"]
        if cached is not None:
            self.cached_tokens = (self.cached_tokens or 0) + int(cached)
        self.processed_tokens += int(processed or 0)
        self.per_call.append((None if cached is None else int(cached),
                              None if processed is None else int(processed)))
        # A draft model guesses ahead and the large one checks the guesses in one pass, so
        # what decides whether it was worth serving is not that it ran but how often it was
        # right. Both are 0 on a llama-server without one and None on a program that
        # cannot say, which is how the table tells the three apart.
        if timings["draft_n"] is not None:
            self.draft_tokens = (self.draft_tokens or 0) + int(timings["draft_n"])
            self.draft_taken = (self.draft_taken or 0) + int(timings["draft_n_accepted"] or 0)
        # A build with no clock over drafting reports none of these, and counts toward none.
        if timings.get("verify_n") is not None:
            self.draft_ms = (self.draft_ms or 0.0) + float(timings.get("draft_ms") or 0.0)
            self.verify_ms = (self.verify_ms or 0.0) + float(timings.get("verify_ms") or 0.0)
            self.verify_n = (self.verify_n or 0) + int(timings["verify_n"])
        return reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


class QuestionTimedOut(RuntimeError):
    """A question ran past its `--per-question` cap; the row records it and the run goes on."""


# What one question may take before it is recorded as timed out and the run moves on.
PER_QUESTION = 300.0
