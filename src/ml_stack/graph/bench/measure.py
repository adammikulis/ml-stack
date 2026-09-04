"""Asking the questions and counting what each cost.

The questions themselves (`read_questions`, `sample` -- a short run that still asks about
every kind), the ordinary way to ask a graph one (`asking`), one question through the
client with the bill kept (`Counting`, `_ask_once`, the `--per-question` cap), a set of
them one at a time (`measure`) or N conversations at once (`concurrent`), and what the
server holding the model costs and whether it is free to be timed (`footprint`, `busy`,
`slot_count`, `_idle`).

The bill is a total; the *trace* (`Counting.trace`, `wants_trace`, `TRACE_CAP`) is what the
total is a total of -- every message sent, every tool call made with its arguments, and
every reply with its timings, kept on the row and in the store. It is what a fine-tune is
built from (`ml-stack-train-tools from-bench`) and what `ml-stack-bench show --trace` reads
back, so nothing here decides what a training example looks like: it records what happened.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.footprint`,
# `bench.busy` -- so anything patchable is looked up there at call time, never bound here
# at import.
from ml_stack.graph import bench
from ml_stack.graph.bench.backends import http_of, processes, served_by, timings_of
from ml_stack.graph.bench.keep import SHORT, SMOKE
from ml_stack.graph.bench.score import Row, prefix_kept, unread_named
from ml_stack.graph.vectors import MARGIN, stands_out


def found(store: str | Path | None, embed_url: str = "",
          embed_model: str = "") -> tuple[str, str]:
    """``(finder, why)``: which look_up a run measures, and why it is not the one asked for.

    ``chars``: no store, so `look_up` matches characters and nothing else. ``words``: a
    store's word index votes as well, so "compilers" finds "compiler". ``meaning``: and its
    vectors, through the embedder at ``embed_url`` -- only when the store holds vectors for
    ``embed_model`` (any model's, when none is named), read from the store itself; an
    embedder named beside a store with none is ``words``, and ``why`` says so. Recorded on
    every run and printed in the table, for the same reason `ctx` is: a comparison across
    finders is two measurements.
    """
    if store is None or not str(store):
        return "chars", ""
    if not embed_url:
        return "words", ""
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.vectors import embedded

    try:
        with GraphStore(store, read_only=True) as held:
            vectors = embedded(held, model=embed_model)
    except Exception:  # noqa: BLE001 - a store that will not open holds no vectors
        vectors = 0
    if vectors:
        return "meaning", ""
    return "words", f"no vectors{f' for {embed_model}' if embed_model else ''} in the store"


def finding(store: str | Path | None, embed_url: str = "", embed_model: str = "") -> str:
    """Which look_up a run measures -- ``chars``, ``words`` or ``meaning`` -- see `found`."""
    return found(store, embed_url, embed_model)[0]


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
                                  "predicted_n", "draft_n", "draft_n_accepted")
                        if timings.get(k) is not None},
        }
        if cut:
            entry["cut"] = True
        self.trace.append(entry)

    def chat(self, messages: Any, **kw: Any) -> Any:
        self.calls += 1
        sent = time.time()
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
            # finishing a reply nobody is waiting for. A client without a `timeout` of its
            # own is only held to the deadline between calls.
            if hasattr(self.client, "timeout"):
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
        return reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


class QuestionTimedOut(RuntimeError):
    """A question ran past its `--per-question` cap; the row records it and the run goes on."""


# What one question may take before it is recorded as timed out and the run moves on.
PER_QUESTION = 300.0


def _how_many(args: Any) -> int:
    """How many questions to ask: --sample wins, then --short, then all of them."""
    asked = int(getattr(args, "sample", 0) or 0)
    if getattr(args, "smoke", False):
        return SMOKE
    return asked or (SHORT if getattr(args, "short", False) else 0)


def filed(questions: Sequence[Mapping[str, Any]],
          graph: Mapping[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The questions grouped by the kind of answer each asks for.

    A question is filed under the *rarest* kind it names, so one that asks for an event and
    a person counts towards events -- the kind that has few questions -- rather than towards
    people, who have most of them. That is what stops a kind with three questions from being
    crowded out of a short run by a kind with fifty. `sample` draws from these groups and
    `mix` counts them, so what a short run covers and what the mix reports are one rule.

    ``graph`` says what kind each id is; without one, the invented community's.
    """
    scored = [dict(q) for q in questions]
    if graph is None:
        from ml_stack.graph.community import graph as invented

        graph = invented()
    kind = {str(node.get("id")): str(node.get("kind") or "") for node in
            (graph.get("nodes") or ())}

    def kinds_of(q: Mapping[str, Any]) -> set[str]:
        return {kind.get(str(e), "?") for e in (q.get("expect") or ())} or {"nobody"}

    asked = {k: sum(1 for q in scored if k in kinds_of(q))
             for q in scored for k in kinds_of(q)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for q in scored:
        grouped.setdefault(min(kinds_of(q), key=asked.__getitem__), []).append(q)
    return grouped


def mix(questions: Sequence[Mapping[str, Any]],
        graph: Mapping[str, Any] | None = None) -> dict[str, int]:
    """How many questions ask for each kind of answer, commonest first.

    The one number that says whether a question set still measures the whole page or has
    drifted into being about people: a set is grown a handful at a time, and the kind each
    addition lands under is not the kind whoever wrote it had in mind. `ml-stack-bench
    prepare --mix` prints it.
    """
    grouped = filed(questions, graph)
    return dict(sorted(((k, len(v)) for k, v in grouped.items()),
                       key=lambda kv: (-kv[1], kv[0])))


def sample(questions: Sequence[Mapping[str, Any]], n: int,
           graph: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """``n`` questions that still cover every kind of answer, or all of them.

    Not the first n, and not an even stride either. A question set is written in groups --
    finding people, then joining them, then the ones whose answer is nobody -- so the first
    eight are eight of one kind, and an even stride over a set that is two-thirds people
    returns two-thirds people and drops whole kinds. Both measure one thing well and call it
    a shorter benchmark.

    So it takes one of each kind first -- person, org, place, topic, opportunity, event, and
    the questions whose right answer is nobody -- and only then fills up in proportion. A
    short run that has stopped asking about places is not a short run, it is a different one.

    Deterministic, so two short runs compare with each other. They do not compare with a
    full run, which is what the `n` column on every line is for.
    """
    scored = [dict(q) for q in questions]
    if n <= 0 or n >= len(scored):
        return scored

    # a question is filed under the rarest kind it asks for, so a kind that appears in
    # only one question is never crowded out by one that appears in twenty
    grouped = filed(scored, graph)

    taken: list[dict[str, Any]] = []
    order = sorted(grouped, key=lambda k: len(grouped[k]))
    for group in order:                       # one of every kind, rarest first
        if len(taken) < n:
            taken.append(grouped[group][0])
    for group in order:                       # then in proportion, evenly within each
        rest = grouped[group][1:]
        share = max(0, round((n - len(order)) * len(grouped[group]) / len(scored)))
        step = (len(rest) / share) if share else 0
        for i in range(min(share, len(rest))):
            if len(taken) < n:
                taken.append(rest[int(i * step)])
    for q in scored:                          # and top up in order if rounding left room
        if len(taken) >= n:
            break
        if q not in taken:
            taken.append(q)
    return [q for q in scored if q in taken][:n]


def read_questions(path: str | Path) -> list[dict[str, Any]]:
    """One question per line: a bare string, or ``{"q": ..., "expect": [ids]}``."""
    out = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = {"q": line}
        out.append(row if isinstance(row, dict) else {"q": str(row)})
    return out


def _ask_once(ask: Callable[..., Any], one: Mapping[str, Any], *, label: str, client: Any,
              graph: Mapping[str, Any] | None = None, turns: Sequence[Mapping[str, str]] = (),
              conversation: int = 0, turn: int = 0,
              per_question: float = PER_QUESTION, trace: bool = False) -> tuple[Row, str]:
    """One question through ``ask(question, client)``, and what it cost; with the answer's
    text, which a conversation carries into its next turn.

    ``per_question`` is the most it may take. Past that the row is kept as timed out: no
    answer, the cap as its wall clock, scored wrong -- and the next question is asked. A
    question that hangs is a result, not a reason for the run to.

    ``trace`` keeps the transcript on the row as well as the totals -- see `Counting` and
    `wants_trace`. A question that failed or timed out keeps the trace it got to: that is
    the transcript worth having, since it says where it went wrong.
    """
    began = time.time()
    counting = Counting(client, deadline=began + per_question if per_question else None,
                        trace=trace)
    row = Row(label=label, question=str(one.get("q") or ""),
              expected=[str(i) for i in (one.get("expect") or ())],
              conversation=conversation, turn=turn)
    said = ""
    try:
        out = ask(row.question, counting, **({"turns": list(turns)} if turns else {}))
        # an Answer, or the payload a project sends its own page; both say the same things
        read = out.get if isinstance(out, Mapping) else lambda k, d=None: getattr(out, k, d)
        row.steps = read("why", "") or ""
        said = str(read("content", "") or "")
        row.answer_chars = len(said)
        row.shown = list(read("show", None) or read("ids", None) or [])
        if graph is not None:
            touched = [str(i) for key in ("found", "read", "path", "show", "ids")
                       for i in (read(key, None) or ())]
            row.unread = unread_named(said, graph, touched)
            row.unread_named = len(row.unread)
    except Exception as exc:  # noqa: BLE001 - a failure is a result, not the end of the run
        row.error = f"{type(exc).__name__}: {exc}"[:200]
    row.seconds = round(time.time() - began, 2)
    if counting.timed_out:
        # Whatever `ask` made of the timeout -- raised it, or caught it and answered with
        # what it had -- the question was not answered inside the cap, and is scored so.
        row.timed_out = True
        row.error = f"timed out after {per_question:.0f}s"
        row.seconds = float(per_question)
        row.shown, row.unread, row.unread_named, row.answer_chars, said = [], [], 0, 0, ""
    # None for every figure the program did not report: not measured is not 0
    row.first_token = counting.first_token
    row.queued = (round(max(0.0, row.seconds - counting.generating_ms / 1000), 2)
                  if counting.generating_ms is not None else None)
    row.calls = counting.calls
    row.prompt_tokens = counting.prompt_tokens
    row.cached_tokens = counting.cached_tokens
    row.processed_tokens = counting.processed_tokens
    row.completion_tokens = counting.completion_tokens
    row.draft_tokens = counting.draft_tokens
    row.draft_taken = counting.draft_taken
    row.cache_calls = [[c, p] for c, p in counting.per_call]
    row.trace = counting.trace
    if any(c is not None for c, _ in counting.per_call):
        row.prefix_kept, row.prefix_turns = prefix_kept(counting.per_call)
        row.prefix_hits = row.prefix_kept / row.prefix_turns if row.prefix_turns else None
    else:
        row.prefix_kept = row.prefix_turns = row.prefix_hits = None
    return row, said


def measure(ask: Callable[[str, Any], Any], questions: Sequence[dict[str, Any]], *,
            label: str, client: Any, log: Callable[[str], None] | None = None,
            graph: Mapping[str, Any] | None = None,
            per_question: float = PER_QUESTION, trace: bool | None = None,
            baseline: Mapping[str, int] | None = None) -> list[Row]:
    """Ask each question once through ``ask(question, client)`` and record what it cost.

    Given the ``graph``, each row also counts the entries the answer named that no tool
    call produced -- see `unread_named`. ``per_question`` caps each question; see `_ask_once`.

    ``trace`` keeps the whole transcript on each row as well: unset, `wants_trace` decides
    it from how many questions are being asked -- on for a sampled run, off for the
    hundred. The default is here rather than on the command line so that every caller gets
    it: the day a sweep scored thousands of tool calls and kept none of them, the flag it
    needed did not exist yet either.

    While the questions are asked, a `Watching` samples what the server holds -- Real Mem,
    Memory, the machine's wired -- every `SAMPLE_EVERY` seconds and keeps the maxima, which
    `footprint` folds into the run's record afterwards. Flash-Next was sized from single
    readings taken after the last answer, and nobody could say whether ~90G was its peak or
    its trough. ``baseline`` is what a caller read *before* the server came up, so the
    server's own wired cost is a subtraction rather than a guess.
    """
    from contextlib import nullcontext

    traced = wants_trace(len(questions), trace)
    rows = []
    # what the server holds while it is answering, sampled -- see `Watching`. The server is
    # found from the client's own base_url, so nothing above has to carry a pid; a client
    # with none, or a URL this machine does not own, samples nothing and says so.
    at = str(getattr(client, "base_url", "") or "")
    with watching(at, baseline=baseline, client=client) if at else nullcontext():
        for one in questions:
            row, _ = _ask_once(ask, one, label=label, client=client, graph=graph,
                               per_question=per_question, trace=traced)
            rows.append(row)
            if log:
                log(f"  {row.seconds:5.1f}s {row.calls:3} calls  {row.question[:56]}"
                    + ("  TIMED OUT" if row.timed_out else ""))
    return rows


def slot_count(base_url: str) -> int:
    """How many slots that server has, read from ``/slots`` as `busy` reads it.

    -1 when it will not say. It is what decides whether concurrent conversations queue:
    more of them than slots, and a turn waits for a slot before a token is read.
    """
    from ml_stack.client.http import request_json

    try:
        slots = request_json(f"{base_url.rstrip('/')}/slots", timeout=5.0, method="GET")
    except Exception:  # noqa: BLE001 - a server that will not answer has an unknown count
        return -1
    return len(slots) if isinstance(slots, list) else -1


# -------------------------------------------------------- what the server held, at its most
#
# `footprint` reads memory once, when the questions are over, and by then the cache that was
# full while four conversations were in flight has been let go. Flash-Next was sized at ~90G
# from readings like that and nobody could say whether 90G was the peak or the trough -- so
# "how many of these fit on this machine" had no number behind it.
#
# Two figures per sample, because macOS shows two and they are not the same thing:
#
#   `resident_peak`  -- Activity Monitor's **Real Mem**. The resident set: every physical
#                       page mapped in, shared and file-backed pages included. `ps -o rss`,
#                       `psutil.Process.memory_info().rss`, `ri_resident_size`.
#   `footprint_peak` -- Activity Monitor's **Memory**. The phys_footprint: dirty and
#                       compressed pages this process is charged for, clean file-backed
#                       pages excluded. `vmmap --summary`'s footprint, `ps -o footprint`,
#                       `ri_phys_footprint`. psutil has no field for it, so it is read
#                       through `proc_pid_rusage` -- see `_rusage_footprint`.
#
# They diverge exactly where it matters: llama.cpp mmaps its weights, so an 87G model can be
# most of the resident set and almost none of the footprint. Memory pressure is charged on
# the footprint and eviction is felt on the resident set, so both are kept and neither is
# called "the memory".
#
# Beside them the machine's own: `wired_peak` (what nothing can page out) against
# `wired_baseline` (the same, before the server came up -- the difference is the server's
# own wired cost), and `available_low`, free plus inactive at its lowest, which is the
# pressure proxy `vm_stat` gives.

# How often the sampler looks, in seconds. Once a second: a runner Ollama spawns on the
# first request holds the weights within a second of it, and a llama-server's resident
# set moves on the scale of a prompt being read.
SAMPLE_EVERY = 1.0

# `proc_pid_rusage(pid, RUSAGE_INFO_V4, &buf)`, and where `ri_phys_footprint` sits in
# `rusage_info_v4`: sixteen bytes of uuid, then user and system time, two wakeup counts,
# pageins, wired size, resident size, and the footprint -- the eighth `uint64_t`. Verified
# against `ps -o rss` on this machine rather than counted off the header, because counting
# it off the header put it one slot late and read the process's start time as a footprint
# of eight terabytes.
RUSAGE_INFO_V4 = 4
PHYS_FOOTPRINT_AT = 16 + 7 * 8


def _rusage_footprint(pid: int) -> int:
    """macOS's phys_footprint for ``pid`` -- Activity Monitor's "Memory" -- or 0.

    The seam: everything else about the sampler is ordinary Python, and this one function
    reaches into libSystem through ctypes. `footprint_of` calls it as ``bench.``, the way
    everything patchable in this package is called, so a test replaces this one function and
    nothing anywhere else has to pretend to be a kernel. 0 for a process that is gone, a platform without the
    call, or any failure at all -- a memory reading is never worth a run not finishing.
    """
    if sys.platform != "darwin":
        return 0
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.dylib")
        buf = ctypes.create_string_buffer(1024)
        if lib.proc_pid_rusage(ctypes.c_int(int(pid)), ctypes.c_int(RUSAGE_INFO_V4),
                               ctypes.byref(buf)) != 0:
            return 0
        return int.from_bytes(buf.raw[PHYS_FOOTPRINT_AT:PHYS_FOOTPRINT_AT + 8], sys.byteorder)
    except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
        return 0


def footprint_of(process: Any) -> int:
    """One process's phys_footprint in bytes -- Activity Monitor's "Memory".

    psutil's own field where a build has one (some expose it in ``memory_full_info``), the
    `proc_pid_rusage` read where it does not, and the resident set on every platform that
    has no such distinction -- Linux and Windows charge a process for what is resident, so
    there the two figures are the same number and the table says so by printing it twice.
    """
    try:
        info = process.memory_info()
        for name in ("phys_footprint", "footprint"):
            got = int(getattr(info, name, 0) or 0)
            if got:
                return got
    except Exception:  # noqa: BLE001
        pass
    through_kernel = bench._rusage_footprint(int(getattr(process, "pid", 0) or 0))
    if through_kernel:
        return through_kernel
    try:
        return int(process.memory_info().rss)
    except Exception:  # noqa: BLE001
        return 0


def machine_memory() -> dict[str, int]:
    """``wired`` and ``unpressured`` for the whole machine, as far as this platform says.

    ``wired`` is what nothing can page out -- the figure a server's own load moves, and the
    one that decides whether a second model fits beside the first. ``unpressured`` is free
    plus inactive: the pages that are there for the asking, which is the closest thing to
    `vm_stat`'s pressure that a portable call gives. Missing keys on a platform whose
    ``virtual_memory`` has no such field, rather than a zero that reads as measured.
    """
    out: dict[str, int] = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001
        return out
    if hasattr(vm, "wired"):
        out["wired"] = int(vm.wired)
    free, inactive = int(getattr(vm, "free", 0) or 0), int(getattr(vm, "inactive", 0) or 0)
    if free or inactive:
        out["unpressured"] = free + inactive
    elif getattr(vm, "available", 0):
        out["unpressured"] = int(vm.available)
    return out


def said_by(client: Any) -> dict[str, Any] | None:
    """What a client says served it (`Client.served_by`), None for one that cannot say --
    a llama-server's record is read by `footprint` from ``/props`` instead."""
    if client is None or not hasattr(client, "served_by"):
        return None
    return served_by(client)


def serving_pids(base_url: str, client: Any = None) -> list[int]:
    """The pids holding the weights behind ``base_url``: what the client says
    (`Client.processes`), else the llama-server with ``--port N`` on its command line.

    Empty for a run against a ``--base-url`` somebody else put up on another machine:
    nothing here owns that port, and a run that samples nothing must say nothing rather
    than report zeroes.
    """
    named = processes(client) if client is not None else []
    if named:
        return named
    try:
        import psutil

        port = int(str(http_of(base_url)).rsplit(":", 1)[-1].strip("/"))
    except Exception:  # noqa: BLE001
        return []
    try:
        for process in psutil.process_iter(["pid", "cmdline"]):
            line = " ".join(process.info.get("cmdline") or ())
            if "llama-server" in line and f"--port {port}" in line:
                return [int(process.info.get("pid") or getattr(process, "pid", 0) or 0)]
    except Exception:  # noqa: BLE001
        return []
    return []


def serving_process(base_url: str, client: Any = None) -> Any | None:
    """The first process holding the weights behind ``base_url``, as a psutil process,
    or None -- see `serving_pids`."""
    pids = serving_pids(base_url, client)
    if not pids:
        return None
    try:
        import psutil

        return psutil.Process(pids[0])
    except Exception:  # noqa: BLE001
        return None


def _every(psutil: Any) -> list[Any]:
    try:
        return list(psutil.process_iter(["pid", "cmdline"]))
    except Exception:  # noqa: BLE001
        return []


def process_tree(pids: Sequence[int]) -> list[Any]:
    """Every process under ``pids`` -- each one and its children, read now -- as psutil
    processes, each once. Ollama spawns the runner that holds the weights after the first
    request, so the tree is re-read on every call rather than kept."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return []
    out: dict[int, Any] = {}
    for pid in pids:
        try:
            parent = psutil.Process(int(pid))
        except Exception:  # noqa: BLE001 - gone, not ours to read, or no such call here
            parent = next((p for p in _every(psutil)
                           if int((getattr(p, "info", None) or {}).get("pid")
                                  or getattr(p, "pid", 0) or 0) == int(pid)), None)
            if parent is None:
                continue
        out.setdefault(int(pid), parent)
        try:
            for child in parent.children(recursive=True):
                out.setdefault(int(getattr(child, "pid", 0) or 0), child)
        except Exception:  # noqa: BLE001
            continue
    return list(out.values())


class Watching:
    """A thread that reads what the server holds every `SAMPLE_EVERY` seconds, and keeps
    the worst of it.

    Started before the questions and stopped after them, so what it reports is the most the
    machine was asked for *while it was answering* -- which is the number "how many users
    fit on this box" is a division of, and the number a single reading after the fact
    cannot give.

    ``peaks`` is what goes into the run's record: ``resident_peak`` (Real Mem, summed over
    the process tree -- listener and runner both -- with ``resident_peak_at`` the sample
    it was seen on and ``processes`` the most the tree held), ``footprint_peak`` (Memory),
    ``wired_peak`` and ``wired_baseline`` for the machine, ``available_low``,
    ``sampled_every`` and ``samples``. A run with no process to watch -- a ``--base-url``
    this machine does not own -- reports ``sampled: "no served process on this machine"``
    and no figures, because nothing measured is not zero.

    The pids come from the ``client`` (`Client.processes`) when it can say, else from the
    port; the tree under them is re-read on every sample, since Ollama's runner is not
    there until the first request has been made.
    """

    def __init__(self, base_url: str, *, every: float = SAMPLE_EVERY,
                 baseline: Mapping[str, int] | None = None, start: bool = True,
                 client: Any = None) -> None:
        self.base_url = base_url
        self.every = float(every)
        self.client = client
        self.pids = serving_pids(base_url, client)
        tree = process_tree(self.pids) if self.pids else []
        self.process = tree[0] if tree else None
        # what served it, read once here so `footprint` finds it beside the peaks: the
        # client is the one thing that can say, and `footprint` is not handed the client
        self.served = said_by(client)
        # taken before the server came up when a caller has one; otherwise the first thing
        # this thread sees, which includes the server and is honest about saying so
        self.baseline = dict(baseline) if baseline is not None else machine_memory()
        self.baseline_before_load = baseline is not None
        self.resident = self.footprint = self.wired = 0
        self.resident_at = 0
        self.processes = 0
        self.available: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        # ``start=False`` leaves the thread unstarted and `_once` the only way it samples,
        # which is how it is tested: a series read at times a test chooses says exactly what
        # the maxima are, where a thread and a clock say something near them.
        self._thread = threading.Thread(target=self._watch, daemon=True)
        if start:
            self._thread.start()

    def _once(self) -> None:
        if not self.pids:
            # a runner that was not there when the watch began: asked again each tick
            self.pids = serving_pids(self.base_url, self.client)
        tree = process_tree(self.pids) if self.pids else []
        if tree:
            resident = footprint = 0
            counted = 0
            for process in tree:
                try:
                    resident += int(process.memory_info().rss)
                    footprint += footprint_of(process)
                    counted += 1
                except Exception:  # noqa: BLE001 - one that has gone is not an error here
                    continue
            if counted:
                self.process = tree[0]
                self.processes = max(self.processes, counted)
                if resident > self.resident:
                    self.resident, self.resident_at = resident, self.samples + 1
                self.footprint = max(self.footprint, footprint)
        elif self.pids:
            self.process = None
        held = machine_memory()
        if "wired" in held:
            self.wired = max(self.wired, held["wired"])
        if "unpressured" in held:
            self.available = (held["unpressured"] if self.available is None
                              else min(self.available, held["unpressured"]))
        self.samples += 1

    def _watch(self) -> None:
        while not self._stop.is_set():
            self._once()
            self._stop.wait(self.every)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
        self._once()                     # one last look, so a short run samples at all
        return self.peaks

    @property
    def peaks(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sampled_every": self.every, "samples": self.samples}
        if self.served:
            out["served_by"] = dict(self.served)
        if self.pids:
            out["pids"] = list(self.pids)
        if self.process is None and not self.resident:
            out["sampled"] = "no served process on this machine"
        if self.resident:
            out["resident_peak"] = self.resident
            out["resident_peak_at"] = self.resident_at
            out["processes"] = self.processes
        if self.footprint:
            out["footprint_peak"] = self.footprint
        if self.wired:
            out["wired_peak"] = self.wired
        if "wired" in self.baseline:
            out["wired_baseline"] = self.baseline["wired"]
            # False when the baseline was taken with the server already up, so the
            # difference is not the server's own wired cost and nothing should read it as one
            out["wired_baseline_before_load"] = self.baseline_before_load
        if self.available is not None:
            out["available_low"] = self.available
        return out


# What the sampler saw, by the server it was watching, for `footprint` to fold into the
# record. The two belong together -- both are "what the server costs" -- and putting the
# handover here rather than through every caller is what lets a run keep peaks without the
# serving path having to carry them. One bench measures one server at a time, under the
# measuring lock, so the key is enough.
_WATCHED: dict[str, dict[str, Any]] = {}


def watched(base_url: str, peaks: Mapping[str, Any]) -> None:
    """Leave what `Watching` recorded where `footprint` will find it."""
    _WATCHED[http_of(base_url)] = dict(peaks)


def watching(base_url: str, *, every: float = SAMPLE_EVERY,
             baseline: Mapping[str, int] | None = None, client: Any = None) -> Any:
    """`Watching` over ``base_url``, as a context manager that files its peaks on exit."""
    from contextlib import contextmanager

    @contextmanager
    def held() -> Any:
        watcher = Watching(base_url, every=every, baseline=baseline, client=client)
        try:
            yield watcher
        finally:
            watched(base_url, watcher.stop())

    return held()


class _Peak:
    """The most a server held while a run was going.

    `footprint` reads resident memory once, after the fact, and a cache that was full while
    four conversations were in flight has been let go by then. Sampled every second on a
    thread; ``stop`` returns the footprint with the largest resident figure seen.
    """

    def __init__(self, base_url: str, every: float = 1.0) -> None:
        self.base_url = base_url
        self.every = every
        self.most = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stop.is_set():
            self.most = max(self.most,
                            int(bench.footprint(self.base_url).get("resident_bytes") or 0))
            self._stop.wait(self.every)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=30)
        out = bench.footprint(self.base_url)
        if self.most > int(out.get("resident_bytes") or 0):
            out["resident_bytes"] = self.most
            for derived_key in ("kv_and_run_bytes", "bytes_per_1k_context", "mmapped"):
                out.pop(derived_key, None)
            try:
                out = beyond_weights(out)
            except Exception:  # noqa: BLE001 - a summary line is never worth a run's answers
                pass
        return out


def concurrent(ask: Callable[..., Any], questions: Sequence[Mapping[str, Any]], *,
               conversations: int, turns: int, label: str, client: Any,
               graph: Mapping[str, Any] | None = None, base_url: str = "",
               log: Callable[[str], None] | None = None,
               per_question: float = PER_QUESTION,
               trace: bool | None = None) -> tuple[list[Row], dict[str, Any]]:
    """N conversations of T turns each, asked of one server at the same time.

    Every other measurement here asks one question at a time, which is right for timing a
    model and wrong for the question a server is actually asked: how many people can talk
    to it at once, and what that costs each of them. Each conversation is a chain of
    questions from the set with the earlier turns carried, run on its own thread, so N of
    them are in flight together. Per turn: the wall clock, the time until the server began
    generating, and what it spent waiting -- the wall clock less what the server itself
    reports reading and generating, which is the queueing when N exceeds the slots. For the
    run: the wall clock of the whole thing, which is not the sum of the turns, the most the
    server held while it ran, and F1 as usual, so a setting that answers faster by
    answering worse is visible.

    Returns the rows and what to keep beside them as the run's ``server``.
    """
    from concurrent.futures import ThreadPoolExecutor

    asked = [dict(q) for q in questions]
    if not asked:
        raise ValueError("no questions to converse about")
    if conversations < 1 or turns < 1:
        raise ValueError("at least one conversation of one turn")
    # each conversation takes its own stretch of the set, so two of them do not ask the
    # same question at the same moment and share a prompt cache the real thing would not
    chains = [[asked[(c * turns + t) % len(asked)] for t in range(turns)]
              for c in range(conversations)]
    # every turn of every conversation is a question, and that is what decides whether the
    # transcripts are worth their kilobytes -- see `wants_trace`
    traced = wants_trace(conversations * turns, trace)

    def one_conversation(c: int) -> list[Row]:
        prior: list[dict[str, str]] = []
        rows: list[Row] = []
        for t, question in enumerate(chains[c]):
            row, said = _ask_once(ask, question, label=label, client=client, graph=graph,
                                  turns=prior, conversation=c, turn=t,
                                  per_question=per_question, trace=traced)
            rows.append(row)
            prior += [{"role": "user", "content": row.question},
                      {"role": "assistant", "content": said}]
            if log:
                log(f"  c{c} t{t} {row.seconds:5.1f}s  first token {_clock(row.first_token)}"
                    f"  queued {_clock(row.queued)}  {row.question[:40]}")
        return rows

    slots = bench.slot_count(base_url) if base_url else -1
    watching = _Peak(base_url) if base_url else None
    began = time.time()
    with ThreadPoolExecutor(max_workers=conversations) as pool:
        got = list(pool.map(one_conversation, range(conversations)))
    wall = round(time.time() - began, 2)
    rows = [row for chain in got for row in chain]
    held = watching.stop() if watching else {}
    # None when no turn reported what the server spent: not measured is not 0
    waited = [float(r.queued) for r in rows if r.queued is not None]
    held["concurrency"] = {"conversations": conversations, "turns": turns, "slots": slots,
                           "seconds": wall, "queued": round(sum(waited), 2) if waited else None}
    return rows, held


def _clock(value: float | None) -> str:
    """``1.2s`` for a clock, ``-`` for one nothing read."""
    return f"{float(value):4.1f}s" if value is not None else "   -"


def footprint(base_url: str, client: Any = None) -> dict[str, Any]:
    """What the server holding this model costs to keep up, and what it is.

    How many conversations a machine can hold at once is decided by the KV cache, not by the
    weights: the weights are paid for once and the cache is paid for per slot per token. This
    build of llama.cpp does not print a "KV self size" line and `/props` does not carry one,
    so what is measured is the server's resident memory against the weights on disk — the
    difference is the cache and the runtime around it. Said that way rather than called KV
    exactly, because it is not exactly KV.

    ``served_by`` is on the record for every run -- the program, its version, the format,
    the runtime and the quant (`backends.served_by`) -- and the weights are its when the
    program can say; the resident figure is summed over the process tree the ``client``
    names (`serving_pids`), which is how Ollama's runner is counted beside its listener.
    """
    from ml_stack.graph.bench.backends import llama_served_by, props_of

    out: dict[str, Any] = {"base_url": base_url}
    # what the sampler saw while the questions were asked, and what it learnt of the
    # server from the client it was handed -- see `Watching.peaks`
    seen = _WATCHED.pop(http_of(base_url), {})
    pids = [int(p) for p in seen.pop("pids", ()) or ()]
    props = props_of(base_url)
    if props:
        out["model"] = str(props.get("model_path") or "").rsplit("/", 1)[-1]
        out["slots"] = int(props.get("total_slots") or 0)
        out["context"] = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
    record = (said_by(client) or seen.pop("served_by", None)
              or llama_served_by(base_url, props=props))
    if record:
        out["served_by"] = dict(record)
        if not out.get("model") and record.get("model"):
            out["model"] = str(record["model"])
        if record.get("weights_bytes"):
            out["weights_bytes"] = int(record["weights_bytes"])
    try:
        tree = process_tree(serving_pids(base_url, client) or pids)
        held = 0
        for process in tree:
            try:
                held += int(process.memory_info().rss)
            except Exception:  # noqa: BLE001 - one that has gone holds nothing
                continue
        if tree and held:
            out["resident_bytes"] = held
    except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
        pass
    # The weights come from /props, not the command line: a model served by `hf:` reference
    # is on the command line as a repository, and only the server knows where it landed.
    if "weights_bytes" not in out:
        try:
            where = Path(str(props.get("model_path") or ""))
            if where.exists():
                shards = sorted(where.parent.glob(where.name.replace("00001", "*"))) or [where]
                out["weights_bytes"] = sum(s.stat().st_size for s in shards if s.is_file())
        except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
            pass
    # Resident minus weights, when that means anything. It does not always: llama.cpp mmaps
    # the weights, so a page is resident only once it has been touched, and an MoE that uses
    # ten experts of five hundred never touches most of them. Qwen3.8-Flash-Next sat at 63G
    # resident against 87G of weights on disk -- the subtraction goes negative, clamps to
    # zero and prints as a dash, which reads as "not measured" rather than "not meaningful".
    #
    # So it is only reported when the model really is fully resident, and `resident_bytes`
    # is carried either way, because what the process actually holds is the number that
    # decides how many of them fit.
    #
    # And what it held at its most, if a `Watching` was running over this server while the
    # questions were asked: a reading taken here is taken after the last answer, when the
    # cache that was full during it has been let go. `beyond_weights` prefers the peak.
    out.update(seen)
    try:
        return beyond_weights(out)
    except Exception:  # noqa: BLE001 - a summary line is never worth a run's answers
        return out


def beyond_weights(out: dict[str, Any]) -> dict[str, Any]:
    """Split what the process holds into the weights and everything else.

    Its own function because it is the part that can be wrong without a server: reading
    `kv_and_run_bytes` when a model is mmapped raised at the *end* of a run, after every
    question had been answered, and threw away a quarter hour of GPU for a summary line.
    """
    # The peak where a run has one, because what decides how many conversations fit is what
    # the machine was asked for while it was answering, not what was left over afterwards.
    if out.get("resident_peak"):
        out["resident_bytes"] = max(int(out["resident_peak"]),
                                    int(out.get("resident_bytes") or 0))
    if "resident_bytes" in out and "weights_bytes" in out:
        beyond = out["resident_bytes"] - out["weights_bytes"]
        if beyond > 0:
            out["kv_and_run_bytes"] = beyond
        else:
            # Less resident than the weights on disk means llama.cpp mapped the file and
            # never paged all of it in -- an MoE's unused experts, most often. The
            # subtraction then says nothing about the cache, so no number is better than a
            # wrong one.
            out["mmapped"] = True
        # What one more conversation costs, which is the question a number like this is
        # asked for. Held tokens are the context times the slots holding one each; dividing
        # by them makes two models comparable however each happened to be configured.
        held = (out.get("context") or 0) * (out.get("slots") or 0)
        if held and "kv_and_run_bytes" in out:
            out["bytes_per_1k_context"] = int(out["kv_and_run_bytes"] / (held / 1024))
    return out


def _idle(url: str, args: Any) -> bool:
    """Refuse to time a server somebody else is using, unless told not to care."""
    working = bench.busy(url)
    if working <= 0:
        if working < 0:
            print(f"note: {url} would not say whether it is busy; timings may not be alone",
                  file=sys.stderr)
        return True
    print(f"error: {url} is already working on {working} request(s). A timing taken while "
          f"another run has the same GPU is not a timing.\n"
          f"       Wait for it, or pass --anyway to measure regardless.", file=sys.stderr)
    return bool(getattr(args, "anyway", False))


def busy(base_url: str) -> int:
    """How many requests that server is already working on.

    A timing taken while somebody else is using the same GPU is not a timing. This is the
    cheapest way to know: llama.cpp's /slots says what each slot is doing, and anything
    above zero means the number about to be measured belongs to two callers at once.

    -1 when the server will not say, which is not the same as idle and is not treated as it.
    """
    from ml_stack.client.http import request_json

    try:
        slots = request_json(f"{base_url.rstrip('/')}/slots", timeout=5.0, method="GET")
    except Exception:  # noqa: BLE001 - a server that will not answer is not known to be idle
        return -1
    if not isinstance(slots, list):
        return -1
    return sum(1 for one in slots if isinstance(one, Mapping) and one.get("is_processing"))


def ask_from(spec: str) -> Callable[[str, Any], Any]:
    """Import ``module:function``. It takes ``(question, client)`` and returns an Answer."""
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:function, got {spec!r}")
    from importlib import import_module

    return getattr(import_module(module), name)


def asking(graph: Mapping[str, Any], *, run: Any = None, shortlist: int = 0,
           store: str | Path | None = None,
           embed_url: str = "", embed_model: str = "", terse: bool = False,
           margin: float = MARGIN, rich: bool = False,
           tight: bool = True, reach: int | None = None, batch: bool = False,
           kinds: bool = False, summary: bool = False, single: bool = False,
           few: bool = False, rounds: int | None = None,
           constrain_ids: bool = False) -> Callable[..., Any]:
    """The ordinary way to ask this graph a question, with or without a search run first.

    ``run`` is a :class:`~ml_stack.serve.Run`, and its ``asking`` is every way below said
    once: given one, none of them is passed again. That is what stops a bench row and a
    page answer over the same model being asked two different questions.

    Nothing here is any project's: it is `converse` over the graph you handed in. Two
    choices, both about where the looking happens. Whether a cheap embedder gets to suggest
    where to look before the large model starts (``shortlist``), which is the thing most
    worth measuring. And what `look_up` is when the model calls it: with a ``store``, the
    same hybrid the application ships -- characters, the word index and, given
    ``embed_url``, vectors, fused -- and without one, characters alone. For months the bench
    had no store on this path and every ranking it wrote measured a look_up nobody ran.

    ``rich`` asks with `converse(..., rich=True)`: look_up results carry a score and why
    they matched, and a topic hit brings the people joined to it. ``tight`` is the asking,
    as it is in `converse`: show is told to light only what answers the question, and what
    is lit is capped. ``reach`` is `converse`'s too: a ceiling in tokens on what one tool
    result may carry, with look_at, look_around and list_kind packing up to it rather than
    stopping at a fixed number of entries. ``None`` is the flat character cut every run
    before it measured, so a model left on the default is measured unchanged.
    ``tight=False`` is the loose asking the ranking runs used, kept as a
    control, and it is passed on rather than left out -- left out it would be the default,
    which is now tight, and `--also loose` would have measured tight twice.

    ``batch``, ``kinds`` and ``summary`` are `converse`'s too, and each is a way a sweep
    measures: a searching tool shown a three-entry call (``batch``), look_up filtered to
    the kind the question's own words settle on (``kinds``), and the whole graph at a
    glance offered as a tool for the broad question no search reaches (``summary`` --
    `converse`'s ``summary_tool``, renamed at this hop only, because `converse`'s
    ``summary`` is a thread's rolling summary and the two must never be confused). Like
    ``rich`` and ``reach``, each is sent only when it is asked for, so a run that asked
    for none reaches `converse` byte for byte as it always did.

    ``constrain_ids`` is `converse`'s too: every turn that offers a tool taking an id
    answers under a grammar in which those ids can only be the graph's.

    ``single``, ``few`` and ``rounds`` are the same again, and they are why this list is a
    space rather than a set of defaults. ``single`` is ``batch`` turned around -- one entry
    to a read, more turns -- for the model that loses the thread of a long tool result;
    ``few`` offers three tools (look_up, look_at, show) and no other way of looking, for
    the model whose tool choice degrades with the number of schemas; ``rounds`` is how many
    tool-calling turns a question may spend, which is the thing the other two trade against
    each other. Measuring them is the only way one model gets asked with three tools and
    twenty rounds while another gets eight and six -- see `ml_stack.serve.profile`.

    The returned callable carries ``.finder`` -- see `finding` -- so a run can write down
    which one it measured, and ``.asking``, which is the way itself: every keyword above
    that changes what `converse` is asked, as `keep.save` writes it beside the rows and
    `show.asked_as` prints it. The ways lived only in the end of a run's label --
    ``...-plain-batch-kv-q8_0-rb0`` -- so nothing could group runs by them, say that two
    runs differed only in the asking, or notice a micro-batch that had been left on. A
    record is read; a suffix is guessed at.

    It also takes ``turns=`` -- the earlier turns of a conversation, as `converse` does --
    so `concurrent` can carry one on.
    """
    from ml_stack.graph.ask import converse, tools_for
    from ml_stack.graph.search import hybrid
    from ml_stack.serve.shape import Asking

    how = run.asking if run is not None else Asking(
        tight=tight, batch=batch, single=single, few=few, kinds=kinds, summary=summary,
        rich=rich, terse=terse, reach=reach or None, rounds=rounds or None,
        constrain_ids=constrain_ids)

    finder_name = finding(store, embed_url, embed_model)

    def embedded(text: str) -> list[float] | None:
        # no vectors to search means no vector to search with
        if not embed_url or finder_name != "meaning":
            return None
        from ml_stack.client.embed import embed
        from ml_stack.graph.vectors import QUERY

        try:
            return embed([QUERY + text], base_url=embed_url, model=embed_model)[0]
        except Exception:  # noqa: BLE001 - the words still vote
            return None

    def likely(question: str, held: Any) -> list[str]:
        if not shortlist:
            return []
        vector = embedded(question)
        if vector is not None and margin > 0:
            near = held.similar(vector, model=embed_model, limit=max(shortlist, 8))
            if not stands_out([r["similarity"] for r in near], margin=margin):
                return []            # nothing here stands out: "hi" is not a search
        found = hybrid(graph, question, store=held, vector=vector, model=embed_model)
        return [r["id"] for r in found][:shortlist]

    def converse_with(question: str, client: Any, finder: Any, opening: Sequence[str],
                      turns: Sequence[Mapping[str, str]]) -> Any:
        # `finder` goes to both, because `converse` swaps look_up's callable in whichever
        # tools it is handed, and the terse set is handed in rather than chosen inside
        extra: dict[str, Any] = (
            {"tools": tools_for(graph, terse=True, finder=finder, **how.tools())}
            if how.terse else {})
        # every way, in the words `converse` takes them, from the one record that holds
        # them: `summary` becomes `summary_tool`, and nothing not asked for is sent, so a
        # way that asked for none reaches `converse` byte for byte as it always did
        extra.update(how.converse())
        return converse(question, graph, client, opening=opening, finder=finder,
                        turns=list(turns), **extra)

    def ask(question: str, client: Any, *, turns: Sequence[Mapping[str, str]] = ()) -> Any:
        if store is None or not str(store):
            return converse_with(question, client, None, [], turns)
        from ml_stack.graph.store import GraphStore

        # one handle for the whole conversation: a question makes several look_ups, and
        # each is a hybrid search over the store, embedded the same way the question is
        with GraphStore(store, read_only=True) as held:
            def finder(text: str) -> list[dict[str, str]]:
                return hybrid(graph, text, store=held, vector=embedded(text),
                              model=embed_model)

            return converse_with(question, client, finder, likely(question, held), turns)

    ask.finder = finder_name  # type: ignore[attr-defined]
    # The way, said once, in the words `converse` uses. Only what was asked for: a way that
    # asked for nothing carries `{"tight": True}` and no other key, so the record says what
    # the run did rather than what every default happened to be that week.
    ask.asking = {  # type: ignore[attr-defined]
        **how.said(), **({"shortlist": int(shortlist)} if shortlist else {}),
    }
    return ask
