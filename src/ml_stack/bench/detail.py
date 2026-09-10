"""The questions behind a score: the misses, the transcript of a traced one, and the
shape of the set they came from.

`missed` prints each question with what it wanted and what it showed, `transcript` reads a
traced question back as a conversation, and `shape` says what kinds of answer a question
set asks for. The columns these lines share are in `ml_stack.bench.show`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.SHORT` -- so
# anything patchable is looked up there at call time, never bound here at import.
from ml_stack import bench
from ml_stack.bench.score import _hit, _times, band, baseline, derived, speedup
from ml_stack.bench.show import _shown, at_once, band_of, cache_turns, timeouts
from ml_stack.log import say

__all__ = ["missed", "shape", "transcript"]


def missed(kept: Sequence[Mapping[str, Any]], *, everything: bool = False,
           among: Sequence[Mapping[str, Any]] = ()) -> None:
    """Question by question: what was wanted, what the answer showed, and what it cost.

    A score is a number to act on only when you can see which questions made it. A run that
    scores 17% has failed in some particular way — no tool calls, an empty answer, the right
    people found and the wrong ones shown — and the aggregate cannot tell you which, so this
    prints the rows themselves. Only the misses by default; ``everything`` for all of them.

    ``among`` is every run kept, so a drafted run's line can say what its head was worth
    against the undrafted baseline `baseline` finds there; ``kept`` alone is one label's.
    """
    if not kept:
        say("nothing kept yet")
        return
    for one in kept:
        base = baseline(one, among or kept)
        faster = speedup(one, among or kept)
        rows = [r for r in (one.get("rows") or []) if r.get("expected")]
        shortfall = [r for r in rows if not everything and _hit(r) < 1.0] if not everything else rows
        server = one.get("server") or {}
        found = str(server.get("finder") or "-")
        together = at_once(server)
        load = server.get("load_s")
        late = timeouts(one)
        say(f"\n{one.get('label', '')}  ({one.get('at', '')}, find {found}"
            + (f", {together} at once" if together else "")
            + (f", load {float(load):.0f}s" if load is not None else "")
            + (f", {late} timed out" if late else "") + ")"
            + (f"  speedup {_times(faster)} over draft:none ({base.get('label', '')})"
                 if faster is not None and base is not None else ""))
        # what these questions could and could not settle, before the questions themselves:
        # a reader about to explain a five-point difference should see the band first
        got = derived(one)
        if band(one) is not None:
            say(f"  F1 {100 * got['right']:.0f}% {band_of(one)}, "
                f"recall {100 * got['recall']:.0f}% {band_of(one, 'recall')}, "
                f"precision {100 * got['precision']:.0f}% {band_of(one, 'precision')}, "
                f"{got['seconds_per_question']:.1f} s/q "
                f"{band_of(one, 'seconds_per_question', unit='s')}"
                f"   (95% over its own {got['questions']:.0f} questions)")
        if not shortfall:
            say("  every question answered in full")
            continue
        for r in shortfall:
            got, want = set(r.get("shown") or ()), set(r.get("expected") or ())
            say(f"  {_hit(r) * 100:3.0f}%  {r.get('question', '')}"
                + (f"   [timed out at {float(r.get('seconds') or 0):.0f}s]"
                     if r.get("timed_out") else ""))
            say(f"        wanted  {', '.join(sorted(want)) or '-'}")
            say(f"        showed  {', '.join(sorted(got)) or '(nothing)'}")
            if want - got:
                say(f"        missed  {', '.join(sorted(want - got))}")
            if r.get("unread"):
                # what F1 cannot see: a name in the prose that no tool call produced
                say(f"        made    {', '.join(r['unread'])}  (named, never found or read)")
            turns = cache_turns(r)
            note = (f"{r.get('calls', 0)} calls, {r.get('answer_chars', 0)} chars"
                    + (f", {turns}" if turns else "")
                    + (f", ERROR {r['error']}" if r.get("error") else ""))
            if together:
                note += (f"; conversation {r.get('conversation', 0)} turn {r.get('turn', 0)}, "
                         f"first token {_s(r.get('first_token'))}, "
                         f"queued {_s(r.get('queued'))}")
            say(f"        {note}")


def _s(value: Any) -> str:
    """``1.2s``, or ``-`` for a clock nothing read."""
    return f"{float(value):.1f}s" if value is not None else "-"


def _args_line(args: Mapping[str, Any], width: int = 60) -> str:
    """A tool call's arguments on one line, cut to ``width``."""
    text = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= width else text[:width - 1] + "…"


def transcript(kept: Sequence[Mapping[str, Any]], label: str = "",
               question: str = "") -> None:
    """A traced question read back as a conversation: one line per call, in order.

    The table says a question took three calls; this says which three. Per call: the round,
    the tool and the arguments it was called with, how much came back and how many entries
    were in it, what the server spent reading and writing, and how much of the draft head's
    guessing was accepted -- the numbers `Spent` totals per answer, before they are totalled.

    It is where a wrong answer is diagnosed and where a training example is read before
    thousands of them are written: `ml-stack-train-tools from-bench` turns exactly these
    turns into a dataset, and a turn that reads wrong here trains wrong there.

    ``label`` narrows to the runs whose label contains it, ``question`` to the questions
    whose text contains it. Only rows that were traced -- see `wants_trace`.
    """
    from ml_stack.bench.measure import TRACE_ENV

    wanted = [one for one in kept if not label or label in str(one.get("label") or "")]
    shown = 0
    for one in wanted:
        rows = [r for r in (one.get("rows") or []) if r.get("trace")
                and (not question or question.casefold() in str(r.get("question") or "").casefold())]
        for r in rows:
            shown += 1
            say(f"\n{one.get('label', '')}  {r.get('question', '')}"
                f"   ({_hit(r) * 100:.0f}%, {r.get('calls', 0)} calls, "
                f"{float(r.get('seconds') or 0):.1f}s"
                + (", TIMED OUT" if r.get("timed_out") else "")
                + (f", ERROR {r['error']}" if r.get("error") else "") + ")")
            for entry in r.get("trace") or []:
                for line in _trace_lines(entry):
                    say(f"  {line}")
    if not shown:
        say(f"no traced question found"
            + (f" for {label!r}" if label else "")
            + (f" matching {question!r}" if question else "")
            + f". A run of {bench.SHORT} questions or fewer traces by default; "
                f"{TRACE_ENV}=1 traces one of any size.")


def _trace_lines(entry: Mapping[str, Any]) -> list[str]:
    """One trace entry as the lines `transcript` prints for it."""
    role = str(entry.get("role") or "")
    if role == "tools":
        names = [str((t.get("function") or {}).get("name") or t.get("name") or "")
                 for t in (entry.get("tools") or ())]
        return [f"     tools  {', '.join(n for n in names if n)}"]
    if role == "tool":
        return [f"     <-  {entry.get('name', '?')}  {int(entry.get('chars') or 0)} chars, "
                f"{int(entry.get('ids') or 0)} ids"
                + ("  (cut)" if entry.get("cut") else "")]
    if role != "assistant":
        return [f"     {role:6} {_shown(str(entry.get('content') or '').replace(chr(10), ' '), 88)}"]
    timings = entry.get("timings") or {}
    read = (f"read {int(timings.get('prompt_n') or 0)}"
            f"+{int(timings.get('cache_n') or 0)} cached in "
            f"{float(timings.get('prompt_ms') or 0):.0f}ms")
    wrote = (f"wrote {int(timings.get('predicted_n') or 0)} in "
             f"{float(timings.get('predicted_ms') or 0):.0f}ms")
    drafted = ""
    if int(timings.get("draft_n") or 0):
        taken, guessed = int(timings.get("draft_n_accepted") or 0), int(timings["draft_n"])
        drafted = f", accepted {taken}/{guessed}"
        if timings.get("draft_ms") is not None:
            drafted += (f" ({float(timings['draft_ms']):.0f}ms drafting"
                        f" + {float(timings.get('verify_ms') or 0):.0f}ms checking"
                        f" over {int(timings.get('verify_n') or 0)} pass(es))")
    head = f"{int(entry.get('call') or 0):3}"
    lines = []
    for call in entry.get("tool_calls") or []:
        lines.append(f"{head}  -> {call.get('name', '?')}({_args_line(call.get('args') or {})})")
        head = "   "
    if not lines:
        lines.append(f"{head}  -- answered {int(entry.get('chars') or 0)} chars")
    thought = int(entry.get("thinking_chars") or 0)
    lines.append(f"        {read}, {wrote}{drafted}"
                 + (f", thought {thought} chars" if thought else "")
                 + (f", {entry['finish']}" if entry.get("finish") else ""))
    return lines


def shape(questions: Sequence[Mapping[str, Any]], graph: Mapping[str, Any]) -> None:
    """What a question set is made of, so its bias is visible without counting by hand.

    A set that is nine-tenths person-shaped rewards anything that prefers people, whether
    or not that rule is right. That was the state of this one, and it flattered a filter
    measured against it -- so the shape of the set is printed rather than assumed.
    """
    kinds = {str(n.get("id")): str(n.get("kind") or "") for n in (graph.get("nodes") or ())}
    scored = [q for q in questions if q.get("expect")]
    if not scored:
        say("no scored questions")
        return
    counted: dict[str, int] = {}
    for q in scored:
        for kind in {kinds.get(str(e), "?") for e in q["expect"]}:
            counted[kind] = counted.get(kind, 0) + 1
    peopleless = sum(1 for q in scored
                     if not any(kinds.get(str(e)) == "person" for e in q["expect"]))
    say(f"{len(questions)} questions, {len(scored)} scored, "
        f"{len(questions) - len(scored)} whose right answer is nobody")
    say(f"graph: {len(graph.get('nodes') or ())} entries, "
        f"{len(graph.get('edges') or ())} links")
    for kind, n in sorted(counted.items(), key=lambda kv: -kv[1]):
        say(f"  {n:>3} question(s) want a {kind}")
    say(f"  {peopleless:>3} question(s) want no person at all "
        f"({100 * peopleless / len(scored):.0f}%)")
    say(f"mean entries expected: {sum(len(q['expect']) for q in scored) / len(scored):.1f}")
    missing = sorted({str(e) for q in scored for e in q["expect"] if str(e) not in kinds})
    if missing:
        say(f"\nEXPECTED IDS THAT DO NOT EXIST IN THE GRAPH: {missing}")
