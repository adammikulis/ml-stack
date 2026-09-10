"""One question, answered with the graph in hand: the turn-by-turn loop, what a turn is
told between turns, and the note a model writes from an answer.

The graph is a mapping with ``nodes`` and ``edges``; nothing here cares what a project calls
its kinds or relations.

    reply = converse("how are Ada and Bea connected?", graph, client)
    reply.content   # what to say
    reply.ids       # what to light up
    reply.steps     # what it did to find out
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ml_stack.asking import ASKING, Asking
from ml_stack.graph.answers import Answer, noted, recorded, result_count, selected, without_images
from ml_stack.graph.grammar import CAP, call_from, call_schema, response_format
from ml_stack.graph.looking import FOUND, cut, enriched, look_at, tools_for
from ml_stack.graph.prompts import (
    ALREADY,
    ANSWER_NOW,
    BATCH_NUDGE,
    BATCH_SYSTEM_SENTENCE,
    CITE_SYSTEM_SENTENCE,
    CONSTRAINED_SYSTEM_SENTENCE,
    DRAFT_SYSTEM,
    EARLIER,
    FEW_PATH_CLAUSE,
    FEW_SYSTEM_SENTENCE,
    FROM_THE_NOTES,
    GO_AND_LOOK,
    NO_ANSWER,
    OVER,
    PATH_CLAUSE,
    RECALLED,
    SEARCHING,
    SHORTLIST,
    SHOW_NUDGE,
    SHOW_PARAGRAPH,
    SINGLE_NUDGE,
    SINGLE_SYSTEM_SENTENCE,
    SYSTEM,
    TIGHT_NUDGE,
    TIGHT_SHOW_PARAGRAPH,
    TIGHT_SYSTEM_SENTENCE,
    as_asked,
)
from ml_stack.graph.replies import all_at_once, is_working, one_by_one, spoken_show, without_notes

# How many tool calls a question may spend. Five was enough to look two names up and read
# them; it is not enough to staff a project, which means looking up each skill, reading the
# people behind them, and then saying which of them the answer is about. Measured against a
# real graph: every staffing question spent all five on searching and never reached `show`.
ROUNDS = 10
# How many times a model may ask for something it has already asked for before the searching
# is ended. Two is a stumble; three is a loop, and a loop always ends in no answer at all.
REPEATS = 2
# How many entries one answer may light.
LIT = 25


def converse(question: str, graph: Mapping[str, Any], client: Any, *,
             asking: Asking = ASKING,
             turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
             limit: int = LIT,
             tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
             finder: Any = None, highlighted: Sequence[str] = (),
             opening: Sequence[str] = (), summary: Any = None,
             recalled: Sequence[Any] = ()) -> Answer:
    """One question, answered with the graph in hand.

    ``asking`` is the :class:`~ml_stack.graph.Asking` -- every asking flag, in one
    record; `serve.profile.asking_for` reads a model's measured one.

    ``client`` is anything with ``chat(messages, tools=...)`` returning a reply carrying
    ``content`` and ``tool_calls``. ``tools`` is ``[(schema, callable), ...]``, each
    callable taking the parsed arguments mapping; ``tools_for(graph)`` by default.
    ``finder`` replaces just look_up's callable. ``highlighted`` names entries already lit
    for the reader: they are told to the model by label and id, and enter ``ids`` only if a
    tool call touches them. ``limit`` caps what ``show`` may light.

    ``turns`` is the window: the last few turns, sent whole and in order. ``summary`` -- a
    thread's rolling summary, as a ``Turn``, a mapping with ``text`` or ``content``, or a
    string -- goes first after the system prompt. ``recalled`` -- earlier turns outside the
    window that match this question, oldest first, the same shapes -- follows it.
    ``opening`` names entries a cheap search already found, read out before the first turn.
    """
    return _converse(question, graph, client, asking=asking, turns=turns, system=system,
                     limit=limit, tools=tools, finder=finder, highlighted=highlighted, emit=None,
                     opening=opening, summary=summary, recalled=recalled)


def converse_stream(question: str, graph: Mapping[str, Any], client: Any, *,
                    on_event: Any, asking: Asking = ASKING,
                    turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
                    limit: int = LIT,
                    tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
                    finder: Any = None, highlighted: Sequence[str] = (),
                    opening: Sequence[str] = (), summary: Any = None,
                    recalled: Sequence[Any] = ()) -> Answer:
    """converse, reporting what is happening to ``on_event`` as it happens.

    ``on_event`` gets one mapping per event: ``{"event": "thinking", "text"}`` as the
    model thinks, ``{"event": "tool", "name", "detail"}`` when it calls a tool,
    ``{"event": "tool_result", "name", "count"}`` with how much came back,
    ``{"event": "answer", "text"}`` as the answer arrives, then ``{"event": "done"}``.
    A client whose ``chat`` takes ``on_delta`` streams the text a piece at a time;
    any other client's text arrives whole.
    """
    return _converse(question, graph, client, asking=asking, turns=turns, system=system,
                     limit=limit, tools=tools, finder=finder, highlighted=highlighted, emit=on_event,
                     opening=opening, summary=summary, recalled=recalled)


def _call_detail(name: str, args: Mapping[str, Any]) -> str:
    if name == "look_up":
        return repr(str(args.get("text") or ""))
    if name in ("look_at", "show", "quote"):
        ids = list(args.get("ids") or ())
        return f"{len(ids)} id" + ("" if len(ids) == 1 else "s")
    if name == "look_around":
        ids = list(args.get("ids") or ())
        hops = int(args.get("hops") or 1)
        return (f"{len(ids)} id" + ("" if len(ids) == 1 else "s")
                + ("" if hops == 1 else f", {hops} hops"))
    if name == "path_between":
        return f"{args.get('from_id')} → {args.get('to_id')}"
    if name == "list_kind":
        return repr(str(args.get("kind") or ""))
    return json.dumps(args, ensure_ascii=False)[:80]


def _plain(value: Any) -> str:
    """What the JSON encoder says about a value it cannot encode, instead of raising.

    A tool result is whatever a caller's tool returned, and one that carries raw bytes --
    a picture the `_images` convention did not cover, a path -- must not take the whole
    answer down with a TypeError. The bytes themselves are never sent: a model reads
    nothing from them and they are the size of a picture.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(value)} bytes>"
    return repr(value)[:80]


SAID_CHARS_BACK = 4000


def _spoken(turn: Any) -> tuple[str, str]:
    """``(role, text)`` of a turn given as a ``Turn``, a mapping, or bare text."""
    if turn is None:
        return "user", ""
    if isinstance(turn, str):
        return "user", turn
    if isinstance(turn, Mapping):
        role = turn.get("role")
        text = turn.get("content") if turn.get("content") is not None else turn.get("text")
    else:
        role = getattr(turn, "role", None)
        text = getattr(turn, "text", None)
    return ("assistant" if role == "assistant" else "user"), str(text or "")


def _under(offer: Sequence[Mapping[str, Any]], ids_known: Sequence[str],
           constrain_ids: bool) -> dict[str, Any] | None:
    """The response_format a turn offering ``offer`` answers under, if it is constrained."""
    if not constrain_ids or not offer:
        return None
    schema = call_schema(offer, ids_known)
    return response_format(schema) if schema is not None else None


def _read_back(reply: Any, constrain_ids: bool) -> Any:
    """A constrained reply's JSON as the tool call or the prose it holds."""
    if not constrain_ids or (getattr(reply, "tool_calls", None) or []):
        return reply
    calls, answer = call_from(getattr(reply, "content", "") or "")
    if calls:
        return replace(reply, content="", tool_calls=calls)
    if answer is not None:
        return replace(reply, content=answer)
    return reply


def _searched(reply: Any) -> bool:
    """Whether that reply asked for anything other than showing."""
    return any((call.get("function") or {}).get("name") in SEARCHING
               for call in (getattr(reply, "tool_calls", None) or []))


def _finding(graph: Mapping[str, Any], finder: Any, *, rich: bool) -> Any:
    """look_up over a caller's own finder: one word or several, in one call."""
    def found(args: Mapping[str, Any]) -> Any:
        wanted = [str(x) for x in (args.get("texts") or ()) if str(x).strip()]
        if not wanted and str(args.get("text") or "").strip():
            wanted = [str(args["text"])]
        rows, seen = [], set()
        for text in wanted:
            for r in finder(text):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
        return enriched(graph, rows) if rich else rows

    return found


def _offer(graph: Mapping[str, Any], tools: Sequence[tuple[Mapping[str, Any], Any]] | None,
           *, finder: Any, asking: Asking) -> list[tuple[Mapping[str, Any], Any]]:
    """The ``(schema, callable)`` pairs one question is answered with."""
    if tools is None:
        return tools_for(graph, asking=asking, finder=finder, cite=asking.cite)
    if finder is not None:
        instead = _finding(graph, finder, rich=asking.rich)
        tools = [(schema, fn) if (schema.get("function") or {}).get("name") != "look_up"
                 else (schema, instead)
                 for schema, fn in tools]
    # A caller's own tools are described on a copy, never in place, and `few` takes some
    # away, so the callables are matched back by name -- a caller's own acting tool keeps
    # the function it came with. `rich` is off: it says what look_up's own hits carry, and
    # a caller's look_up returns whatever it returns.
    told = {str((schema.get("function") or {}).get("name") or ""): schema
            for schema in as_asked([schema for schema, _ in tools],
                                 replace(asking, rich=False))}
    return [(told[name], fn) for schema, fn in tools
            if (name := str((schema.get("function") or {}).get("name") or "")) in told]


def _telling(system: str, *, asking: Asking, known: set[str], graph: Mapping[str, Any],
             highlighted: Sequence[str], out: Answer) -> tuple[str, bool]:
    """The system prompt as this asking says it, and whether ids are held to the graph's.

    Constraining is dropped over `CAP` ids, which is written into the answer's steps.
    """
    if asking.tight:
        system = (system.replace(SHOW_PARAGRAPH, TIGHT_SHOW_PARAGRAPH)
                  + " " + TIGHT_SYSTEM_SENTENCE)
    if asking.cite:
        system = system + "\n\n" + CITE_SYSTEM_SENTENCE
    if asking.batch:
        system = system + "\n\n" + BATCH_SYSTEM_SENTENCE
    if asking.single:
        system = system + "\n\n" + SINGLE_SYSTEM_SENTENCE
    if asking.few:
        # the prompt names `path_between` in its second sentence, and this offer has no
        # such tool: swapped for what to do instead, never left to be reached for
        system = system.replace(PATH_CLAUSE, FEW_PATH_CLAUSE) + "\n\n" + FEW_SYSTEM_SENTENCE
    constrain_ids = asking.constrain_ids
    if constrain_ids and len(known) > CAP:
        out.steps.append(f"ids not constrained: {len(known)} entries, over the cap of {CAP}")
        constrain_ids = False
    if constrain_ids:
        system = system + "\n\n" + CONSTRAINED_SYSTEM_SENTENCE
    lit = [str(h) for h in highlighted if str(h) in known]
    if lit:
        label = {str(n["id"]): str(n.get("label") or "") for n in (graph.get("nodes") or ())}
        system = (system + "\n\nCurrently highlighted: "
                  + ", ".join(f"{label[h]} ({h})" for h in lit))
    return system, constrain_ids


def _remembered(turns: Sequence[Mapping[str, str]], summary: Any, recalled: Sequence[Any],
                out: Answer) -> list[dict[str, Any]]:
    """The messages before the question: the rolling summary, what was recalled, then the
    window, each counted against the part of the prompt it filled.

    In that order. The summary is the thing that changes least, so it goes first and the
    cached prefix covers it; the window is the thing that must be complete, so nothing
    here is taken out of it.
    """
    ahead: list[dict[str, Any]] = []
    _, told = _spoken(summary)
    if told.strip():
        ahead.append({"role": "user", "content": EARLIER + " ".join(told.split())})
    for one in recalled:
        role, text = _spoken(one)
        if text.strip():
            ahead.append({"role": role, "content": RECALLED + text[:SAID_CHARS_BACK]})
    said = [{"role": ("assistant" if t.get("role") == "assistant" else "user"),
             "content": str(t.get("content") or "")[:SAID_CHARS_BACK]}
            for t in turns if str(t.get("content") or "").strip()]
    for one in ahead:
        text = str(one.get("content") or "")
        out.spent.part("summary" if text.startswith(EARLIER) else "recalled", text)
    for one in said:
        out.spent.part("window", one.get("content"))
    return [*ahead, *said]


@dataclass
class _Loop:
    """One question in flight: what has been sent, what is being built, what was asked for."""

    client: Any
    graph: Mapping[str, Any]
    emit: Any
    schemas: list[Mapping[str, Any]]
    run: dict[str, Any]
    known: set[str]
    ids_known: list[str]
    constrain_ids: bool
    reach: int | None
    out: Answer
    messages: list[dict[str, Any]]
    called: bool = False
    repeats: int = 0
    asked_for: set[tuple[str, str]] = field(default_factory=set)

    def say(self, *, spoken: bool = True) -> Any:
        """One model turn over the messages so far, offered the same tools as every other.

        ``spoken`` streams the reply to ``emit``; off, the reply comes back unannounced.
        """
        kw: dict[str, Any] = {"tools": self.schemas} if self.schemas else {}
        form = _under(self.schemas, self.ids_known, self.constrain_ids)
        if form is not None:
            kw["response_format"] = form
        sent = time.monotonic()
        if self.emit is None or not spoken:
            reply = self.client.chat(self.messages, think=False, **kw)
            self.out.spent.note(reply, time.monotonic() - sent)
            return _read_back(reply, self.constrain_ids)
        streamed = {"thinking": False, "answer": False}

        def on_delta(kind: str, text: str) -> None:
            if not text:
                return
            name = "thinking" if kind == "thinking" else "answer"
            streamed[name] = True
            self.emit({"event": name, "text": text})

        # a constrained turn is JSON as it streams and prose only once read back, so it
        # arrives whole
        if form is None:
            reply = self.client.chat(self.messages, think=False, on_delta=on_delta, **kw)
        else:
            reply = self.client.chat(self.messages, think=False, **kw)
        self.out.spent.note(reply, time.monotonic() - sent)
        reply = _read_back(reply, self.constrain_ids)
        if not streamed["thinking"]:
            trace = (getattr(reply, "thinking", "") or "").strip()
            if trace:
                self.emit({"event": "thinking", "text": trace})
        if not streamed["answer"] and not (getattr(reply, "tool_calls", None) or []):
            whole = (getattr(reply, "content", "") or "")
            if whole.strip():
                self.emit({"event": "answer", "text": whole})
        return reply

    def act(self, reply: Any, *, searching_over: bool = False) -> bool:
        """Run whatever the reply asked for. False when it asked for nothing.

        With ``searching_over`` a call to a searching tool is refused instead of run: its
        tool message says so, and the tools that act still run.
        """
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            return False
        self.called = True
        # One round, however many calls it carried. A reply may ask for several at once --
        # llama-server returns them in one message -- and every one of them is run below,
        # in order, each answered with a tool message of its own. A reply refused whole
        # spent nothing and is no round.
        if not (searching_over and all((c.get("function") or {}).get("name") in SEARCHING
                                       for c in calls)):
            self.out.rounds += 1
        self.messages.append({"role": "assistant", "content": reply.content or "",
                              "tool_calls": calls})
        for call in calls:
            self._one(call, searching_over=searching_over)
        return True

    def _one(self, call: Mapping[str, Any], *, searching_over: bool) -> None:
        """Run one call and answer it with a tool message."""
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        if self.emit is not None:
            self.emit({"event": "tool", "name": name, "detail": _call_detail(name, args)})
        do = self.run.get(name)
        again = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if do is None:
            result: Any = {"error": f"no such tool: {name}"}
        elif searching_over and name in SEARCHING:
            result = {"refused": OVER}
            self.out.steps.append(f"refused {name}: the searching is over")
        elif again in self.asked_for:
            # Asking the same thing twice is how a budget disappears: measured against a
            # real graph, one question spent six of its ten rounds looking up one word,
            # over and over, and never answered. Being told does not stop it -- the tools
            # are refused after `REPEATS`, which does.
            self.repeats += 1
            result = {"already": ALREADY}
            self.out.steps.append(f"asked {name} the same thing again")
        else:
            result = recorded(self.out, name, args, do(args), self.known)
        self.asked_for.add(again)
        result, seen, kept = without_images(name, result, self.out)
        count = result_count(result)
        if kept and isinstance(result, Mapping) and not ({"entries", "path"} & set(result)):
            count = kept
        if self.emit is not None:
            self.emit({"event": "tool_result", "name": name, "count": count})
        self.messages.append({"role": "tool", "tool_call_id": call.get("id") or name,
                              "name": name,
                              "content": cut(json.dumps(result, ensure_ascii=False,
                                                         default=_plain), self.reach)})
        self.out.spent.part("tool_results", self.messages[-1]["content"])
        if seen is not None:
            self.messages.append(seen)

    def settle(self, reply: Any) -> Any:
        """The last turn's reply, after what it asked for: a search is refused once and
        the tools that act are run; the reply that follows is returned."""
        if self.act(reply, searching_over=True):
            reply = self.say()
            if not _searched(reply) and self.act(reply, searching_over=True):
                reply = self.say()
        return reply


def _shortlist(loop: _Loop, opening: Sequence[str]) -> list[str]:
    """The entries a cheap search already found, read out before the first turn.

    Before the question, as candidates to check, and never after it. Measured on
    gemma-4-E4B: eight likely entries handed over as the last message after the question,
    phrased "use them if they answer it", took it from 58% F1 to 33% -- it echoed the list
    rather than selecting from it.
    """
    start = [str(i) for i in opening if str(i) in loop.known][:FOUND]
    if not start:
        return start
    do = loop.run.get("look_at") or (lambda a: look_at(loop.graph, a["ids"]))
    material = do({"ids": start})
    noted(loop.out.found, start, loop.known)
    loop.out.steps.append(f"was handed {len(start)} to start from")
    loop.messages.append({"role": "user", "content": SHORTLIST + str(material)})
    loop.out.spent.part("shortlist", material)
    if loop.emit is not None:
        loop.emit({"event": "tool", "name": "shortlist",
                   "detail": f"{len(start)} to start from"})
    return start


def _nudge(loop: _Loop, reply: Any, asking: Asking, nudged: dict[str, bool]) -> None:
    """Ask a turn to read the way the asking wants, once each.

    ``batch`` when it read one entry and left the rest of what it found unread: the round
    it is about to spend on the second one buys nothing the first call could not have
    carried. ``single`` from the other end, when it read several at once. Said once -- a
    model that ignores it twice is not going to be told into it, and the reminder costs a
    message in every prompt after it.
    """
    if (asking.batch and not nudged["batch"] and one_by_one(reply)
            and any(i not in loop.out.read for i in loop.out.found)):
        nudged["batch"] = True
        loop.messages.append({"role": "user", "content": BATCH_NUDGE})
        loop.out.spent.part("question", BATCH_NUDGE)
        loop.out.steps.append("asked it to read the rest in one call")
    if asking.single and not nudged["single"] and all_at_once(reply):
        nudged["single"] = True
        loop.messages.append({"role": "user", "content": SINGLE_NUDGE})
        loop.out.spent.part("question", SINGLE_NUDGE)
        loop.out.steps.append("asked it to read one at a time")


def _search(loop: _Loop, asking: Asking, rounds: int) -> tuple[Any, bool]:
    """Turn after turn until the model stops calling tools, says what to light, goes in
    circles or runs out: ``(the last reply, whether that reply is the answer)``."""
    reply, answered = None, False
    nudged = {"batch": False, "single": False}
    for _ in range(rounds):
        if loop.repeats >= REPEATS:
            # going in circles: the loop ends here and the answer is asked for below
            loop.out.steps.append("stopped searching in circles")
            break
        reply = loop.say()
        if not loop.act(reply):
            answered = True          # it stopped calling tools, so this reply is the answer
            break
        _nudge(loop, reply, asking, nudged)
        # Once it has said what its answer is about, more searching cannot improve that --
        # `show` is the last thing a turn does, and a round after it is a round trip spent
        # to be told the same. Only when the round did nothing else: a turn that showed and
        # kept looking in the same breath has not finished looking.
        if loop.out.show and not _searched(reply):
            loop.out.steps.append("said what to light, so the searching stopped")
            break
    return reply, answered


def _last_call(loop: _Loop) -> str:
    """What a turn that said nothing is told: answer now, and for one that only ever
    searched, what the graph holds on its top finds read out first."""
    out = loop.out
    if out.read or not out.found:
        return ANSWER_NOW
    top = out.found[:FOUND]
    do = loop.run.get("look_at")
    material = (do({"ids": top}) if do is not None else look_at(loop.graph, top)) or ""
    noted(out.read, top, loop.known)
    out.steps.append(f"read the top {len(top)} find" + ("" if len(top) == 1 else "s"))
    if loop.emit is not None:
        loop.emit({"event": "tool", "name": "look_at", "detail": f"{len(top)} ids"})
        loop.emit({"event": "tool_result", "name": "look_at",
                   "count": result_count(material)})
    return "What the graph holds on what you found:\n" + str(material) + "\n\n" + ANSWER_NOW


def _finish(loop: _Loop, reply: Any, answered: bool) -> Any:
    """The reply the answer is read out of.

    The searching is over, one way or another, and the turn is told so: the tools that act
    -- saying what to light, asking for the graph to be changed -- are run, not ignored. A
    thinking model can stop calling tools and still say nothing, and it can run out of
    rounds the same way; either silence gets one plain instruction to answer.
    """
    if loop.called and not answered:
        loop.messages.append({"role": "user", "content": OVER})
        loop.out.spent.part("question", OVER)
        reply = loop.settle(loop.say())
    if loop.called and not (getattr(reply, "content", "") or "").strip():
        loop.messages.append({"role": "user", "content": _last_call(loop)})
        reply = loop.say()
    return reply


def _from_the_notes(loop: _Loop, reply: Any) -> Any:
    """One more turn, offered the model's own thinking back.

    Thinking is a scratchpad -- "Actually the look_at shows... need to check X" -- and
    showing it as the answer reads as a broken machine. Only prose that reads like an
    answer is used, never the working itself.
    """
    trace = (getattr(reply, "thinking", "") or "").strip()
    if not (trace or loop.messages[-1].get("role") == "assistant"):
        return reply
    loop.messages.append({"role": "user", "content": FROM_THE_NOTES})
    reply = loop.say(spoken=False)
    said = (getattr(reply, "content", "") or "").strip()
    loop.out.content = "" if is_working(said) else without_notes(said)
    if loop.out.content and loop.emit is not None:
        loop.emit({"event": "answer", "text": loop.out.content})
    return reply


def _gave_up(loop: _Loop, reply: Any) -> None:
    """Say there is no answer, and why, from the last reply.

    The assumption is always the token budget and it is almost never that. Measured: the
    same failing question at n_predict 2048 and 6144 came back `finish_reason: stop` both
    times, with 628 and 767 characters of reasoning and nothing after it.
    """
    why = getattr(reply, "finish_reason", None) or "unknown"
    thought = len((getattr(reply, "thinking", "") or "").strip())
    wrote = len((getattr(reply, "content", "") or "").strip())
    loop.out.steps.append(f"no answer: finish_reason={why}, thinking {thought} chars, "
                          f"answer {wrote} chars")
    loop.out.content = NO_ANSWER
    if loop.emit is not None:
        loop.emit({"event": "answer", "text": NO_ANSWER})


def _worded(loop: _Loop, reply: Any) -> None:
    """Read the answer out of the reply: a written-out `show` taken off it, the planning
    trimmed from the front, and one more turn asked for when what is left is only notes."""
    out = loop.out
    out.content = (getattr(reply, "content", "") or "").strip()
    if out.content:
        out.content, meant = spoken_show(out.content)
        if meant:
            # what it wrote down is what it meant, and a tighter set than asking again
            noted(out.show, meant, loop.known)
            out.steps.append("said what to light in words, so it was not asked again")
    if out.content and not is_working(out.content):
        trimmed = without_notes(out.content)
        if trimmed != out.content:
            out.steps.append("trimmed the notes off the front of the answer")
            out.content = trimmed
    if out.content and is_working(out.content):
        # It answered with its notes. Same remedy as saying nothing at all -- the notes
        # are the material, and what is wanted is the answer they were working towards.
        out.steps.append("answered with its notes, so was asked again")
        loop.messages.append({"role": "assistant", "content": out.content})
        out.content = ""
    if not out.content:
        reply = _from_the_notes(loop, reply)
    if not out.content:
        _gave_up(loop, reply)


def _sent_to_look(loop: _Loop) -> None:
    """One more chance for a model that answered without touching the graph.

    It answered from nothing, and nothing is what it knows: this graph was not in its
    training data. Measured over the invented community, six of gemma-4-E4B's nine
    failures were exactly this -- two model calls, a hundred characters of prose, no
    search. If it declines again, its answer stands as it is.
    """
    out = loop.out
    loop.messages.append({"role": "assistant", "content": out.content})
    loop.messages.append({"role": "user", "content": GO_AND_LOOK})
    if loop.emit is not None:
        # a reader watching this stream has the first answer on screen already, and the
        # page appends what arrives; without being told to start over it would show the
        # two answers run together
        loop.emit({"event": "restart", "why": "answered without searching"})
    reply = loop.say()
    if loop.act(reply):
        reply = loop.settle(loop.say())
    said = (getattr(reply, "content", "") or "").strip()
    if said:
        # whatever it said last is what the reader is looking at, so it is the answer
        out.content, meant = spoken_show(said)
        noted(out.show, meant, loop.known)
    if out.read or out.found or out.path:
        out.steps.append("answered without looking, so it was sent to look")


def _asked_to_show(loop: _Loop, *, tight: bool) -> None:
    """Ask outright what the answer is about, once, with the answer in front of it.

    A turn that spends its budget searching never reaches `show` on its own -- measured
    against a real graph, not one staffing question in six did. The tools offered are the
    same as on every other call, and only a `show` it calls is read.
    """
    out = loop.out
    loop.messages.append({"role": "assistant", "content": out.content})
    loop.messages.append({"role": "user", "content": TIGHT_NUDGE if tight else SHOW_NUDGE})
    last = loop.say(spoken=False)
    for call in (getattr(last, "tool_calls", None) or []):
        fn = call.get("function") or {}
        if (fn.get("name") or "") != "show":
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            continue
        noted(out.show, [str(i) for i in (args.get("ids") or ())], loop.known)
    if out.show:
        out.steps.append(f"selected {len(out.show)} entr"
                         + ("y" if len(out.show) == 1 else "ies"))


def _converse(question: str, graph: Mapping[str, Any], client: Any, *, asking: Asking,
              turns: Sequence[Mapping[str, str]], system: str, limit: int,
              tools: Sequence[tuple[Mapping[str, Any], Any]] | None,
              finder: Any, highlighted: Sequence[str], emit: Any, opening: Sequence[str] = (),
              summary: Any = None, recalled: Sequence[Any] = ()) -> Answer:
    out = Answer()
    offer = _offer(graph, tools, finder=finder, asking=asking)
    known = {str(n["id"]) for n in (graph.get("nodes") or ())}
    system, constrain_ids = _telling(system, asking=asking, known=known, graph=graph,
                                     highlighted=highlighted, out=out)
    schemas = [schema for schema, _ in offer]
    loop = _Loop(client=client, graph=graph, emit=emit, schemas=schemas,
                 run={str((schema.get("function") or {}).get("name") or ""): fn
                      for schema, fn in offer},
                 known=known, ids_known=sorted(known), constrain_ids=constrain_ids,
                 reach=asking.reach, out=out,
                 messages=[{"role": "system", "content": system}])
    # what filled the prompt, by part -- with a window, a summary and recall the length of
    # the conversation says nothing about what the slot holds; this does, and `spent`
    # keeps the server's exact count beside it
    out.spent.part("system", system)
    out.spent.part("tools", json.dumps(schemas))
    loop.messages += _remembered(turns, summary, recalled, out)
    start = _shortlist(loop, opening)
    loop.messages.append({"role": "user", "content": question})
    out.spent.part("question", question)
    rounds = ROUNDS if asking.rounds is None else int(asking.rounds)
    reply, answered = _search(loop, asking, rounds)
    _worded(loop, _finish(loop, reply, answered))
    if out.content and not (out.read or out.found or out.path) and "look_up" in loop.run:
        _sent_to_look(loop)
    if out.content and not out.show and (out.read or out.found or out.path):
        _asked_to_show(loop, tight=asking.tight)
    if (asking.tight or asking.kinds) and out.show:
        selected(out, graph, question, asking, start)
    for one in (*out.read, *out.path, *out.found):
        if one not in out.ids:
            out.ids.append(one)
    out.ids = out.ids[:limit]
    out.show = out.show[:limit]
    if emit is not None:
        emit({"event": "done"})
    return out


#: the most entries one note introduces
DRAFTED = 6


def draft(ids: Sequence[str], question: str, answer: str, graph: Mapping[str, Any],
          client: Any, *, system: str = DRAFT_SYSTEM, most: int = DRAFTED) -> dict[str, Any]:
    """A note introducing ``ids`` to each other, for a person to send.

    Returns ``{"text": note, "ids": [the ones written about]}``, or ``{"text": "", "why":
    reason}`` when fewer than two of the ids are in the graph or the model did not finish
    a note. The model is given the entries the answer was about, not the whole graph.
    """
    known = {str(n["id"]) for n in graph.get("nodes") or ()}
    wanted = [i for i in ids if i in known][:most]
    if len(wanted) < 2:
        return {"text": "", "why": "an introduction needs at least two entries"}
    material = look_at(graph, wanted) or ""
    said = [{"role": "system", "content": system},
            {"role": "user", "content":
             f"The question was: {question}\n\nWhat was said in reply:\n{answer}\n\n"
             f"What the graph holds on each of them:\n{material}\n\n"
             "Write the note."}]
    out = (getattr(client.chat(said, think=False), "content", "") or "").strip()
    if is_working(out):
        return {"text": "", "why": "the model did not finish a note"}
    return {"text": without_notes(out), "ids": wanted}
