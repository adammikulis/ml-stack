"""Asking a model a question about a graph.

Handing a model the whole graph does not scale and handing it a pre-chosen slice makes the
choosing the answer. This gives it three things it can do instead — find entries by name, read
what is held on them, trace how two of them connect — and lets it decide which to use. What it
touched comes back with the answer, which is what a caller needs to show its working.

The graph is a mapping with ``nodes`` and ``edges``; nothing here cares what a project calls
its kinds or relations.

    reply = converse("how are Ada and Bea connected?", graph, client)
    reply.content   # what to say
    reply.ids       # what to light up
    reply.steps     # what it did to find out
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SYSTEM = (
    "You are answering a question about a graph. You cannot see it; you read it with the tools "
    "you have been given. Look up the names in the question to get their ids, read what is held "
    "on them, and when the question is about how two things relate, trace the path between "
    "them. When some entries are named as currently highlighted, read them so your answer "
    "knows what the reader is looking at; that is context, not subject matter, and an entry "
    "belongs in your answer only if you have something to say about it.\n\n"
    "Then write the answer. Do not narrate what you looked up — the reader can see that "
    "already. Say what the entries add up to: what they have in common, where they differ, "
    "what connects them, what a reader should do with it. Quote the words that make your point "
    "when the graph holds them. Four to eight sentences of plain prose, no bullet points and no "
    "headings, naming the things you mean rather than their ids.\n\n"
    "Last, call show once with the ids of the entries your answer is about, so the reader can "
    "see them on the graph. Every one you named belongs in it — including any you named from a "
    "quote rather than by reading it. What you opened on the way and did not write about does "
    "not.\n\n"
    "Everything you say comes from what the tools returned. Say plainly when the graph does not "
    "answer the question, and never invent an entry the tools did not show you."
)

# How many tool calls a question may spend. Five was enough to look two names up and read
# them; it is not enough to staff a project, which means looking up each skill, reading the
# people behind them, and then saying which of them the answer is about. Measured against a
# real graph: every staffing question spent all five on searching and never reached `show`.
ROUNDS = 10
# How many times a model may ask for something it has already asked for before the tools are
# taken away. Two is a stumble; three is a loop, and a loop always ends in no answer at all.
REPEATS = 2
# Openings that mean the model is planning rather than answering. gpt-oss puts its analysis
# in a channel of its own, but when a turn spends its whole budget deciding what to do the
# reply that comes back IS the analysis — not empty, so nothing caught it, and the reader
# was shown "We need to answer: ... Search for X again maybe missing." as the answer.
WORKING = ("we need to", "i need to", "need to", "need ", "let me", "let's", "lets ", "from data:",
           "the user asks", "the user wants", "first, i", "i should", "we should", "okay,",
           "ok,", "so the question", "search for", "we must", "i'll produce", "we can use",
           "we need", "look up ", "maybe ")


# A sentence end, including the one with no space after it: the leak arrives welded to the
# answer as "…tool search.Grace Hopper brings…", and splitting on ". " alone keeps them
# as one sentence, so cutting the notes cuts the answer with it.
_ENDS = re.compile(r"(?<=[.!?])(?=\s|[A-Z\u201c\u2018\"'])")


def _sentences(said: str) -> list[str]:
    return [x.strip() for x in _ENDS.split((said or "").replace("\n", " ")) if x.strip()]


def _is_note(one: str) -> bool:
    head = " ".join((one or "").strip().split()).casefold()
    return any(head.startswith(x) for x in WORKING)


def without_notes(said: str) -> str:
    """``said`` with the planning it opens with removed.

    Measured against a real model: a good answer arrives with one sentence of the analysis
    channel stuck to the front — "We need to use tool search.Grace Hopper brings decades
    of healthcare experience…". Throwing the whole reply away over that opening loses a real
    answer and shows the reader "the model did not finish an answer", which is worse than
    the leak. So the leak is cut off and the answer kept.
    """
    parts = _sentences(said)
    while parts and _is_note(parts[0]):
        parts.pop(0)
    return " ".join(parts).strip()


# A tool call written out as prose. On the last turn the tools are taken away so the model
# answers in words — and a model that wanted to call `show` writes the call instead, on the
# end of an otherwise good answer: "…could enhance Dan's swarm projects.
# show({"ids":["person:ada","person:bea"]})". It is not an answer, so it is cut; but it is
# also the model saying exactly what it meant, so the ids are kept.
_SPOKEN = re.compile(r"\s*(?:<[^>]*>)?\s*show\s*\(\s*(\{.*\})\s*\)\s*(?:<[^>]*>)?\s*$",
                     re.DOTALL | re.IGNORECASE)


def spoken_show(said: str) -> tuple[str, list[str]]:
    """``said`` without a trailing written-out ``show`` call, and the ids it named."""
    found = _SPOKEN.search(said or "")
    if not found:
        return (said or "").strip(), []
    try:
        args = json.loads(found.group(1))
    except ValueError:
        return said[:found.start()].strip(), []
    ids = [str(i) for i in (args.get("ids") or ())] if isinstance(args, Mapping) else []
    return said[:found.start()].strip(), ids


def is_working(said: str) -> bool:
    """Whether a reply is notes and nothing else.

    Length is not the test — "Nobody in the graph does both." is a short answer, not a note.
    What decides it is whether anything survives taking the planning off the front.
    """
    return bool((said or "").strip()) and not without_notes(said)


FOUND = 12
JOINED = 12
SAID = 2
SAID_CHARS = 220
LIT = 25

# Every example below is invented, and deliberately shares no name with the community the
# bench asks its questions of: an example that used the bench's own people would be teaching
# the answers rather than the calling convention, and the score would stop meaning anything.
#
# The examples are here because they are what the small models need. Measured over the
# invented community at 32k: gemma-4-E4B scored 17%, and six of its nine failures were the
# same shape — two model calls, a hundred characters of prose, no tool call at all. It was
# not reasoning badly, it was answering from nothing. gpt-oss failed the opposite way, with
# *more* calls on a wrong answer than a right one. A description that says what a tool is
# leaves a small model to infer that it should be called; one that shows a call does not.
TOOLS = [
    {"type": "function", "function": {
        "name": "look_up",
        "description": "Search the graph for entries whose name or attached words match some "
                       "text. Call this first, before answering anything: you have never seen "
                       "this graph, nothing in it is in your memory, and an entry that this "
                       "tool did not return does not exist. Pass every word you want in one "
                       "call — it costs the same as one, and asking one at a time is how a "
                       "question runs out of turns. If the question's own words find "
                       "nothing, search the idea behind them instead of asking again with "
                       "more of them: a cracked kiln is looked for as \"ceramics\", "
                       "\"firing\", \"studio\", not as \"cracked kiln\" again. Example: for "
                       "\"who runs a pottery studio in "
                       "Ambleford?\" call look_up with "
                       "{\"texts\": [\"pottery\", \"studio\", \"Ambleford\"]}, not three "
                       "separate calls.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "one thing to look for, e.g. \"glassblowing\""},
            "texts": {"type": "array", "items": {"type": "string"},
                      "description": "several things to look for, in one call, e.g. "
                                     "[\"pottery\", \"studio\", \"Ambleford\"]"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "look_at",
        "description": "Read what the graph holds on entries you found: their attributes, what "
                       "they are joined to, and a line or two of what was actually said. "
                       "look_up gives you names only — this is where the facts to answer from "
                       "come from, so call it on anything you intend to write about. Example: "
                       "look_at with {\"ids\": [\"person:wren\", \"topic:ceramics\"]}.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids exactly as look_up returned them, e.g. "
                                   "[\"person:wren\", \"org:tinsley\"]"}},
            "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "path_between",
        "description": "Trace how two entries are connected, as the chain of entries between "
                       "them. Use it when the question is about a relationship rather than a "
                       "fact — who could introduce two people, how someone relates to a "
                       "company — and they are not already joined directly. Example: "
                       "path_between with {\"from_id\": \"person:wren\", "
                       "\"to_id\": \"org:tinsley\"}.",
        "parameters": {"type": "object", "properties": {
            "from_id": {"type": "string", "description": "an entry id, e.g. \"person:wren\""},
            "to_id": {"type": "string", "description": "another entry id, e.g. \"person:hollis\""}},
            "required": ["from_id", "to_id"]}}},
    {"type": "function", "function": {
        "name": "show",
        "description": "Say which entries your answer is about, so they light up on the graph. "
                       "Every answer ends with this call. An answer without it lights nothing "
                       "and the reader is left looking at an empty graph, however good the "
                       "words were. Pass everyone and everything you actually wrote about — "
                       "including anyone you named from a quote — and nothing you merely "
                       "opened on the way. A question that asks *who* is answered by people: "
                       "show the people, not the subject they have in common. Example: having "
                       "written \"Wren Halloway fires "
                       "the kiln and Hollis Fen runs the studio\", call show with "
                       "{\"ids\": [\"person:wren\", \"person:hollis\"]}.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids the answer is about, e.g. "
                                   "[\"person:wren\", \"person:hollis\"]"}},
            "required": ["ids"]}}},
]


# The same four tools, said briefly. What a model needs to be told depends entirely on the
# model: the worked examples above took gemma-4-E4B from 17% to 70% recall, and cost
# gpt-oss-120b twenty points over the same questions. A model that already reaches for a
# tool does not need telling to, and being told anyway spends its attention on instructions
# instead of on the question.
#
# So both exist and the caller chooses. `ml-stack-bench --terse` measures which a given
# model wants, because there is no answering that from first principles.
TERSE = [
    {"type": "function", "function": {
        "name": "look_up",
        "description": "Find entries in the graph whose name or attached words match some "
                       "text. Several words in one call cost the same as one.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "what to look for"},
            "texts": {"type": "array", "items": {"type": "string"},
                      "description": "several things to look for, in one call"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "look_at",
        "description": "What the graph holds on some entries: their attributes, what they "
                       "are joined to, and a line or two of what was actually said.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids, as returned by look_up"}},
            "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "path_between",
        "description": "How two entries are connected, as the chain of entries between "
                       "them. For a question about how two things relate.",
        "parameters": {"type": "object", "properties": {
            "from_id": {"type": "string"}, "to_id": {"type": "string"}},
            "required": ["from_id", "to_id"]}}},
    {"type": "function", "function": {
        "name": "show",
        "description": "The entries your answer is about, to light up on the graph. Call it "
                       "once, last, with what you actually wrote about -- not everything "
                       "you opened on the way.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids the answer is about"}},
            "required": ["ids"]}}},
]


# What a *question* looks like when it wants each tool -- for an embedder to match against,
# and never sent to the chat model.
#
# Two consumers want two different texts. A chat model wants to know what a tool does, in
# prose: "search the graph for entries whose name or attached words match some text". An
# embedder wants the opposite. Comparing a user's question to prose *about a capability* is
# comparing unlike things, and embeddinggemma is measurably poor at it -- the same asymmetry
# that made DOCUMENT and QUERY prefixes necessary in `graph.vectors`, where telling it which
# side was which moved a robotics technician from unplaced to second.
#
# So these are questions, not descriptions, because a question against questions is
# like-to-like and that is where the signal is. Putting them in the description instead
# would serve neither: it lengthens the text the chat model reads, which already measured
# too long for a large model, to help something that is not reading it.
# The name for "this question wants no tool at all". Not a tool, so it can never be called;
# it exists so a router has somewhere to put a greeting other than the nearest search.
CHAT = "chat"

TOOL_PROMPTS: dict[str, tuple[str, ...]] = {
    "look_up": (
        "who knows about robotics?",
        "is there anyone here who does marketing?",
        "find me people working on healthcare",
        "which companies are represented?",
        "somebody who can sell things",
        "who fixes machines?",
    ),
    "look_at": (
        "tell me about Iris Bellweather",
        "what is she good at?",
        "what does that company do?",
        "what has this person actually said?",
        "more detail on those two",
    ),
    "path_between": (
        "how are these two connected?",
        "who could introduce me to a lawyer?",
        "what links the foundry and the survey firm?",
        "is there anyone in common between them?",
        "how do I reach that person?",
    ),
    "show": (
        "highlight those on the graph",
        "show me who you mean",
        "light up the people in that answer",
    ),
    # Not a tool: the questions that want *no* graph at all. Without somewhere for these to
    # go, a greeting is matched against four search tools and wins one of them -- "hi"
    # scored 0.900 against "highlight them on the graph", because everything is close to
    # everything in embedding space and the question is only ever "close to what".
    #
    # A greeting is not a failed search, it is a different kind of message, and answering it
    # costs one turn instead of six.
    CHAT: (
        "hi",
        "hello there",
        "thanks, that is helpful",
        "tell me a joke",
        "what can you do?",
        "how does this work?",
        "who are you?",
        "what is the capital of France?",
        "write me a haiku about rain",
        "never mind",
    ),
}


def prompts_for(name: str) -> tuple[str, ...]:
    """Example questions that should route to ``name``, for an embedder. May be empty."""
    return TOOL_PROMPTS.get(name, ())


@dataclass
class Answer:
    """What to say, what to light up, and what was done to find out.

    ``found`` holds what look_up returned, ``read`` what look_at was given, ``path`` what
    path_between traversed. ``ids`` is their union — read first, then path, then found —
    capped at converse's ``limit``.

    ``show`` is different in kind from all three: they record what the tools *touched*,
    which is the working, while ``show`` is what the answer is *about*, because the model
    said so. Reading an entry to describe someone else is not writing about it, and a
    person named from a quote was never read at all — so lighting up ``read`` lights the
    search and hides the answer. Empty when the model never said; the caller decides what
    to fall back to.
    """

    content: str = ""
    ids: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    show: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    @property
    def why(self) -> str:
        return "; ".join(self.steps)


def look_up(graph: Mapping[str, Any], text: str, *, limit: int = FOUND) -> list[dict[str, str]]:
    """Entries whose name, attributes or own words carry that text, best match first.

    Characters only. For a search that also stems and also knows what a word means, pass
    ``finder=`` to converse — see ``ml_stack.graph.search.hybrid``.
    """
    want = " ".join((text or "").split()).casefold()
    if not want:
        return []
    messages = graph.get("messages") or {}
    scored: list[tuple[int, int, Mapping[str, Any]]] = []
    for node in graph.get("nodes") or ():
        label = str(node.get("label") or "").casefold()
        attrs = node.get("attrs") or {}
        if label == want:
            score = 4
        elif want in label:
            score = 3
        elif any(want in str(v).casefold() for v in attrs.values()):
            score = 2
        elif any(want in str((messages.get(mid) or {}).get("text") or "").casefold()
                 for mid in (node.get("messages") or ())[:20]):
            score = 1
        else:
            continue
        scored.append((score, int(node.get("mentions") or 0), node))
    scored.sort(key=lambda row: (-row[0], -row[1], str(row[2].get("label") or "")))
    return [{"id": str(n["id"]), "label": str(n.get("label") or ""), "kind": str(n.get("kind") or "")}
            for _, _, n in scored[:limit]]


def look_at(graph: Mapping[str, Any], ids: Sequence[str]) -> str:
    """What the graph holds on those entries, as text a model can answer from."""
    by_id = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    messages = graph.get("messages") or {}
    lines: list[str] = []
    for node_id in ids:
        node = by_id.get(str(node_id))
        if node is None:
            continue
        attrs = node.get("attrs") or {}
        joined = [f"{e.get('rel', 'joined to')} {by_id[e['target']].get('label')}"
                  for e in (graph.get("edges") or ())
                  if e.get("source") == node_id and e.get("target") in by_id]
        joined += [f"{by_id[e['source']].get('label')} {e.get('rel', 'joined to')} this"
                   for e in (graph.get("edges") or ())
                   if e.get("target") == node_id and e.get("source") in by_id]
        line = f"- {node.get('label')} ({attrs.get('type') or node.get('kind') or 'entry'})"
        for key in ("role", "location"):
            if attrs.get(key):
                line += f", {attrs[key]}"
        if joined:
            line += ": " + "; ".join(joined[:JOINED])
        lines.append(line)
        for mid in (node.get("messages") or ())[:SAID]:
            text = str((messages.get(mid) or {}).get("text") or "")
            if text:
                lines.append(f'    said: "{" ".join(text.split())[:SAID_CHARS]}"')
    return "\n".join(lines)


def path_between(graph: Mapping[str, Any], start: str, goal: str) -> dict[str, Any]:
    """The chain of entries from one to another, and how it reads."""
    from ml_stack.entities.paths import between, shortest_path

    edges = list(graph.get("edges") or ())
    ids = between(edges, start, goal)
    if not ids:
        return {"path": [], "why": "nothing in the graph joins those two"}
    label = {str(n["id"]): str(n.get("label") or "") for n in (graph.get("nodes") or ())}
    steps = shortest_path(edges, start, goal)
    return {"path": ids, "rels": [e.get("rel", "") for e in steps],
            "reads": " → ".join(label.get(i, i) for i in ids)}


def tools_for(graph: Mapping[str, Any], *, finder: Any = None,
              terse: bool = False) -> list[tuple[dict[str, Any], Any]]:
    """The built-in tools over that graph, as ``(schema, callable)`` pairs.

    Each callable takes the parsed arguments mapping. ``finder`` replaces how look_up looks:
    it takes the text and returns ``[{"id", "label", "kind"}, ...]``.
    """
    def find(args: Mapping[str, Any]) -> Any:
        # one word or several: a staffing question needs a lookup per skill, and doing them
        # one round at a time is what spent every turn a question had
        wanted = [str(x) for x in (args.get("texts") or ()) if str(x).strip()]
        if not wanted and str(args.get("text") or "").strip():
            wanted = [str(args["text"])]
        rows, seen = [], set()
        for text in wanted:
            for r in (finder(text) if finder is not None else look_up(graph, text)):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
        # an empty list reads as "try again"; saying nothing matched reads as "move on"
        return rows or {"none": f"Nothing in the graph matches {', '.join(map(repr, wanted))}. "
                                "Try different words, or answer with what you already have."}

    def read(args: Mapping[str, Any]) -> str:
        return look_at(graph, [str(i) for i in (args.get("ids") or ())])

    def trace(args: Mapping[str, Any]) -> dict[str, Any]:
        return path_between(graph, str(args.get("from_id") or ""), str(args.get("to_id") or ""))

    def light(args: Mapping[str, Any]) -> str:
        # the ids are the whole result; the model is told they arrived so it stops calling it
        return f"lit {len(list(args.get('ids') or ()))} on the graph"

    said = TERSE if terse else TOOLS
    return [(said[0], find), (said[1], read), (said[2], trace), (said[3], light)]


def converse(question: str, graph: Mapping[str, Any], client: Any, *,
             turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
             rounds: int = ROUNDS, limit: int = LIT,
             tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
             finder: Any = None, held: Sequence[str] = (),
             opening: Sequence[str] = ()) -> Answer:
    """One question, answered with the graph in hand.

    ``client`` is anything with ``chat(messages, tools=...)`` returning a reply carrying
    ``content`` and ``tool_calls`` — ``ml_stack.client.Client`` does. ``tools`` is
    ``[(schema, callable), ...]``, each callable taking the parsed arguments mapping;
    ``tools_for(graph)`` by default. ``finder`` replaces just look_up's callable.
    ``held`` names entries already highlighted for the reader: they are told to the model
    by label and id, and enter ``ids`` only if a tool call touches them.
    """
    return _converse(question, graph, client, turns=turns, system=system, rounds=rounds,
                     limit=limit, tools=tools, finder=finder, held=held, emit=None,
                     opening=opening)


def converse_stream(question: str, graph: Mapping[str, Any], client: Any, *,
                    on_event: Any,
                    turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
                    rounds: int = ROUNDS, limit: int = LIT,
                    tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
                    finder: Any = None, held: Sequence[str] = (),
                    opening: Sequence[str] = ()) -> Answer:
    """converse, reporting what is happening to ``on_event`` as it happens.

    ``on_event`` gets one mapping per event: ``{"event": "thinking", "text"}`` as the
    model thinks, ``{"event": "tool", "name", "detail"}`` when it calls a tool,
    ``{"event": "tool_result", "name", "count"}`` with how much came back,
    ``{"event": "answer", "text"}`` as the answer arrives, then ``{"event": "done"}``.
    A client whose ``chat`` takes ``on_delta`` streams the text a piece at a time;
    any other client's text arrives whole.
    """
    return _converse(question, graph, client, turns=turns, system=system, rounds=rounds,
                     limit=limit, tools=tools, finder=finder, held=held, emit=on_event,
                     opening=opening)


def _call_detail(name: str, args: Mapping[str, Any]) -> str:
    if name == "look_up":
        return repr(str(args.get("text") or ""))
    if name in ("look_at", "show"):
        ids = list(args.get("ids") or ())
        return f"{len(ids)} id" + ("" if len(ids) == 1 else "s")
    if name == "path_between":
        return f"{args.get('from_id')} → {args.get('to_id')}"
    return json.dumps(args, ensure_ascii=False)[:80]


def _result_count(result: Any) -> int:
    if isinstance(result, (list, tuple)):
        return len(result)
    if isinstance(result, Mapping) and "path" in result:
        return len(result.get("path") or ())
    if isinstance(result, str):
        return len(result.splitlines()) if result.strip() else 0
    return 1 if result else 0


def _converse(question: str, graph: Mapping[str, Any], client: Any, *,
              turns: Sequence[Mapping[str, str]], system: str, rounds: int, limit: int,
              tools: Sequence[tuple[Mapping[str, Any], Any]] | None,
              finder: Any, held: Sequence[str], emit: Any, opening: Sequence[str] = ()) -> Answer:
    if tools is None:
        tools = tools_for(graph, finder=finder)
    elif finder is not None:
        def _found(args: Mapping[str, Any]) -> Any:
            wanted = [str(x) for x in (args.get("texts") or ()) if str(x).strip()]
            if not wanted and str(args.get("text") or "").strip():
                wanted = [str(args["text"])]
            rows, seen = [], set()
            for text in wanted:
                for r in finder(text):
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        rows.append(r)
            return rows

        tools = [(schema, fn) if (schema.get("function") or {}).get("name") != "look_up"
                 else (schema, _found)
                 for schema, fn in tools]
    schemas = [schema for schema, _ in tools]
    run = {str((schema.get("function") or {}).get("name") or ""): fn for schema, fn in tools}

    known = {str(n["id"]) for n in (graph.get("nodes") or ())}
    lit = [str(h) for h in held if str(h) in known]
    if lit:
        label = {str(n["id"]): str(n.get("label") or "") for n in (graph.get("nodes") or ())}
        system = (system + "\n\nCurrently highlighted: "
                  + ", ".join(f"{label[h]} ({h})" for h in lit))
    out = Answer()

    def note(into: list[str], ids: Sequence[str]) -> None:
        for one in ids:
            if str(one) in known and str(one) not in into:
                into.append(str(one))

    said = [{"role": ("assistant" if t.get("role") == "assistant" else "user"),
             "content": str(t.get("content") or "")[:4000]}
            for t in turns if str(t.get("content") or "").strip()]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *said,
                                      {"role": "user", "content": question}]

    # A search has already been run — cheaply, by a small model that only measures meaning —
    # and what it found is read out before the first turn. Most questions are then answered
    # without calling anything, which is the point: every look_up is a whole round trip
    # through a large model, and the small one costs milliseconds. It is a suggestion, not
    # an answer: the tools are all still there, and the prompt says to go looking when this
    # is not what the question was about.
    start = [str(i) for i in opening if str(i) in known][:FOUND]
    if start:
        material = (run.get("look_at") or (lambda a: look_at(graph, a["ids"])))({"ids": start})
        note(out.found, start)
        out.steps.append(f"was handed {len(start)} to start from")
        messages.append({"role": "user", "content":
                         "A search for this question already found these, most likely first. "
                         "Use them if they answer it, look for something else if they do not, "
                         "and say so plainly if nothing here does:\n" + str(material)})
        if emit is not None:
            emit({"event": "tool", "name": "shortlist", "detail": f"{len(start)} to start from"})

    # The tools that go looking. Taking these away is how a turn is made to stop searching
    # and answer; taking away *every* tool is how it was also made unable to act. A message
    # asking for an entry to be changed came in, the search went round in circles, the loop
    # stopped — and the one call left had no `request_change` to reach for, so a member's
    # request became "the model did not finish an answer". Stop the searching, not the rest.
    searching = {"look_up", "look_at", "path_between"}
    quiet = [x for x in schemas if str((x.get("function") or {}).get("name")) not in searching]

    def step(with_tools: bool) -> Any:
        offer = schemas if with_tools else quiet
        kw: dict[str, Any] = {"tools": offer} if offer else {}
        if emit is None:
            return client.chat(messages, think=False, **kw)
        streamed = {"thinking": False, "answer": False}

        def on_delta(kind: str, text: str) -> None:
            if not text:
                return
            name = "thinking" if kind == "thinking" else "answer"
            streamed[name] = True
            emit({"event": name, "text": text})

        reply = client.chat(messages, think=False, on_delta=on_delta, **kw)
        if not streamed["thinking"]:
            trace = (getattr(reply, "thinking", "") or "").strip()
            if trace:
                emit({"event": "thinking", "text": trace})
        if not streamed["answer"] and not (getattr(reply, "tool_calls", None) or []):
            whole = (getattr(reply, "content", "") or "")
            if whole.strip():
                emit({"event": "answer", "text": whole})
        return reply

    def _searched(reply: Any) -> bool:
        """Whether that reply asked for anything other than showing."""
        return any((call.get("function") or {}).get("name") in searching
                   for call in (getattr(reply, "tool_calls", None) or []))

    spent = False
    spent_on: set[tuple[str, str]] = set()
    repeats = 0
    reply = None
    def dispatch(reply: Any) -> bool:
        """Run whatever the reply asked for. False when it asked for nothing."""
        nonlocal spent, repeats
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            return False
        spent = True
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            if emit is not None:
                emit({"event": "tool", "name": name, "detail": _call_detail(name, args)})
            do = run.get(name)
            again = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if do is None:
                result: Any = {"error": f"no such tool: {name}"}
            elif again in spent_on:
                # Asking the same thing twice is how a budget disappears: measured against a
                # real graph, one question spent six of its ten rounds looking up one word,
                # over and over, and never answered. Being told does not stop it — the tools
                # are taken away below, which does.
                repeats += 1
                result = {"already": "You have asked this exactly before and got the same "
                                     "answer. Use what it gave you, look for something else, "
                                     "or answer the question."}
                out.steps.append(f"asked {name} the same thing again")
            elif name == "look_up":
                result = do(args)
                note(out.found, [r["id"] for r in result] if isinstance(result, list) else [])
                asked = [str(x) for x in (args.get("texts") or ())] or [str(args.get("text") or "")]
                out.steps.append("looked up " + ", ".join(repr(x) for x in asked))
            elif name == "look_at":
                # one guard, in note: an id the model made up is neither read nor lit up
                ids = [str(i) for i in (args.get("ids") or ())]
                note(out.read, ids)
                result = do(args) or "nothing on those"
                real = sum(1 for i in ids if i in known)
                out.steps.append(f"read {real} entr" + ("y" if real == 1 else "ies"))
            elif name == "path_between":
                result = do(args)
                note(out.path, result.get("path") or [])
                out.steps.append("traced a path" if result.get("path") else "found no path")
            elif name == "show":
                # the same guard as look_at: an id the model made up is not lit up
                ids = [str(i) for i in (args.get("ids") or ())]
                note(out.show, ids)
                result = do(args)
                real = sum(1 for i in ids if i in known)
                out.steps.append(f"lit {real} entr" + ("y" if real == 1 else "ies"))
            else:
                result = do(args)
                out.steps.append(f"used {name}")
            spent_on.add(again)
            if emit is not None:
                emit({"event": "tool_result", "name": name, "count": _result_count(result)})
            messages.append({"role": "tool", "tool_call_id": call.get("id") or name,
                             "name": name, "content": json.dumps(result, ensure_ascii=False)[:6000]})
        return True

    answered = False
    for _ in range(rounds):
        if repeats >= REPEATS:
            # going in circles: the loop ends here and the answer is asked for below
            out.steps.append("stopped searching in circles")
            break
        reply = step(True)
        if not dispatch(reply):
            answered = True          # it stopped calling tools, so this reply is the answer
            break
        # Once it has said what its answer is about, more searching cannot improve that --
        # `show` is the last thing a turn does, and a round after it is a round trip spent
        # to be told the same. Only when the round did nothing else: a turn that showed and
        # kept looking in the same breath has not finished looking.
        if out.show and not _searched(reply):
            out.steps.append("said what to light, so the searching stopped")
            break
    if spent and not answered:
        # The searching is over, one way or another. What is left on the table are the tools
        # that act — saying what to light, asking for the graph to be changed — and they are
        # run, not ignored: a member's request arrived on exactly this turn and was dropped
        # because nothing here executed it.
        reply = step(False)
        if dispatch(reply):
            reply = step(False)
    # a thinking model can stop calling tools and still say nothing, and it can run out of
    # rounds the same way. Either silence gets one plain instruction to answer; one that
    # only ever searched also gets the top finds read to it first
    if spent and not (getattr(reply, "content", "") or "").strip():
        nudge = "Answer the question now, in plain words."
        if not out.read and out.found:
            top = out.found[:FOUND]
            do = run.get("look_at")
            material = (do({"ids": top}) if do is not None else look_at(graph, top)) or ""
            note(out.read, top)
            out.steps.append(f"read the top {len(top)} find" + ("" if len(top) == 1 else "s"))
            if emit is not None:
                emit({"event": "tool", "name": "look_at", "detail": f"{len(top)} ids"})
                emit({"event": "tool_result", "name": "look_at",
                      "count": _result_count(material)})
            nudge = ("What the graph holds on what you found:\n" + str(material)
                     + "\n\n" + nudge)
        messages.append({"role": "user", "content": nudge})
        reply = step(False)
    out.content = (getattr(reply, "content", "") or "").strip()
    if out.content:
        out.content, meant = spoken_show(out.content)
        if meant:
            # what it wrote down is what it meant, and it is a tighter set than asking again
            note(out.show, meant)
            out.steps.append("said what to light in words, so it was not asked again")
    if out.content and not is_working(out.content):
        trimmed = without_notes(out.content)
        if trimmed != out.content:
            out.steps.append("trimmed the notes off the front of the answer")
            out.content = trimmed
    if out.content and is_working(out.content):
        # It answered with its notes. Same remedy as saying nothing at all — the notes are
        # the material, and what is wanted is the answer they were working towards.
        out.steps.append("answered with its notes, so was asked again")
        messages.append({"role": "assistant", "content": out.content})
        out.content = ""
    if not out.content:
        # Thinking is a scratchpad — "Actually the look_at shows… need to check X" — and
        # showing it as the answer reads as a broken machine. It is offered back as
        # material and asked for the answer it was working towards; only prose that reads
        # like an answer is used, never the working itself.
        trace = (getattr(reply, "thinking", "") or "").strip()
        if trace or messages[-1].get("role") == "assistant":
            messages.append({"role": "user", "content":
                             "Write the answer your notes were working towards, in plain "
                             "prose for someone who cannot see them. Do not narrate what you "
                             "looked up."})
            said = (getattr(client.chat(messages, think=False), "content", "") or "").strip()
            out.content = "" if is_working(said) else without_notes(said)
            if out.content and emit is not None:
                emit({"event": "answer", "text": out.content})
    if not out.content:
        out.content = ("The model did not finish an answer. What it opened is lit up on the "
                       "graph; asking again, or more narrowly, usually gets one.")
        if emit is not None:
            emit({"event": "answer", "text": out.content})
    # A model that answered without touching the graph answered from nothing, and nothing is
    # what it knows: this graph was not in its training data. Measured over the invented
    # community, six of gemma-4-E4B's nine failures were exactly this — two model calls, a
    # hundred characters of prose, no search. The larger models never do it, which is why it
    # went unseen until a small one was measured. It gets one chance to go and look, with the
    # searching tools back on the table; if it declines again, its answer stands as it is.
    if out.content and not (out.read or out.found or out.path) and "look_up" in run:
        messages.append({"role": "assistant", "content": out.content})
        messages.append({"role": "user", "content":
                         "You answered without searching the graph. You have not seen this "
                         "graph before and cannot answer it from memory. Call look_up now "
                         "with the words from the question, then answer from what it returns."})
        if emit is not None:
            # a reader watching this stream has the first answer on screen already, and the
            # page appends what arrives; without being told to start over it would show the
            # two answers run together
            emit({"event": "restart", "why": "answered without searching"})
        reply = step(True)
        if dispatch(reply):
            reply = step(False)
        said = (getattr(reply, "content", "") or "").strip()
        if said:
            # whatever it said last is what the reader is looking at, so it is the answer
            out.content, meant = spoken_show(said)
            note(out.show, meant)
        if out.read or out.found or out.path:
            out.steps.append("answered without looking, so it was sent to look")

    # What the answer is about is the one thing only the model knows, and a turn that spends
    # its budget searching never reaches `show` on its own — measured against a real graph,
    # not one staffing question in six did. So it is asked outright, once, with the answer it
    # just wrote in front of it and nothing else to call.
    if out.content and not out.show and (out.read or out.found or out.path):
        messages.append({"role": "assistant", "content": out.content})
        messages.append({"role": "user", "content":
                         "Now call show once with the ids of the entries your answer is "
                         "about — everyone and everything you named in it, including any you "
                         "named from a quote. Nothing you opened and did not write about."})
        last = client.chat(messages, think=False, tools=[TOOLS[3]])
        for call in (getattr(last, "tool_calls", None) or []):
            fn = call.get("function") or {}
            if (fn.get("name") or "") != "show":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                continue
            ids = [str(i) for i in (args.get("ids") or ())]
            note(out.show, ids)
        if out.show:
            out.steps.append(f"lit {len(out.show)} entr" + ("y" if len(out.show) == 1 else "ies"))
    for one in (*out.read, *out.path, *out.found):
        if one not in out.ids:
            out.ids.append(one)
    out.ids = out.ids[:limit]
    out.show = out.show[:limit]
    if emit is not None:
        emit({"event": "done"})
    return out
