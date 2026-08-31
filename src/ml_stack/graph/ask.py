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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SYSTEM = (
    "You are answering a question about a graph. You cannot see it; you read it with the tools "
    "you have been given. Look up the names in the question to get their ids, read what is held "
    "on them, and when the question is about how two things relate, trace the path between "
    "them. When some entries are named as currently highlighted and the question builds on what "
    "is already shown, re-read them together with one look_at of their ids so they stay in "
    "the answer; when it moves to "
    "something else, leave them.\n\n"
    "Then write the answer. Do not narrate what you looked up — the reader can see that "
    "already. Say what the entries add up to: what they have in common, where they differ, "
    "what connects them, what a reader should do with it. Quote the words that make your point "
    "when the graph holds them. Four to eight sentences of plain prose, no bullet points and no "
    "headings, naming the things you mean rather than their ids.\n\n"
    "Everything you say comes from what the tools returned. Say plainly when the graph does not "
    "answer the question, and never invent an entry the tools did not show you."
)

ROUNDS = 5
FOUND = 12
JOINED = 12
SAID = 2
SAID_CHARS = 220
LIT = 25

TOOLS = [
    {"type": "function", "function": {
        "name": "look_up",
        "description": "Find entries in the graph whose name or attached words match some text. "
                       "Use it to turn a name in the question into ids you can work with.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "what to look for"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "look_at",
        "description": "What the graph holds on some entries: their attributes, what they are "
                       "joined to, and a line or two of what was actually said.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "array", "items": {"type": "string"},
                    "description": "entry ids, as returned by look_up"}},
            "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "path_between",
        "description": "How two entries are connected, as the chain of entries between them. "
                       "Use it when a question is about how two things relate and they are not "
                       "joined directly.",
        "parameters": {"type": "object", "properties": {
            "from_id": {"type": "string"}, "to_id": {"type": "string"}},
            "required": ["from_id", "to_id"]}}},
]


@dataclass
class Answer:
    """What to say, what to light up, and what was done to find out.

    ``found`` holds what look_up returned, ``read`` what look_at was given, ``path`` what
    path_between traversed. ``ids`` is their union — read first, then path, then found —
    capped at converse's ``limit``.
    """

    content: str = ""
    ids: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
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


def tools_for(graph: Mapping[str, Any], *, finder: Any = None) -> list[tuple[dict[str, Any], Any]]:
    """The built-in tools over that graph, as ``(schema, callable)`` pairs.

    Each callable takes the parsed arguments mapping. ``finder`` replaces how look_up looks:
    it takes the text and returns ``[{"id", "label", "kind"}, ...]``.
    """
    def find(args: Mapping[str, Any]) -> list[dict[str, str]]:
        text = str(args.get("text") or "")
        return finder(text) if finder is not None else look_up(graph, text)

    def read(args: Mapping[str, Any]) -> str:
        return look_at(graph, [str(i) for i in (args.get("ids") or ())])

    def trace(args: Mapping[str, Any]) -> dict[str, Any]:
        return path_between(graph, str(args.get("from_id") or ""), str(args.get("to_id") or ""))

    return [(TOOLS[0], find), (TOOLS[1], read), (TOOLS[2], trace)]


def converse(question: str, graph: Mapping[str, Any], client: Any, *,
             turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
             rounds: int = ROUNDS, limit: int = LIT,
             tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
             finder: Any = None, held: Sequence[str] = ()) -> Answer:
    """One question, answered with the graph in hand.

    ``client`` is anything with ``chat(messages, tools=...)`` returning a reply carrying
    ``content`` and ``tool_calls`` — ``ml_stack.client.Client`` does. ``tools`` is
    ``[(schema, callable), ...]``, each callable taking the parsed arguments mapping;
    ``tools_for(graph)`` by default. ``finder`` replaces just look_up's callable.
    ``held`` names entries already highlighted for the reader: they are told to the model
    by label and id, and enter ``ids`` only if a tool call touches them.
    """
    return _converse(question, graph, client, turns=turns, system=system, rounds=rounds,
                     limit=limit, tools=tools, finder=finder, held=held, emit=None)


def converse_stream(question: str, graph: Mapping[str, Any], client: Any, *,
                    on_event: Any,
                    turns: Sequence[Mapping[str, str]] = (), system: str = SYSTEM,
                    rounds: int = ROUNDS, limit: int = LIT,
                    tools: Sequence[tuple[Mapping[str, Any], Any]] | None = None,
                    finder: Any = None, held: Sequence[str] = ()) -> Answer:
    """converse, reporting what is happening to ``on_event`` as it happens.

    ``on_event`` gets one mapping per event: ``{"event": "thinking", "text"}`` as the
    model thinks, ``{"event": "tool", "name", "detail"}`` when it calls a tool,
    ``{"event": "tool_result", "name", "count"}`` with how much came back,
    ``{"event": "answer", "text"}`` as the answer arrives, then ``{"event": "done"}``.
    A client whose ``chat`` takes ``on_delta`` streams the text a piece at a time;
    any other client's text arrives whole.
    """
    return _converse(question, graph, client, turns=turns, system=system, rounds=rounds,
                     limit=limit, tools=tools, finder=finder, held=held, emit=on_event)


def _call_detail(name: str, args: Mapping[str, Any]) -> str:
    if name == "look_up":
        return repr(str(args.get("text") or ""))
    if name == "look_at":
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
              finder: Any, held: Sequence[str], emit: Any) -> Answer:
    if tools is None:
        tools = tools_for(graph, finder=finder)
    elif finder is not None:
        tools = [(schema, fn) if (schema.get("function") or {}).get("name") != "look_up"
                 else (schema, lambda args: finder(str(args.get("text") or "")))
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

    def step(with_tools: bool) -> Any:
        kw: dict[str, Any] = {"tools": schemas} if with_tools else {}
        if emit is None:
            return client.chat(messages, think=False, **kw)
        pending: list[str] = []
        streamed = {"any": False}

        def on_delta(kind: str, text: str) -> None:
            if not text:
                return
            streamed["any"] = True
            if kind == "thinking":
                emit({"event": "thinking", "text": text})
            elif with_tools:
                pending.append(text)
            else:
                emit({"event": "answer", "text": text})

        reply = client.chat(messages, think=False, on_delta=on_delta, **kw)
        if not streamed["any"]:
            trace = (getattr(reply, "thinking", "") or "").strip()
            if trace:
                emit({"event": "thinking", "text": trace})
        if not (getattr(reply, "tool_calls", None) or []):
            whole = "".join(pending) if streamed["any"] else (getattr(reply, "content", "") or "")
            if whole.strip():
                emit({"event": "answer", "text": whole})
        return reply

    spent = False
    reply = None
    for _ in range(rounds):
        reply = step(True)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            break
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
            if do is None:
                result: Any = {"error": f"no such tool: {name}"}
            elif name == "look_up":
                result = do(args)
                note(out.found, [r["id"] for r in result])
                out.steps.append(f"looked up {str(args.get('text'))!r}")
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
            else:
                result = do(args)
                out.steps.append(f"used {name}")
            if emit is not None:
                emit({"event": "tool_result", "name": name, "count": _result_count(result)})
            messages.append({"role": "tool", "tool_call_id": call.get("id") or name,
                             "name": name, "content": json.dumps(result, ensure_ascii=False)[:6000]})
    else:
        # the rounds ran out mid-loop: the last reply is a tool call, not an answer. Ask once
        # more with the tools taken away, so the question is answered rather than dropped.
        if spent:
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
    if not out.content:
        # a model can put every word of its answer in the thinking channel
        out.content = (getattr(reply, "thinking", "") or "").strip()
        if out.content and emit is not None:
            emit({"event": "answer", "text": out.content})
    for one in (*out.read, *out.path, *out.found):
        if one not in out.ids:
            out.ids.append(one)
    out.ids = out.ids[:limit]
    if emit is not None:
        emit({"event": "done"})
    return out
