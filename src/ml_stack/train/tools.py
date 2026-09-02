"""ml-stack-train-tools: a project's tool schemas, turned into a model that calls them.

One command does three things, each skipped when its output is already in ``--out``:

1. **synth** — read the worked examples out of the tool descriptions (``for "Q" call T with
   {...}`` and ``"Q" → T(k="v")``), add the example questions a project keeps for its
   router, and template them into chat conversations: system, user, and an assistant turn
   that calls the tool — plus a share of turns that call nothing, so the model learns when
   *not* to. A served model (``--ask``) writes more, with the examples as few-shots.
2. **train** — the ``tool-calls`` recipe: the base model's own chat template renders each
   conversation, and the loss is on the assistant tokens only.
3. **export** — the checkpoint back into Hugging Face layout, then ``ml_stack.gguf.export``.

The examples are the seed because they are what was measured to matter: a worked call in a
description took gemma-4-E4B from 17% to 70% recall on the same weights. A schema that
already teaches a model at inference time is the right thing to teach a smaller one from.

And one subcommand, ``from-bench``, whose seed is not the descriptions but what a model
actually did: every traced question a benchmark kept, above a score, turned into one
training example per model turn — the conversation up to that turn as the input, the call
the model made as the target (`from_bench`, `examples_from`). Synthetic data teaches the
shape of a call; a bench trace teaches the calls that scored, on this graph, with these ids.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CHAT", "Example", "SYSTEM", "examples_from", "examples_in", "from_bench",
           "load_tools", "main", "schemas_of", "split", "synthesise", "traced_rows",
           "would_yield", "write_dataset"]

CHAT = "chat"
"""The prompts key for messages that want no tool — the same key ``graph.ask.TOOL_PROMPTS``
uses, so a project's router examples can be handed over as they are."""

SYSTEM = ("You answer questions with the tools you have been given. When a question needs "
          "one, call it with the arguments the question asks for; when it needs none, reply "
          "in a sentence.")

HOLDOUT_EVERY = 10
"""One seed question in ten is held out, by hash, with every paraphrase of it."""

CHAT_SHARE = 0.15
"""The share of rows that call nothing. Measured need, not taste: a model given tools and
never shown a turn without one calls a tool on "hi"."""


@dataclass(frozen=True)
class Example:
    """One worked example: a question and the call that answers it.

    ``question`` is empty for a bare call written without one (``look_at with {...}``),
    which still seeds arguments. ``arguments`` is ``None`` for a question that only a
    router example supplied, which the synthesiser fills.
    """

    question: str
    tool: str
    arguments: dict[str, Any] | None


# -- reading the schemas -----------------------------------------------------------------

def schemas_of(tools: Any) -> list[dict[str, Any]]:
    """Plain ``{"type": "function", "function": {...}}`` schemas, whatever shape came in.

    Accepts the list ``tools_for`` emits (``(schema, callable)`` pairs), a list of schemas,
    or a list of bare ``{"name", "description", "parameters"}`` mappings.
    """
    out = []
    for item in tools:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], Mapping):
            item = item[0]
        if not isinstance(item, Mapping):
            raise ValueError(f"not a tool schema: {item!r}")
        if "function" in item:
            out.append({"type": "function", "function": dict(item["function"])})
        elif "name" in item:
            out.append({"type": "function", "function": dict(item)})
        else:
            raise ValueError(f"a tool schema needs a name: {item!r}")
    return out


def _fn(schema: Mapping[str, Any]) -> dict[str, Any]:
    return schema["function"]


def _properties(schema: Mapping[str, Any]) -> dict[str, Any]:
    return dict((_fn(schema).get("parameters") or {}).get("properties") or {})


def _required(schema: Mapping[str, Any]) -> list[str]:
    return list((_fn(schema).get("parameters") or {}).get("required") or [])


_QUOTED = r'"((?:[^"\\]|\\.)+?)"'
_FOR_CALL = re.compile(rf'for\s+{_QUOTED}\s+call\s+([A-Za-z_]\w*)\s+with\s+(?=\{{)')
_BARE_CALL = re.compile(r'\b([A-Za-z_]\w*)\s+with\s+(?=\{)')
_ARROW = re.compile(rf'{_QUOTED}\s*(?:→|->)\s*([A-Za-z_]\w*)\(([^()]*)\)')
_KWARG = re.compile(r'([A-Za-z_]\w*)\s*=\s*("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)')
_DECODER = json.JSONDecoder()


def _calls_in(text: str, names: set[str]) -> list[Example]:
    """Every worked example in one description, in both shapes."""
    found: list[Example] = []
    taken: list[tuple[int, int]] = []

    for m in _FOR_CALL.finditer(text):
        tool = m.group(2)
        if tool not in names:
            continue
        try:
            args, end = _DECODER.raw_decode(text, m.end())
        except ValueError:
            continue
        if isinstance(args, dict):
            found.append(Example(m.group(1).replace('\\"', '"'), tool, args))
            taken.append((m.start(), end))

    for m in _ARROW.finditer(text):
        tool = m.group(2)
        if tool not in names:
            continue
        args: dict[str, Any] = {}
        for k, raw in _KWARG.findall(m.group(3)):
            try:
                args[k] = json.loads(raw)
            except ValueError:
                args[k] = raw
        found.append(Example(m.group(1).replace('\\"', '"'), tool, args))
        taken.append((m.start(), m.end()))

    for m in _BARE_CALL.finditer(text):
        tool = m.group(1)
        if tool not in names or any(a <= m.start() < b for a, b in taken):
            continue
        try:
            args, end = _DECODER.raw_decode(text, m.end())
        except ValueError:
            continue
        if isinstance(args, dict):
            found.append(Example("", tool, args))
    return found


def examples_in(tools: Any, prompts: Mapping[str, Sequence[str]] | None = None) -> list[Example]:
    """The worked examples a set of tool descriptions carries, plus a router's questions.

    ``prompts`` is ``{tool: [question, ...]}``; its questions become examples whose
    arguments are left for the synthesiser, and its ``CHAT`` key becomes questions that
    want no tool. A prompts key that names no tool is an error, because a typo there would
    otherwise train nothing and say so to nobody.
    """
    schemas = schemas_of(tools)
    names = {_fn(s)["name"] for s in schemas}
    out: list[Example] = []
    for schema in schemas:
        out.extend(_calls_in(str(_fn(schema).get("description") or ""), names))
    for tool, questions in (prompts or {}).items():
        if tool != CHAT and tool not in names:
            raise ValueError(f"prompts name a tool {tool!r} the schemas do not have; "
                             f"they have {sorted(names)}")
        out.extend(Example(str(q), tool, None) for q in questions if str(q).strip())
    return out


# -- filling in arguments ------------------------------------------------------------------

_EG = re.compile(r'e\.g\.\s*(\[[^\]]*\]|"(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*)')
_STOP = frozenset("""
a about actually all am an and any anyone anybody anything are as at be by can could did
do does doing everything find for from get give has have he her here him his how i if in
is it its just know knows like make me more my need nothing of on one or our out people
person please she should so some somebody someone something tell than thanks that the their
them then there these they this those to up us want was we what when where which who whom
whose why will with work works would you your
""".split())
_URL = re.compile(r"https?://\S+")
_ID = re.compile(r"^[a-z]+:[\w-]+$")


def _values_of(prop: Mapping[str, Any]) -> list[Any]:
    """What a parameter's schema says it looks like: its enum, or the ``e.g.`` values."""
    if prop.get("enum"):
        return list(prop["enum"])
    items = prop.get("items") or {}
    if items.get("enum"):
        return [list(items["enum"])]
    out: list[Any] = []
    for m in _EG.finditer(str(prop.get("description") or "")):
        raw = m.group(1)
        try:
            value = json.loads(raw if raw.startswith("[") else f"[{raw}]")
        except ValueError:
            continue
        out.extend([value] if raw.startswith("[") else value)
    return out


def _content_words(question: str) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z][\w'-]*", question)
             if w.lower() not in _STOP and len(w) > 2]
    return words[:3]


def _looks_like_free_text(values: Sequence[Any]) -> bool:
    flat = [v for value in values for v in (value if isinstance(value, list) else [value])]
    return all(isinstance(v, str) and not _ID.match(v) and not _URL.match(v) for v in flat)


def _fill(schema: Mapping[str, Any], question: str, seeds: Sequence[dict[str, Any]],
          rng: random.Random) -> dict[str, Any]:
    """Arguments for a question that came without any.

    The question supplies what it can — words for a free-text parameter, a URL for a URL
    parameter, an enum value it mentions — and a worked example supplies the rest, so an
    id-shaped argument is always one a description already showed. That teaches the shape
    of a call, not the id; the ids a real turn uses come from an earlier tool result.
    """
    props = _properties(schema)
    required = _required(schema)
    seed = dict(rng.choice(list(seeds))) if seeds else {}
    wanted = list(required) or ([rng.choice(sorted(props))] if props else [])
    for name in props:
        if name not in wanted and name in seed and rng.random() < 0.3:
            wanted.append(name)

    out: dict[str, Any] = {}
    for name in wanted:
        prop = props.get(name) or {}
        kind = str(prop.get("type") or "string").lower()
        values = _values_of(prop)
        # an enum, or a description that lists several alternatives, is a vocabulary; one
        # ``e.g.`` value ("glassblowing") is free text the question supplies itself
        vocab = [v for v in values if isinstance(v, str)] \
            if (prop.get("enum") or len(values) >= 2) else []
        mentioned = [v for v in vocab if v.lower() in question.lower()]
        url = _URL.search(question)
        free = _looks_like_free_text(values) if values else \
            _looks_like_free_text([seed[name]]) if name in seed else True
        if kind == "boolean":
            out[name] = bool(seed.get(name, False))
        elif kind == "string" and url and ("url" in name.lower() or "link" in name.lower()):
            out[name] = url.group(0).rstrip(".,)")
        elif mentioned:
            out[name] = mentioned[0]
        elif vocab:
            out[name] = vocab[0]
        elif not free:
            out[name] = seed[name] if name in seed else \
                (values[0] if not isinstance(values[0], list) or kind == "array" else values[0][0])
        elif kind == "array":
            out[name] = _content_words(question) or list(seed.get(name) or values[:1] or ["?"])[:3]
        elif kind == "string":
            out[name] = " ".join(_content_words(question)) or str(seed.get(name) or (values or ["?"])[0])
        elif name in seed:
            out[name] = seed[name]
        elif kind in ("integer", "number"):
            out[name] = 1
        else:
            out[name] = values[0] if values else ""
    return out


# -- paraphrasing ----------------------------------------------------------------------------

_PREFIXES = ("", "Quick one: ", "Hey, ", "Question: ", "One more: ", "OK — ", "Sorry, ",
             "Next: ")
_SUFFIXES = ("", " Thanks.", " Please.")


def _variants(question: str) -> list[str]:
    """Every wording of a question the templates produce, the original first."""
    base = question.strip()
    bare = base.rstrip("?.! ")
    seen: list[str] = []
    for prefix in _PREFIXES:
        for suffix in _SUFFIXES:
            body = base if not prefix else base[:1].lower() + base[1:]
            for text in (f"{prefix}{body}{suffix}", f"{prefix}{bare}{suffix}"):
                text = text.strip()
                if text and text not in seen:
                    seen.append(text)
    return seen


_GREETING = re.compile(r"\b(hi|hello|hey|good (morning|afternoon|evening))\b", re.I)
_THANKS = re.compile(r"\b(thanks|thank you|cheers|bye|goodbye|never mind|that's all)\b", re.I)
_ASIDES = (
    "hi", "hello there", "hey", "thanks, that is helpful", "good morning", "tell me a joke",
    "what can you do?", "how does this work?", "who are you?", "never mind",
    "what is the capital of Peru?", "write me a haiku about rain", "how are you today?",
    "what year is it?", "that's all for now, bye", "can you count to ten?",
    "what does the word serendipity mean?", "ok", "sounds good", "I am just testing you",
)
_REPLIES = {
    "greeting": ("Hello! Ask me about what the tools here can look up and I will find it.",
                 "Hi — happy to help. What would you like to know?"),
    "thanks": ("You're welcome. Ask whenever you need something looked up.",
               "Glad it helped. I am here when you need something found."),
    "other": ("That is outside what I can look up here; ask me about what the tools cover "
              "and I will find it.",
              "I can only answer from what my tools reach, and that is not in them.",
              "Nothing here answers that. Ask me something the tools can look up instead."),
}


# -- the synthesiser -------------------------------------------------------------------------

def _hash(text: str) -> int:
    return int(hashlib.sha256(text.strip().lower().encode()).hexdigest(), 16)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _row(system: str, question: str, tool: str, arguments: dict[str, Any] | None,
         reply: str | None, tools: list[dict[str, Any]], seed_question: str) -> dict[str, Any]:
    if tool == CHAT:
        assistant: dict[str, Any] = {"role": "assistant", "content": reply}
    else:
        # arguments as a mapping, which is what a chat template renders; the OpenAI wire
        # shape carries them as a JSON string, and functiongemma's template prints that raw
        assistant = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": tool, "arguments": dict(arguments or {})}}]}
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": question}, assistant],
            "tools": tools, "tool": tool, "from": seed_question,
            "split": "holdout" if _hash(seed_question) % HOLDOUT_EVERY == 0 else "train"}


def _asked(ask: Callable[[str], str], schema: Mapping[str, Any] | None,
           examples: Sequence[Example], count: int) -> list[Example]:
    """More examples from a served model, the known ones as few-shots. Bad lines are dropped."""
    if schema is None:
        shots = "\n".join(json.dumps({"question": e.question}) for e in examples[:8])
        prompt = (f"These are messages a user sends an assistant that needs no tool call at "
                  f"all — greetings, asides, small talk, questions about the world:\n{shots}\n"
                  f"Write {count} more, one JSON object per line, {{\"question\": ...}}, and "
                  "nothing else.")
    else:
        fn = _fn(schema)
        shots = "\n".join(json.dumps({"question": e.question, "arguments": e.arguments})
                          for e in examples if e.question and e.arguments is not None)[:4000]
        prompt = (f"A tool called {fn['name']}: {fn.get('description', '')}\n"
                  f"Its parameters: {json.dumps(fn.get('parameters') or {})}\n"
                  f"Questions a user asks that this tool answers, with the call:\n{shots}\n"
                  f"Write {count} more, varied in wording and subject, one JSON object per "
                  "line, {\"question\": ..., \"arguments\": {...}}, and nothing else.")
    text = ask(prompt) or ""
    props = set(_properties(schema)) if schema is not None else set()
    required = set(_required(schema)) if schema is not None else set()
    out: list[Example] = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        if schema is None:
            out.append(Example(question, CHAT, None))
            continue
        args = row.get("arguments")
        if not isinstance(args, dict) or not set(args) <= props or not required <= set(args):
            continue
        out.append(Example(question, _fn(schema)["name"], args))
    return out


def synthesise(tools: Any, *, prompts: Mapping[str, Sequence[str]] | None = None,
               per_tool: int = 40, seed: int = 0, ask: Callable[[str], str] | None = None,
               system: str = SYSTEM, chat_share: float = CHAT_SHARE) -> list[dict[str, Any]]:
    """Chat conversations that call these tools, and some that do not.

    Each row is ``{"messages": [system, user, assistant], "tools": [...], "tool": name or
    "chat", "from": the seed question, "split": "train" | "holdout"}``. The assistant turn
    carries ``tool_calls`` with arguments as a mapping, or a sentence of content for a
    turn that calls nothing. Reproducible for a ``seed``; ``ask`` is a served model,
    ``(prompt) -> str``, asked for more questions per tool with the examples as few-shots.

    One seed question in ten is held out, by hash, and every paraphrase of it goes with it:
    a paraphrase in training and its original in the holdout would score the memorising,
    not the calling.
    """
    schemas = schemas_of(tools)
    if not schemas:
        raise ValueError("no tools to synthesise from")
    by_name = {_fn(s)["name"]: s for s in schemas}
    rng = random.Random(seed)
    examples = examples_in(schemas, prompts)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(question: str, tool: str, arguments: dict[str, Any] | None, reply: str | None,
            seed_question: str) -> bool:
        key = _norm(question)
        if not key or key in seen:
            return False
        seen.add(key)
        rows.append(_row(system, question, tool, arguments, reply, schemas, seed_question))
        return True

    for name, schema in by_name.items():
        own = [e for e in examples if e.tool == name]
        seeds = [e.arguments for e in own if e.arguments is not None]
        pairs = [(e.question, e.arguments) for e in own if e.question and e.arguments is not None]
        asked_for = [e.question for e in own if e.question and e.arguments is None]
        if ask is not None:
            more = _asked(ask, schema, own, per_tool)
            pairs.extend((e.question, e.arguments) for e in more)
            seeds.extend(e.arguments for e in more if e.arguments is not None)
        pairs.extend((q, _fill(schema, q, seeds, rng)) for q in asked_for)
        if not pairs:
            continue
        # the originals first, so a tool with one example still teaches it verbatim
        for question, arguments in pairs:
            add(question, name, arguments, None, question)
        wordings = [(q, a, v) for q, a in pairs for v in _variants(q)[1:]]
        rng.shuffle(wordings)
        made = sum(1 for r in rows if r["tool"] == name)
        for question, arguments, variant in wordings:
            if made >= per_tool:
                break
            made += add(variant, name, arguments, None, question)

    chat_seeds = [e.question for e in examples if e.tool == CHAT] or list(_ASIDES)
    if ask is not None:
        chat_seeds.extend(e.question for e in _asked(ask, None, [Example(q, CHAT, None)
                                                                  for q in chat_seeds], per_tool))
    wanted_chat = max(len(chat_seeds), int(len(rows) * chat_share / max(1e-9, 1 - chat_share)))

    def reply(question: str) -> str:
        kind = ("thanks" if _THANKS.search(question) else
                "greeting" if _GREETING.search(question) else "other")
        return rng.choice(_REPLIES[kind])

    made = 0
    for question in chat_seeds:
        made += add(question, CHAT, None, reply(question), question)
    chat_wordings = [(q, v) for q in chat_seeds for v in _variants(q)[1:]]
    rng.shuffle(chat_wordings)
    for question, variant in chat_wordings:
        if made >= wanted_chat:
            break
        made += add(variant, CHAT, None, reply(question), question)

    return rows


def split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(train, holdout)`` as the rows' ``split`` field says."""
    train = [dict(r) for r in rows if r.get("split") != "holdout"]
    holdout = [dict(r) for r in rows if r.get("split") == "holdout"]
    return train, holdout


def write_dataset(out: Path, rows: Sequence[Mapping[str, Any]], *, base: str,
                  **manifest: Any) -> dict[str, Any]:
    """``train.jsonl``, ``holdout.jsonl`` and a ``manifest.json`` naming the base model.

    The manifest is how the ``tool-calls`` recipe knows which chat template to render
    with: a conversation set is made *for* a model, and the recipe reads that here rather
    than asking again.
    """
    out.mkdir(parents=True, exist_ok=True)
    train, holdout = split(rows)
    for name, part in (("train.jsonl", train), ("holdout.jsonl", holdout)):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    summary = {"base": base, "rows": len(rows), "train": len(train), "holdout": len(holdout),
               "per_tool": _counts(rows), **manifest}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


# -- from what a model actually did -------------------------------------------------------

def _scored(row: Mapping[str, Any]) -> float:
    """F1 for one kept bench row, as the bench scores it."""
    from ml_stack.graph.bench.score import _hit

    return float(_hit(row))


def traced_rows(kept: Sequence[Mapping[str, Any]], *, model: str = "",
                min_f1: float = 0.8) -> list[dict[str, Any]]:
    """Every scored question in these runs that kept its transcript and scored well enough.

    ``model`` is a substring of the run's label or of the model file the server was
    holding, because a run is named for how it was asked (``e4b-shortlist``) and the file
    is what was actually loaded; either identifies it. A question with nothing expected is
    left out -- it was never scored, so "above the threshold" means nothing about it.
    """
    out = []
    for one in kept:
        server = one.get("server") or {}
        named = f"{one.get('label', '')} {server.get('model', '')}"
        if model and model.lower() not in named.lower():
            continue
        for row in one.get("rows") or ():
            if not row.get("trace") or not row.get("expected") or row.get("timed_out"):
                continue
            if _scored(row) + 1e-9 < min_f1:
                continue
            out.append({**row, "run": one.get("key", ""), "run_label": one.get("label", "")})
    return out


def would_yield(kept: Sequence[Mapping[str, Any]], *, model: str = "",
                min_f1: float = 0.8) -> dict[str, int]:
    """What these runs *would* have yielded had they been traced: questions, and turns.

    One training example per model turn, and a question's turns are its calls -- so the
    count a run of untraced rows would have given is the sum of their ``calls``. Written
    down because the first answer this command gives on a store filled before tracing
    existed is zero, and zero is worth nothing without the number beside it: today's
    thousands of scored tool calls are recoverable only by spending the GPU again.
    """
    questions = turns = traced = 0
    for one in kept:
        server = one.get("server") or {}
        named = f"{one.get('label', '')} {server.get('model', '')}"
        if model and model.lower() not in named.lower():
            continue
        for row in one.get("rows") or ():
            if not row.get("expected") or row.get("timed_out"):
                continue
            if _scored(row) + 1e-9 < min_f1:
                continue
            questions += 1
            turns += int(row.get("calls") or 0)
            traced += 1 if row.get("trace") else 0
    return {"questions": questions, "turns": turns, "traced": traced}


def _as_message(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """One trace entry as the chat message a recipe renders, or None for what is not one."""
    role = str(entry.get("role") or "")
    if role in ("system", "user"):
        return {"role": role, "content": str(entry.get("content") or "")}
    if role == "tool":
        return {"role": "tool", "name": str(entry.get("name") or ""),
                "content": str(entry.get("content") or "")}
    return None


def _target(entry: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
    """The assistant turn a trace entry teaches, and the tool it calls; None for no lesson.

    A turn that called nothing and wrote nothing teaches nothing, and a turn the ceiling
    cut off (``finish`` of ``length``) teaches a truncated call -- the one thing a tool
    caller must never learn. Both are dropped rather than trained on.
    """
    calls = list(entry.get("tool_calls") or ())
    text = str(entry.get("content") or "")
    if str(entry.get("finish") or "") == "length" or entry.get("cut"):
        return None
    if calls:
        return ({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": str(c.get("name") or ""),
                          "arguments": dict(c.get("args") or {})}}
            for i, c in enumerate(calls)]}, str(calls[0].get("name") or ""))
    if text.strip():
        return {"role": "assistant", "content": text}, CHAT
    return None


def examples_from(row: Mapping[str, Any], *, system: str = "") -> list[dict[str, Any]]:
    """One traced question as one training example per model turn.

    The conversation up to a turn is the input -- the system prompt, the question, and
    every tool result the model had already been given -- and the turn itself is the
    target. A question of four calls is four examples, each one a decision made with
    strictly more evidence than the last, which is the thing being taught: not *a* call
    but the next one, given what came back.

    The tools each example carries are the ones that were actually offered on that call:
    `graph.ask` takes tools away as a question goes on, and an example that offers a tool
    the model was not offered teaches it to reach for something that will not be there.
    """
    schemas: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    question = str(row.get("question") or "")
    for entry in row.get("trace") or ():
        if str(entry.get("role") or "") == "tools":
            schemas = schemas_of(entry.get("tools") or [])
            continue
        message = _as_message(entry)
        if message is not None:
            conversation.append(message)
            continue
        if str(entry.get("role") or "") != "assistant":
            continue
        offered = [str(n) for n in (entry.get("offered") or ())]
        taught = _target(entry)
        if taught is not None:
            assistant, tool = taught
            usable = [s for s in schemas if not offered or _fn(s)["name"] in offered]
            out.append({"messages": [*conversation, assistant], "tools": usable,
                        "tool": tool, "from": question, "call": int(entry.get("call") or 0),
                        "run": str(row.get("run") or ""),
                        "model": str(entry.get("model") or ""),
                        "split": "holdout" if _hash(question) % HOLDOUT_EVERY == 0 else "train"})
        # whether or not it was taught, the model said it, and the next turn saw it
        calls = list(entry.get("tool_calls") or ())
        conversation.append({"role": "assistant", "content": str(entry.get("content") or "") or None,
                             **({"tool_calls": [
                                 {"id": f"call_{i}", "type": "function",
                                  "function": {"name": str(c.get("name") or ""),
                                               "arguments": dict(c.get("args") or {})}}
                                 for i, c in enumerate(calls)]} if calls else {})})
    if system:
        for one in out:
            if not one["messages"] or one["messages"][0].get("role") != "system":
                one["messages"] = [{"role": "system", "content": system}, *one["messages"]]
    return out


def from_bench(kept: Sequence[Mapping[str, Any]], *, model: str = "", min_f1: float = 0.8,
               system: str = "") -> list[dict[str, Any]]:
    """Training rows from kept bench runs: every good traced question, turn by turn.

    The rows are the shape `synthesise` writes and the ``tool-calls`` recipe reads, so the
    two sources mix in one directory: synthetic conversations teach the shape of a call,
    and these teach the calls that actually scored, on a real graph, with real ids.
    """
    return [example for row in traced_rows(kept, model=model, min_f1=min_f1)
            for example in examples_from(row, system=system)]


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get("tool"))] = out.get(str(r.get("tool")), 0) + 1
    return dict(sorted(out.items()))


# -- the command -------------------------------------------------------------------------------

def load_tools(spec: str) -> Any:
    """A JSON file of schemas, or ``python:module:attribute`` imported live.

    The attribute may be the schemas, ``(schema, callable)`` pairs, a prompts mapping, or a
    function of no arguments that returns one of those (``ml_stack.web:tools``).
    """
    if spec.startswith("python:"):
        parts = spec.split(":")
        module, attribute = (parts[1], parts[2]) if len(parts) == 3 else ("", "")
        if not module or not attribute:
            raise ValueError(f"expected python:module:attribute, got {spec!r}")
        try:
            value = getattr(importlib.import_module(module), attribute)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot import {spec}: {exc}") from exc
        return value() if callable(value) else value
    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no such file {path}; pass a JSON file or python:module:attr")
    return json.loads(path.read_text())


def _asker(url: str) -> Callable[[str], str]:
    from ml_stack.client import Client

    client = Client(base_url=url)

    def ask(prompt: str) -> str:
        reply = client.chat([{"role": "user", "content": prompt}])
        return reply.content or ""

    return ask


def _settings(pairs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


FROM_BENCH = ("Training data from what a model actually did: every traced question a bench "
              "run kept that scored well enough, as one example per model turn -- the "
              "conversation up to that turn as the input, the call the model made as the "
              "target. Runs are traced by default at 20 questions or fewer "
              "(ml-stack-bench); an untraced run yields nothing, and this says what it "
              "would have yielded.")


def _from_bench_arguments(ap: Any) -> Any:
    """``from-bench``'s flags, onto either parser that has to carry them.

    There are two: the one this subcommand is parsed by, and the one registered under the
    top-level parser so that ``--help`` names it and the documentation check can find its
    flags. Written once so the two cannot drift.
    """
    ap.add_argument("--kept", default="", metavar="STORE",
                    help="the bench store the runs are in (default: ~/.ml-stack/bench/"
                         "runs.ladybug, or MLSTACK_BENCH_HOME's)")
    ap.add_argument("--model", default="", metavar="SUBSTRING",
                    help="only runs whose label or served model file contains this, e.g. "
                         "e4b. Mixing two models' turns into one dataset teaches the "
                         "average of two callers")
    ap.add_argument("--min-f1", type=float, default=0.8, metavar="F1",
                    help="only questions that scored at least this, 0-1 (default: "
                         "%(default)s). A wrong answer's tool calls are exactly what must "
                         "not be learned")
    ap.add_argument("--label", default="", metavar="SUBSTRING",
                    help="only runs whose label contains this (narrower than --model)")
    ap.add_argument("--system", default="", metavar="TEXT",
                    help="a system prompt to put in front of any conversation that has "
                         "none; by default the trace's own is used, which is the one the "
                         "model was actually served")
    ap.add_argument("--base", default="google/functiongemma-270m-it",
                    help="what the manifest names as the model this data is rendered for")
    ap.add_argument("--out", required=True, metavar="FILE.jsonl",
                    help="one JSONL file of rows, or a directory -- which gets train.jsonl, "
                         "holdout.jsonl and manifest.json, ready for ml-stack-train-run "
                         "--recipe tool-calls --data")
    ap.add_argument("--dry-run", action="store_true",
                    help="count what would be written and write nothing")
    return ap


def _from_bench_parser() -> Any:
    import argparse

    return _from_bench_arguments(argparse.ArgumentParser(
        prog="ml-stack-train-tools from-bench", allow_abbrev=False, description=FROM_BENCH))


def _from_bench(argv: list[str]) -> int:
    """``ml-stack-train-tools from-bench``: kept bench traces into a training file."""
    from ml_stack.graph import bench

    a = _from_bench_parser().parse_args(argv)
    store = Path(a.kept).expanduser() if a.kept else bench.HOME / "runs.ladybug"
    if not store.exists():
        raise FileNotFoundError(f"no bench store at {store}; run ml-stack-bench first")
    kept = [r for r in bench.runs(store) if not a.label or a.label in str(r.get("label") or "")]
    rows = from_bench(kept, model=a.model, min_f1=a.min_f1, system=a.system)
    could = would_yield(kept, model=a.model, min_f1=a.min_f1)
    print(f"from-bench: {len(kept)} run(s) in {store}, "
          f"{could['questions']} question(s) at or above F1 {a.min_f1:g}"
          + (f" for {a.model!r}" if a.model else "")
          + f", {could['traced']} of them traced")
    if not rows:
        print(f"from-bench: 0 examples. Those questions made {could['turns']} model turns "
              f"between them, which is what a traced run of the same questions would have "
              f"yielded (one example per turn). Trace the next run: a run of "
              f"{bench.SHORT} questions or fewer traces by default, and "
              f"{bench.TRACE_ENV}=1 traces one of any size.")
        return 0
    train, holdout = split(rows)
    counts = _counts(rows)
    print(f"from-bench: {len(rows)} examples from {could['traced']} traced question(s) "
          f"({len(train)} train, {len(holdout)} held out): "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))
    out = Path(a.out).expanduser()
    if a.dry_run:
        print(f"from-bench: would write {out}")
        return 0
    if out.suffix == ".jsonl":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        print(f"from-bench: wrote {out}")
    else:
        summary = write_dataset(out, rows, base=a.base, source=str(store), model=a.model,
                                min_f1=a.min_f1, traced_questions=could["traced"])
        print(f"from-bench: wrote {out}/train.jsonl, holdout.jsonl, manifest.json "
              f"({summary['train']} + {summary['holdout']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "from-bench":
        try:
            return _from_bench(argv[1:])
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    ap = argparse.ArgumentParser(
        prog="ml-stack-train-tools", allow_abbrev=False,
        description="Plug in a project's tools, make training data from them, fine-tune a "
                    "base model, end with a GGUF. Each stage is skipped when --out has it. "
                    "`ml-stack-train-tools from-bench --help` builds the same data out of "
                    "what a model actually did, from the traces a bench run kept.")
    ap.add_argument("--tools", required=True,
                    help="JSON list of tool schemas, or python:module:attr "
                         "(e.g. python:ml_stack.graph.ask:TOOLS)")
    ap.add_argument("--prompts", default="",
                    help="JSON {tool: [question, ...]} or python:module:attr "
                         "(e.g. python:ml_stack.graph.ask:TOOL_PROMPTS); a 'chat' key is "
                         "the messages that want no tool")
    ap.add_argument("--base", default="google/functiongemma-270m-it",
                    help="Hugging Face id or a local directory to fine-tune")
    ap.add_argument("--out", required=True, help="where data/, run/ and the GGUF go")
    ap.add_argument("--ask", default="", metavar="URL",
                    help="a served model to write more questions per tool")
    ap.add_argument("--per-tool", type=int, default=40, help="conversations per tool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="a tool-calls recipe setting: steps, context, batch_size, "
                         "learning_rate")
    ap.add_argument("--quant", default="Q8_0", help="the GGUF quantisation to end with")
    ap.add_argument("--only", choices=("synth", "train", "export"), default="",
                    help="run one stage")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan with counts; load no model, write nothing")
    # Registered so that --help names it and a reader finds its flags where they look for
    # them. It is not parsed through here: the three stages' --tools and --out are required
    # at this level, and from-bench takes neither. Both carry the same arguments from
    # `_from_bench_arguments`, so the two cannot describe different commands.
    _from_bench_arguments(ap.add_subparsers(dest="cmd", metavar="{from-bench}")
                          .add_parser("from-bench", allow_abbrev=False,
                                      description=FROM_BENCH,
                                      help="training data out of a bench run's traces"))
    a = ap.parse_args(argv)

    try:
        return _run(a)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:                       # ToolNotFound, ConversionError
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(a: Any) -> int:
    from ml_stack.train.recipes import validate

    out = Path(a.out).expanduser()
    data, run_dir = out / "data", out / "run"
    stages = (a.only,) if a.only else ("synth", "train", "export")
    config = validate("tool-calls", _settings(a.set))
    summary: dict[str, Any] = {"out": str(out), "base": a.base, "dry_run": a.dry_run}

    if "synth" in stages:
        if (data / "train.jsonl").exists() and not a.dry_run:
            manifest = json.loads((data / "manifest.json").read_text())
            print(f"synth: {manifest['rows']} rows already in {data}, skipping")
        else:
            tools = schemas_of(load_tools(a.tools))
            prompts = load_tools(a.prompts) if a.prompts else None
            ask = _asker(a.ask) if a.ask and not a.dry_run else None
            rows = synthesise(tools, prompts=prompts, per_tool=a.per_tool, seed=a.seed, ask=ask)
            train, holdout = split(rows)
            counts = _counts(rows)
            print(f"synth: {len(rows)} conversations over {len(tools)} tools "
                  f"({len(train)} train, {len(holdout)} held out): "
                  + ", ".join(f"{k} {v}" for k, v in counts.items()))
            manifest = {"rows": len(rows), "train": len(train), "holdout": len(holdout),
                        "per_tool": counts}
            if not a.dry_run:
                manifest = write_dataset(data, rows, base=a.base, tools=a.tools,
                                         prompts=a.prompts, seed=a.seed, asked=bool(a.ask))
                print(f"synth: wrote {data}/train.jsonl, holdout.jsonl, manifest.json")
            elif a.ask:
                print(f"synth: would also ask {a.ask} for {a.per_tool} more per tool")
        summary["synth"] = manifest

    if "train" in stages:
        from ml_stack.train.checkpoint import find_latest, load_state
        from ml_stack.train.recipes.tool_calls import device_for

        latest = find_latest(run_dir)
        done = load_state(latest).step if latest else 0
        plan = {"base": a.base, "steps": config["steps"], "context": config["context"],
                "batch_size": config["batch_size"], "learning_rate": config["learning_rate"],
                "device": str(device_for()), "resumed_from": done}
        if done >= config["steps"]:
            print(f"train: {run_dir} already at step {done}, skipping")
        elif a.dry_run:
            print(f"train: would fine-tune {a.base} for {config['steps']} steps of "
                  f"{config['batch_size']} on {plan['device']}, context {config['context']}, "
                  f"lr {config['learning_rate']}, into {run_dir}"
                  + (f" (resuming from {done})" if done else ""))
        else:
            from ml_stack.train.run import run

            if not (data / "train.jsonl").exists():
                raise FileNotFoundError(f"no {data}/train.jsonl; run the synth stage first")
            print(f"train: {a.base} for {config['steps']} steps on {plan['device']}"
                  + (f", resuming from {done}" if done else ""))
            plan.update(run("tool-calls", config, data, run_dir))
            print(f"train: final loss {plan['final_loss']:.3f}, best held-out "
                  f"{plan['best_metric']}, {plan['seconds']}s")
        summary["train"] = plan

    if "export" in stages:
        from ml_stack.gguf.tools import find_converter, find_quantize

        existing = sorted(out.glob("*.gguf"))
        converter, quantizer = find_converter(), find_quantize()
        plan = {"quant": a.quant, "converter": str(converter or ""),
                "quantize": str(quantizer or "")}
        if existing:
            print(f"export: {existing[0]} already exists, skipping")
            plan["gguf"] = str(existing[0])
        elif a.dry_run:
            print(f"export: would write {out / 'model'} in Hugging Face layout, then "
                  f"{a.quant} GGUF via " + (str(converter) if converter else
                                            "convert_hf_to_gguf.py, which is NOT installed"))
        else:
            from ml_stack.gguf import export
            from ml_stack.train.recipes.tool_calls import save_pretrained

            saved = save_pretrained(run_dir, a.base, out / "model")
            print(f"export: wrote {saved}")
            result = export(saved, out, name=f"{Path(a.base).name}-tools", quant=a.quant,
                            fix_space_prefix=None)
            print(f"export: {result.path} ({result.size_mb:.0f} MB)")
            plan["gguf"] = str(result.path)
        summary["export"] = plan

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
