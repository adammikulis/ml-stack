"""Chat conversations made out of the schemas: arguments filled in from what a question
says, every question reworded, and a share of turns that call nothing."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ml_stack.train.tools.dataset import side_of
from ml_stack.train.tools.schemas import (
    CHAT,
    Example,
    _fn,
    _properties,
    _required,
    examples_in,
    schemas_of,
)

SYSTEM = ("You answer questions with the tools you have been given. When a question needs "
          "one, call it with the arguments the question asks for; when it needs none, reply "
          "in a sentence.")

CHAT_SHARE = 0.15
"""The share of rows that call nothing. Measured need, not taste: a model given tools and
never shown a turn without one calls a tool on "hi"."""

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
            "split": side_of(seed_question)}


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
