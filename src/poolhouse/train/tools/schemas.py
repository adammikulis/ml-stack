"""Tool schemas whatever shape they arrive in, and the worked examples their descriptions
carry."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CHAT = "chat"
"""The prompts key for messages that want no tool — the same key ``graph.prompts.TOOL_PROMPTS``
uses, so a project's router examples can be handed over as they are."""


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
