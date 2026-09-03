"""Keeping a model's tool arguments to ids the graph holds.

`constrained` pins every id field in an offer of tool schemas to a list of ids;
`call_schema` is the JSON Schema one turn answers under -- one tool call out of that offer,
or an answer -- which llama.cpp turns into the grammar it samples with when it is sent as
the request's ``response_format``; `ids_grammar` is the same constraint as GBNF text, for a
caller driving ``/completion`` with a ``grammar`` of its own; `call_from` reads the JSON a
constrained turn came back as.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ml_stack.contracts.jsonschema import PRIMITIVES, _Builder, _literal

# Over this many ids no constraint is built: every id is an alternative in the grammar, and
# the server compiles the grammar again for every request that carries it.
CAP = 2000

# The argument fields that hold an entry id, by tool.
ID_FIELDS: dict[str, tuple[str, ...]] = {
    "look_at": ("ids",),
    "look_around": ("ids",),
    "show": ("ids",),
    "path_between": ("from_id", "to_id"),
}

ANSWER: dict[str, Any] = {"type": "object", "properties": {"answer": {"type": "string"}},
                          "required": ["answer"]}

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _name(schema: Mapping[str, Any]) -> str:
    return str((schema.get("function") or {}).get("name") or "")


def constrained(schemas: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> list[dict[str, Any]]:
    """Copies of ``schemas`` whose id fields accept only ``ids``; the rest are the same objects."""
    out: list[dict[str, Any]] = []
    for schema in schemas:
        fields = ID_FIELDS.get(_name(schema))
        if not fields:
            out.append(schema)  # type: ignore[arg-type]
            continue
        told = copy.deepcopy(dict(schema))
        properties = told["function"].setdefault("parameters", {}).setdefault("properties", {})
        for field in fields:
            one = properties.get(field)
            if not isinstance(one, dict):
                continue
            if one.get("type") == "array":
                one["items"] = {"enum": list(ids)}
            else:
                properties[field] = {"enum": list(ids),
                                     **({"description": one["description"]}
                                        if "description" in one else {})}
        out.append(told)
    return out


def call_schema(schemas: Sequence[Mapping[str, Any]], ids: Iterable[str]) -> dict[str, Any] | None:
    """The JSON Schema a constrained turn answers under: one call to a tool in ``schemas``,
    with every id field pinned to ``ids``, or an answer. None when nothing offered takes an
    id, when there are no ids, or when there are more than `CAP`."""
    known: list[str] = []
    for one in ids:
        if str(one) not in known:
            known.append(str(one))
    if not known or len(known) > CAP:
        return None
    if not any(_name(s) in ID_FIELDS for s in schemas):
        return None
    calls = []
    for schema in constrained(schemas, known):
        params = (schema.get("function") or {}).get("parameters") or {"type": "object",
                                                                        "properties": {}}
        calls.append({"type": "object",
                      "properties": {"name": {"const": _name(schema)}, "arguments": params},
                      "required": ["name", "arguments"]})
    return {"anyOf": [*calls, ANSWER]}


def response_format(schema: Mapping[str, Any]) -> dict[str, Any]:
    """``schema`` as the ``response_format`` field of a chat completion."""
    return {"type": "json_schema", "json_schema": {"name": "turn", "schema": dict(schema)}}


def call_from(content: str | None) -> tuple[list[dict[str, Any]] | None, str | None]:
    """``(tool_calls, answer)`` read out of a constrained reply; both None when it is neither."""
    text = (content or "").strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    if not text.startswith("{"):
        return None, None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    name = parsed.get("name")
    if isinstance(name, str) and name:
        args = parsed.get("arguments")
        if not isinstance(args, dict):
            args = {}
        return [{"id": "c1", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}], None
    if "answer" in parsed:
        return None, str(parsed.get("answer") or "")
    return None, None


def ids_grammar(ids: Iterable[str], schemas: Sequence[Mapping[str, Any]] | None = None) -> str:
    """The GBNF a constrained turn samples under, or "" when there is nothing to constrain
    or more than `CAP` ids. ``schemas`` is the offer; every built-in tool by default."""
    if schemas is None:
        from ml_stack.graph.ask import TOOLS
        schemas = TOOLS
    schema = call_schema(schemas, ids)
    if schema is None:
        return ""
    builder = _Rules(schema)
    root = builder.rule_for(schema)
    lines = [f"root ::= {root}"]
    lines += [f"{name} ::= {body}" for name, body in builder.rules]
    lines += [f"{name} ::= {PRIMITIVES[name]}"
              for name in PRIMITIVES if name in builder.reached()]
    return "\n".join(lines) + "\n"


class _Rules(_Builder):
    """`grammar_for`'s builder, reading ``const``, ``anyOf`` and optional properties too."""

    def rule_for(self, schema: Any) -> str:
        if isinstance(schema, dict):
            if "const" in schema:
                return self.enum([schema["const"]])
            if "anyOf" in schema:
                name = self.fresh("either")
                self.rules.append((name, " | ".join(self.rule_for(s) for s in schema["anyOf"])))
                return name
        return super().rule_for(schema)

    def obj(self, schema: dict[str, Any]) -> str:
        name = self.fresh("object")
        ws = self.use("ws")
        properties = schema.get("properties") or {}
        required = [k for k in properties if k in set(schema.get("required") or ())]
        optional = [k for k in properties if k not in required]
        comma = f"{ws} {_literal(',')} {ws}"

        def member(key: str) -> str:
            return (f"{_literal(json.dumps(str(key)))} {ws} {_literal(':')} {ws} "
                    f"{self.rule_for(properties[key])}")

        if required:
            body = f" {comma} ".join(member(k) for k in required)
            body += "".join(f" ({comma} {member(k)})?" for k in optional)
        elif optional:
            body = "(" + " | ".join(member(k) for k in optional) + ")?"
        else:
            body = ""
        self.rules.append((name, f"{_literal('{')} {ws} {body} {ws} {_literal('}')}".replace("  ", " ")))
        return name
