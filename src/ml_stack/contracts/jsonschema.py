"""Turn a JSON Schema into the GBNF grammar llama.cpp constrains decoding with."""

from __future__ import annotations

import json
from typing import Any

from ml_stack.contracts.loader import ContractError

__all__ = ["grammar_for"]

SCALARS = ("string", "integer", "number", "boolean", "null")

PRIMITIVES: dict[str, str] = {
    "ws": r"[ \t\n]*",
    "string": r'"\"" char* "\""',
    "char": r'[^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)',
    "hex": r"[0-9a-fA-F]",
    "integer": r'"-"? ("0" | [1-9] [0-9]*)',
    "number": r'integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
    "boolean": r'"true" | "false"',
    "null": r'"null"',
}

NEEDS: dict[str, tuple[str, ...]] = {
    "string": ("char",),
    "char": ("hex",),
    "number": ("integer",),
}


def grammar_for(schema: dict[str, Any]) -> str:
    """The GBNF text constraining a completion to ``schema``."""
    if not isinstance(schema, dict):
        raise ContractError(
            f"a JSON Schema is an object, not {type(schema).__name__}")
    builder = _Builder(schema)
    root = builder.rule_for(schema)
    lines = [f"root ::= {root}"]
    lines += [f"{name} ::= {body}" for name, body in builder.rules]
    lines += [f"{name} ::= {PRIMITIVES[name]}"
              for name in PRIMITIVES if name in builder.reached()]
    return "\n".join(lines) + "\n"


class _Builder:
    """Names and bodies for every rule one schema turns into."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.defs: dict[str, Any] = {}
        for section in ("$defs", "definitions"):
            found = schema.get(section)
            if isinstance(found, dict):
                self.defs.update(found)
        self.rules: list[tuple[str, str]] = []
        self.used: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.count = 0

    def reached(self) -> set[str]:
        """Every primitive rule used, plus the ones those are written in terms of."""
        out = set(self.used)
        pending = list(out)
        while pending:
            for need in NEEDS.get(pending.pop(), ()):
                if need not in out:
                    out.add(need)
                    pending.append(need)
        return out

    def fresh(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}-{self.count}"

    def use(self, name: str) -> str:
        self.used.add(name)
        return name

    def rule_for(self, schema: Any) -> str:
        """The name of the rule matching ``schema``, defining it if it is new."""
        if not isinstance(schema, dict):
            raise ContractError(f"schema fragment is not an object: {schema!r}")
        if "$ref" in schema:
            return self.alias(str(schema["$ref"]))
        if "enum" in schema:
            return self.enum(schema["enum"])

        kind = schema.get("type")
        if kind == "object":
            return self.obj(schema)
        if kind == "array":
            return self.arr(schema)
        if isinstance(kind, str) and kind in SCALARS:
            return self.use(kind)
        raise ContractError(
            f"unsupported schema type {kind!r}; grammar_for handles "
            + ", ".join(("object", "array", *SCALARS))
        )

    def alias(self, ref: str) -> str:
        key = ref.rsplit("/", 1)[-1]
        if not ref.startswith("#/") or key not in self.defs:
            raise ContractError(
                f"cannot resolve {ref!r}; grammar_for reads local $defs only, and has "
                + (", ".join(sorted(self.defs)) or "none")
            )
        known = self.aliases.get(key)
        if known is not None:
            return known
        name = self.fresh("def")
        self.aliases[key] = name
        self.rules.append((name, self.rule_for(self.defs[key])))
        return name

    def enum(self, values: Any) -> str:
        if not isinstance(values, list) or not values:
            raise ContractError(f"enum must be a non-empty list, got {values!r}")
        name = self.fresh("enum")
        self.rules.append((name, " | ".join(_literal(json.dumps(v)) for v in values)))
        return name

    def obj(self, schema: dict[str, Any]) -> str:
        name = self.fresh("object")
        ws = self.use("ws")
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            raise ContractError(f"properties must be an object, got {properties!r}")

        parts = [_literal("{"), ws]
        for index, (key, sub) in enumerate(properties.items()):
            if index:
                parts += [_literal(","), ws]
            parts += [_literal(json.dumps(str(key))), ws, _literal(":"), ws,
                      self.rule_for(sub), ws]
        parts.append(_literal("}"))
        self.rules.append((name, " ".join(parts)))
        return name

    def arr(self, schema: dict[str, Any]) -> str:
        name = self.fresh("array")
        ws = self.use("ws")
        items = schema.get("items")
        if items is None:
            raise ContractError("an array schema needs items")
        item = self.rule_for(items)
        body = (f'{_literal("[")} {ws} ({item} ({ws} {_literal(",")} {ws} {item})*)? '
                f'{ws} {_literal("]")}')
        self.rules.append((name, body))
        return name


def _literal(text: str) -> str:
    """``text`` as a GBNF quoted literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
