"""Tool schemas for a small model: MCP tools turned into OpenAI function schemas, a model's
tool-call arguments parsed and repaired, and those arguments checked against the schema."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["PROFILES", "from_mcp", "index_by_name", "parse_arguments", "validate"]

PROFILES: dict[str, tuple[int, int]] = {"full": (10_000, 99), "lean": (200, 4), "tiny": (80, 3)}
"""``{profile: (longest description, deepest nesting)}``."""

_TYPES: dict[str, Any] = {"object": dict, "array": list, "string": str, "null": type(None),
                          "boolean": bool}


def _pruned(schema: Any, profile: str, depth: int = 0) -> Any:
    if not isinstance(schema, dict):
        return schema
    longest, deepest = PROFILES[profile]
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "description" and isinstance(value, str) and len(value) > longest:
            out[key] = value[:longest].rstrip() + "..."
        elif key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = ({} if depth >= deepest else
                        {k: _pruned(v, profile, depth + 1) for k, v in value.items()})
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            out[key] = _pruned(value, profile, depth + 1) if depth < deepest else {}
        else:
            out[key] = value
    return out


def from_mcp(tools: Iterable[Mapping[str, Any]], profile: str = "full") -> list[dict[str, Any]]:
    """OpenAI function schemas for MCP ``tools/list`` entries; ``lean`` and ``tiny`` trim
    descriptions and nesting for a small model."""
    if profile not in PROFILES:
        raise ValueError(f"profile {profile!r} is not one of {sorted(PROFILES)}")
    out = []
    for tool in tools:
        if not tool.get("name"):
            continue
        params = _pruned(tool.get("inputSchema") or tool.get("input_schema")
                         or {"type": "object", "properties": {}}, profile)
        if not isinstance(params, dict) or "type" not in params:
            params = {"type": "object", "properties": params if isinstance(params, dict) else {}}
        out.append({"type": "function", "function": {
            "name": tool["name"],
            "description": str(tool.get("description") or "")[:PROFILES[profile][0]],
            "parameters": params}})
    return out


def index_by_name(schemas: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{tool name: parameters schema}`` for OpenAI function schemas."""
    return {s["function"]["name"]: s["function"].get("parameters") or {}
            for s in schemas if s.get("type") == "function"}


_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_BARE_WORDS = ((re.compile(r"\bTrue\b"), "true"), (re.compile(r"\bFalse\b"), "false"),
               (re.compile(r"\bNone\b"), "null"))
_DECODER = json.JSONDecoder()


def _closed(text: str) -> str:
    """``text`` with unterminated strings, objects and arrays closed."""
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            escaped = ch == "\\" and not escaped
            if ch == '"' and not escaped:
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return text + ('"' if in_string else "") + "".join(reversed(stack))


def _variants(text: str) -> Iterable[str]:
    plain = _FENCE.sub("", text.strip()).strip()
    yield plain
    start = plain.find("{")
    inner = plain[start:] if start > 0 else plain
    yield inner
    fixed = _TRAILING_COMMA.sub(r"\1", inner)
    for pattern, word in _BARE_WORDS:
        fixed = pattern.sub(word, fixed)
    yield fixed
    yield fixed.replace("'", '"')
    yield _TRAILING_COMMA.sub(r"\1", _closed(fixed))


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str]:
    """``(arguments, "")`` from a model's tool-call arguments, or ``(None, why)``.

    A dict passes through and empty text is no arguments. Text is repaired when strict JSON
    fails: code fences and prose around the object, trailing commas, Python literals,
    single quotes, and a truncated tail are all handled.
    """
    if isinstance(raw, dict):
        return raw, ""
    if raw is None or not str(raw).strip():
        return {}, ""
    text = str(raw)
    for candidate in _variants(text):
        try:
            value, _end = _DECODER.raw_decode(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value, ""
    return None, f"arguments are not a JSON object: {text[:120]!r}"


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = _TYPES.get(kind)
    return True if expected is None else isinstance(value, expected)


def _object_errors(value: dict[str, Any], schema: Mapping[str, Any], path: str) -> list[str]:
    props = schema.get("properties") or {}
    errors = [f"{path or '<root>'}: missing required property {r!r}"
              for r in schema.get("required") or () if r not in value]
    if schema.get("additionalProperties") is False:
        errors += [f"{path or '<root>'}: unexpected property {k!r}" for k in value
                   if k not in props]
    for key, item in value.items():
        if key in props:
            errors += validate(item, props[key], f"{path}.{key}" if path else key)
    return errors


def validate(value: Any, schema: Mapping[str, Any], path: str = "") -> list[str]:
    """What is wrong with ``value`` against the JSON Schema subset tools declare: type,
    enum, required, additionalProperties false, numeric bounds, array bounds and items."""
    here = path or "<root>"
    kind = schema.get("type")
    kinds = [kind] if isinstance(kind, str) else list(kind or ())
    if kinds and not any(_type_ok(value, k) for k in kinds):
        return [f"{here}: expected {' or '.join(kinds)}, got {type(value).__name__}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{here}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        errors += _object_errors(value, schema, path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for bound, bad in (("minimum", lambda v, b: v < b), ("maximum", lambda v, b: v > b)):
            if bound in schema and bad(value, schema[bound]):
                errors.append(f"{here}: {value} violates {bound} {schema[bound]}")
    if isinstance(value, list):
        for bound, bad in (("minItems", lambda n, b: n < b), ("maxItems", lambda n, b: n > b)):
            if bound in schema and bad(len(value), schema[bound]):
                errors.append(f"{here}: {len(value)} items violates {bound} {schema[bound]}")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors += validate(item, schema["items"], f"{here}[{i}]")
    return errors
