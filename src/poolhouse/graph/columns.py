"""The JSON columns a graph store keeps: writing one, reading one back, and naming the
record and column that refused."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ["as_json", "blame", "column", "differences", "from_json", "refused", "shown"]


def as_json(value: Any) -> str:
    """A value as the JSON the store keeps, or a ValueError naming what will not encode.

    Objects go through ``str`` -- a Path, a dataclass, a numpy scalar. What is left
    unencodable is a key that is not a string, or a value that refers to itself.
    """
    try:
        return json.dumps(value or {}, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as why:
        raise ValueError(f"{blame(value)}: {why}") from why


def blame(value: Any, path: str = "value", seen: frozenset[int] = frozenset()) -> str:
    """The path to the first thing in ``value`` that ``json.dumps`` refuses: a key that is
    not a string, a container that holds itself, or a leaf ``str`` cannot render."""
    if isinstance(value, (Mapping, list, tuple)):
        if id(value) in seen:
            return f"{path} (refers to itself)"
        seen = seen | {id(value)}
    if isinstance(value, Mapping):
        for key, inner in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                return f"{path}[{key!r}]"
            found = blame(inner, f"{path}.{key}" if isinstance(key, str) else f"{path}[{key!r}]",
                          seen)
            if found:
                return found
        return ""
    if isinstance(value, (list, tuple)):
        for n, inner in enumerate(value):
            found = blame(inner, f"{path}[{n}]", seen)
            if found:
                return found
        return ""
    try:
        json.dumps(value, default=str)
    except (TypeError, ValueError):
        return path
    return ""


def from_json(raw: Any) -> dict[str, Any]:
    """A JSON column as a dict. None, "" and null read as {}; anything else that is not an
    object, or is not JSON at all, raises ValueError."""
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            out = json.loads(raw)
        except ValueError:
            raise ValueError(f"not JSON: {raw[:40]!r}") from None
    else:
        out = raw
    if out is None:
        return {}
    if not isinstance(out, dict):
        kind = type(out).__name__
        raise ValueError(f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}, not an object")
    return out


def column(raw: Any, what: str) -> dict[str, Any]:
    """`from_json`, with the record and column named when it refuses."""
    try:
        return from_json(raw)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None


def refused(raw: Any, what: str) -> list[str]:
    """One line naming the record and column when it does not hold a JSON object; else none."""
    try:
        column(raw, what)
    except ValueError as exc:
        return [str(exc)]
    return []


def shown(value: Any) -> str:
    """A value short enough for one line: long strings by their length."""
    if isinstance(value, str) and len(value) > 40:
        return f"{len(value)} chars"
    return repr(value)


def differences(a: Mapping[str, Any], b: Mapping[str, Any], a_name: str = "written",
                b_name: str = "read") -> str:
    """The columns two rows disagree on, as ``col (written X, read Y)``, on one line."""
    parts = [f"{col} ({a_name} {shown(a.get(col))}, {b_name} {shown(b.get(col))})"
             for col in sorted(set(a) | set(b)) if a.get(col) != b.get(col)]
    return ", ".join(parts) or "nothing, yet the rows differ"
