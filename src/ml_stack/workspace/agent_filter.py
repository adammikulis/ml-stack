"""`--for-agent NAME`: narrow what a read command shows to one agent's records.

The filter selects records the caller may already read; it never sets a sender and never acts as
that agent. Private records (scratch folders) are guarded where they are read, not here."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ml_stack.workspace.identity import Denied


def resolve(names: Iterable[str], text: str) -> str:
    """The agent ``text`` names: an exact unique name, else the one name it is a prefix of.

    `Denied` lists the candidates when a prefix fits several, and says so when it fits none."""
    known = sorted(set(names))
    if text in known:
        return text
    found = [name for name in known if name.startswith(text)]
    if len(found) == 1:
        return found[0]
    if found:
        raise Denied(f"{text!r} fits several agents; give more of the name: {', '.join(found[:10])}")
    raise Denied(f"no agent is called {text!r} or starts with it")


def mentions(value: Any, name: str) -> bool:
    """Whether ``name`` appears as a whole value anywhere inside a record."""
    if isinstance(value, str):
        return value == name
    if isinstance(value, dict):
        return any(mentions(item, name) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(mentions(item, name) for item in value)
    return False


def narrow(result: Any, name: str) -> Any:
    """``result`` with every list of records cut to the ones that name ``name``."""
    if isinstance(result, list):
        return [row for row in result if mentions(row, name)]
    if isinstance(result, dict):
        return {key: narrow(value, name) if isinstance(value, list) and value
                and all(isinstance(row, dict) for row in value) else value
                for key, value in result.items()}
    rows = getattr(result, "rows", None)
    return result if rows is None else narrow(list(rows), name)
