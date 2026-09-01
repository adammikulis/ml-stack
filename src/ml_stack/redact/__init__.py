"""Replace known names in text with stable placeholders."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["Redactor", "names_in", "tag"]


def tag(name: str, prefix: str = "person") -> str:
    """A stable stand-in for one name. The same name always gives the same tag."""
    return f"{prefix}#" + hashlib.sha256(name.strip().casefold().encode()).hexdigest()[:4]


class Redactor:
    """Swaps a given set of names out of any text, longest name first, case-insensitively."""

    def __init__(self, names: Iterable[str], *, prefix: str = "person") -> None:
        self.prefix = prefix
        ordered = sorted({n for n in names if n and n.strip()}, key=len, reverse=True)
        self.pattern = re.compile(
            "|".join(rf"(?<![\w.]){re.escape(n)}(?![\w])" for n in ordered),
            re.I) if ordered else None

    def __call__(self, text: Any) -> str:
        body = str(text)
        if self.pattern is None:
            return body
        return self.pattern.sub(lambda m: tag(m.group(0), self.prefix), body)


def names_in(graph: Path | None = None, messages: Path | None = None, *,
             kind: str = "person", field: str = "sender", min_length: int = 3) -> set[str]:
    """Every name a graph and its message log hold, read fresh, for a :class:`Redactor`.

    The graph is a JSON mapping: the ``label`` of each node of ``kind``, and ``field`` of each
    value in its ``messages`` mapping. The log is JSON lines, ``field`` of each row. Either
    may be absent, missing or not JSON, and contributes nothing then. Names shorter than
    ``min_length`` are left out, since a two-letter name is also a word in most sentences.
    """
    names: set[str] = set()
    if graph is not None:
        try:
            g = json.loads(Path(graph).read_text(encoding="utf-8"))
            names |= {n["label"] for n in g.get("nodes", []) if n.get("kind") == kind}
            names |= {m[field] for m in (g.get("messages") or {}).values() if m.get(field)}
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            pass
    if messages is not None:
        try:
            lines = Path(messages).read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get(field):
                names.add(row[field])
    return {n for n in names if isinstance(n, str) and len(n) >= min_length}
