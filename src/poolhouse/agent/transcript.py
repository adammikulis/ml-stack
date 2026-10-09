"""An append-only record of what compaction removed, so none of it is lost.

Each line of the file is one JSON object: ``{"n", "ts", "kind", "messages" | "text"}``. A
reference such as ``agent-ab12.jsonl#7`` names a line and `Transcript.fetch` reads it back.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from poolhouse import home, jsonl

__all__ = ["Transcript"]


class Transcript:
    """The file one conversation's removed messages and elided results are written to."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (Path(path) if path
                     else home.state("compaction", f"{secrets.token_hex(4)}.jsonl"))
        self._next = len(jsonl.read(self.path)) + 1 if self.path.exists() else 1

    def _write(self, kind: str, **body: Any) -> str:
        n = self._next
        jsonl.append(self.path, [{"n": n, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "kind": kind, **body}])
        self._next += 1
        return f"{self.path.name}#{n}"

    def record(self, kind: str, messages: Sequence[Mapping[str, Any]]) -> str:
        """Write the ``messages`` removed for ``kind``; returns the reference."""
        return self._write(kind, messages=[dict(m) for m in messages])

    def spill(self, text: str) -> str:
        """Write a tool result that was cut short; returns the reference."""
        return self._write("elided", text=text)

    def fetch(self, reference: str) -> dict[str, Any]:
        """The record a reference names. Raises ``KeyError`` for one this file lacks."""
        name, _, n = reference.partition("#")
        for row in jsonl.rows(self.path):
            if name == self.path.name and str(row.get("n")) == n:
                return dict(row)
        raise KeyError(reference)
