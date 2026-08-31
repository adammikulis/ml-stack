"""What has already been read, so a second run is cheap.

A watermark per source is most of it: everything newer than the last row read is new. The rest
is knowing when something *old* changed — a thread that grew, a page that was edited — which a
watermark cannot tell you, so a small mark is kept beside it and compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Seen:
    """A record of what each source had, last time anyone looked."""

    path: Path
    marks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Seen:
        where = Path(path).expanduser()
        try:
            raw = json.loads(where.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        marks = {k: v for k, v in raw.items() if isinstance(v, dict)}
        # an older file kept a bare watermark per source; read it as one
        marks.update({k: {"mark": v} for k, v in raw.items() if not isinstance(v, dict)})
        return cls(path=where, marks=marks)

    def save(self) -> None:
        self.path.expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.path.expanduser().write_text(json.dumps(self.marks, indent=1, sort_keys=True),
                                          encoding="utf-8")

    def mark(self, source: str) -> str:
        return str(self.marks.get(source, {}).get("mark") or "")

    def fresh(self, source: str, rows: list[dict[str, Any]], *, key: str = "key"
              ) -> list[dict[str, Any]]:
        """The rows newer than the watermark. Keys sort, which is what makes this work."""
        mark = self.mark(source)
        return [r for r in rows if str(r.get(key) or "") > mark] if mark else list(rows)

    def changed(self, source: str, counts: dict[str, Any]) -> list[str]:
        """Which rows have a different mark than last time — something under them grew."""
        before = self.marks.get(source, {}).get("counts") or {}
        return sorted(k for k, v in counts.items() if str(before.get(k, "")) != str(v))

    def record(self, source: str, rows: list[dict[str, Any]], *, key: str = "key",
               counts: dict[str, Any] | None = None) -> None:
        """Remember how far this source was read, and what was under each row."""
        keys = [str(r.get(key) or "") for r in rows if r.get(key)]
        entry = dict(self.marks.get(source) or {})
        if keys:
            entry["mark"] = max([max(keys), self.mark(source)])
        if counts is not None:
            entry["counts"] = dict(counts)
        self.marks[source] = entry
