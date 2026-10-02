"""Labelled decisions: the JSONL format benchmarks, training sets and evaluations share.

One line per case::

    {"id": "g-001", "question": "...", "state": "...", "options": {"safe": "...", "unsafe": "..."},
     "label": "safe", "group": "optional split key", "tags": ["optional"]}

``options`` is a list of names or an object of name to description.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import jsonl
from ml_stack.decide.base import Request, State
from ml_stack.decide.types import Option, options_of


@dataclass(frozen=True, slots=True)
class Case:
    """One question with its answer."""

    question: str
    state: State
    options: tuple[Option, ...]
    label: str
    id: str = ""
    group: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.label not in {o.name for o in self.options}:
            raise ValueError(f"case {self.id or self.question[:40]!r}: label {self.label!r} "
                             f"is not one of {[o.name for o in self.options]}")

    @property
    def label_index(self) -> int:
        """Position of the label among the options."""
        return [o.name for o in self.options].index(self.label)

    def request(self, abstain_below: float | None = None) -> Request:
        """This case as a `Request`."""
        return Request(self.question, self.state, self.options, abstain_below=abstain_below)

    def public(self) -> dict[str, Any]:
        """A JSON-ready dict in the file format."""
        out: dict[str, Any] = {
            "id": self.id, "question": self.question, "state": self.state,
            "options": {o.name: o.description for o in self.options}, "label": self.label}
        if self.group:
            out["group"] = self.group
        if self.tags:
            out["tags"] = list(self.tags)
        return out


def case_from(row: Mapping[str, Any]) -> Case:
    """A `Case` from one parsed line; a missing or mistyped field raises ValueError."""
    for key in ("question", "state", "options", "label"):
        if key not in row:
            raise ValueError(f"case has no {key!r}: {str(dict(row))[:120]}")
    return Case(question=str(row["question"]), state=row["state"],
                options=options_of(row["options"]), label=str(row["label"]),
                id=str(row.get("id", "")), group=str(row.get("group", "")),
                tags=tuple(str(t) for t in row.get("tags", ())))


def read_cases(path: Path | str) -> list[Case]:
    """Every case in a JSONL file."""
    return [case_from(row) for row in jsonl.rows(path)]


def write_cases(path: Path | str, cases: Iterable[Case]) -> int:
    """Write cases as JSONL; returns how many."""
    return jsonl.write(path, (c.public() for c in cases))


def fingerprint(cases: Sequence[Case]) -> str:
    """A SHA-256 over the cases in order: the data hash a model card records."""
    digest = hashlib.sha256()
    for c in cases:
        digest.update(json.dumps(c.public(), sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()
