"""What one source's reads are kept as: a `Read` per unit, the units their provenance
names, and the files beside the store they are written into."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Read", "reads_path", "tokens_of", "unit_of", "units_of"]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-") or "untitled"


@dataclass
class Read:
    """One unit, extracted once, and everything it cost."""

    unit: str
    source: str
    chapter: str
    section: str
    title: str
    pages: list[int] = field(default_factory=list)
    seconds: float = 0.0
    concepts: int = 0
    relations: int = 0
    figures: int = 0
    images: int = 0
    timed_out: bool = False
    retried: bool = False    # read a second time, the server having reset inside the first
    error: str = ""
    raw: str = ""            # what the model wrote when it failed, whole, for reading later
    run: str = ""            # the run node that read it -- model, build, head, hashes, when
    extracted: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _Unit:
    """A unit as its reads file remembers it: the provenance `build` needs, and no PDF."""

    id: str
    source: str
    chapter: str
    section: str
    section_title: str
    first_page: int
    last_page: int

    @property
    def where(self) -> dict[str, Any]:
        return {"source": self.source, "chapter": self.chapter, "section": self.section,
                "page": self.first_page, "pages": [self.first_page, self.last_page],
                "unit": self.id}


def unit_of(read: Mapping[str, Any]) -> _Unit:
    """One row of a reads file as something `build` and `fold_source` can take.

    The unit id is the one on the row rather than one recomputed, so a section split into
    parts keeps the id its provenance already names.
    """
    pages = list(read.get("pages") or ())
    return _Unit(id=str(read.get("unit") or ""), source=str(read.get("source") or ""),
                 chapter=str(read.get("chapter") or ""), section=str(read.get("section") or ""),
                 section_title=str(read.get("title") or ""),
                 first_page=int(pages[0]) if pages else 0,
                 last_page=int(pages[-1]) if pages else 0)


def units_of(reads: Iterable[Mapping[str, Any]]) -> dict[str, _Unit]:
    """``{unit id: unit}`` for one source's reads -- what `fold_source` wants, from the
    reads files alone."""
    out: dict[str, _Unit] = {}
    for read in reads:
        unit = unit_of(read)
        if unit.id:
            out[unit.id] = unit
    return out


def reads_path(out: str | Path, slug: str) -> Path:
    """Where one source's extractions are kept, beside the store."""
    return Path(str(Path(out).expanduser()) + f".{slug}.reads.json")


def tokens_of(reads: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    """``(read, written)`` tokens over some reads, from the `Call` each one kept.

    A row from before the calls were kept, or one whose extraction never reached the
    server, counts nothing rather than being left out of the total.
    """
    prompt = completion = 0
    for row in reads:
        for call in row.get("calls") or ():
            if isinstance(call, Mapping):
                prompt += int(call.get("prompt_tokens") or 0)
                completion += int(call.get("completion_tokens") or 0)
    return prompt, completion


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value: Any) -> None:
    """JSON into ``path`` through a temporary file and a rename: a kill mid-write leaves
    the file that was there, never half of the one being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        temp.write_text(json.dumps(value, indent=1), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _keep_reads(out: str | Path, slug: str, reads: Sequence[Mapping[str, Any]]) -> None:
    """Every unit's extraction, beside the store, keyed by unit -- what ``--resume`` folds."""
    path = reads_path(out, slug)
    held = _read_json(path)
    kept = dict(held) if isinstance(held, dict) else {}
    for read in reads:
        kept[str(read.get("unit") or "")] = dict(read)
    _write_json(path, kept)
