"""The vocabulary a store is read with: the core verbs and kinds, and the ones a reading
named itself, with how often each has been used."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.ingest.extract import CORE_KINDS, VERBS

__all__ = ["DOC", "MOST", "Vocabulary"]


DOC = "ingest:vocabulary"
"""The store document a vocabulary is kept in."""

MOST = 40
"""How many coined verbs, and how many coined kinds, a unit's prompt is shown. The core
lists are always shown in full; the coined ones are shown most-used first."""


def _entries(core: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    return {name: {"core": True, "uses": 0, "first": "", "gloss": gloss}
            for name, gloss in core.items()}


@dataclass
class Vocabulary:
    """What has been used to read a store so far: ``{name: {core, uses, first}}`` for the
    relation verbs and for the concept kinds."""

    verbs: dict[str, dict[str, Any]] = field(default_factory=lambda: _entries(VERBS))
    kinds: dict[str, dict[str, Any]] = field(default_factory=lambda: _entries(CORE_KINDS))

    @classmethod
    def read(cls, out: str | Path | None) -> Vocabulary:
        """The vocabulary a store remembers, the core lists always in it.

        A store with no vocabulary document -- one read before the vocabulary was kept, or
        a run that stopped before its first fold -- is counted from the extractions beside
        it instead.
        """
        document = _document_in(out)
        if not document:
            return cls.from_reads(out)
        held = cls()
        for section, into in (("verbs", held.verbs), ("kinds", held.kinds)):
            for name, entry in (document.get(section) or {}).items():
                if not isinstance(entry, Mapping):
                    continue
                kept = into.setdefault(str(name), {"core": False, "uses": 0, "first": "",
                                                   "gloss": ""})
                kept["uses"] = int(entry.get("uses") or 0)
                kept["first"] = str(entry.get("first") or "")
        return held

    @classmethod
    def from_reads(cls, out: str | Path | None) -> Vocabulary:
        """Counted from the extractions in the reads files beside a store."""
        held = cls()
        if not out:
            return held
        from ml_stack.ingest.sources import Sources

        held_sources = Sources(out)
        for source in held_sources.sources():
            for read in held_sources.reads(source.slug):
                extracted = read.get("extracted")
                if isinstance(extracted, Mapping):
                    held.note(extracted, str(read.get("unit") or ""))
        return held

    def note(self, extraction: Mapping[str, Any], unit: str = "") -> list[str]:
        """Count one extraction's verbs and kinds; return the names it coined that the
        vocabulary had never seen."""
        new: list[str] = []
        for said, field_name, into in (
                (extraction.get("relations") or (), "rel", self.verbs),
                (extraction.get("concepts") or (), "kind", self.kinds)):
            for one in said:
                if not isinstance(one, Mapping):
                    continue
                name = " ".join(str(one.get(field_name) or "").split())
                if not name:
                    continue
                entry = into.get(name)
                if entry is None:
                    entry = into[name] = {"core": False, "uses": 0, "first": str(unit),
                                          "gloss": ""}
                    new.append(name)
                entry["uses"] = int(entry.get("uses") or 0) + 1
        return new

    def coined(self) -> tuple[list[str], list[str]]:
        """The verbs and the kinds outside the core lists, most used first."""
        def by_use(entries: Mapping[str, Mapping[str, Any]]) -> list[str]:
            return sorted((n for n, e in entries.items() if not e.get("core")),
                          key=lambda n: (-int(entries[n].get("uses") or 0), n))

        return by_use(self.verbs), by_use(self.kinds)

    def seen(self, most: int = MOST) -> tuple[list[str], list[str]]:
        """The coined verbs and kinds a unit's prompt is shown: most used first, ``most``
        of each."""
        verbs, kinds = self.coined()
        return verbs[:most], kinds[:most]

    def document(self) -> dict[str, Any]:
        """The vocabulary as the store keeps it."""
        return {"verbs": {n: dict(e) for n, e in self.verbs.items()},
                "kinds": {n: dict(e) for n, e in self.kinds.items()}}

    def write(self, out: str | Path) -> None:
        """Put the vocabulary in the store under `DOC`."""
        from ml_stack.graph.store import GraphStore

        with GraphStore(out) as store:
            store.put_doc(DOC, self.document())

    def lines(self, most: int = 10) -> list[str]:
        """The vocabulary as a person reads it: how much of the core is used, and what was
        named beside it."""
        verbs, kinds = self.coined()
        used = sum(1 for e in self.verbs.values() if e.get("core") and e.get("uses"))
        out = [f"vocabulary: {used} of {len(VERBS)} core verbs used, "
               f"{len(verbs)} coined; {len(kinds)} kind(s) coined"]
        for label, names, entries in (("verbs", verbs, self.verbs),
                                      ("kinds", kinds, self.kinds)):
            if names:
                out.append(f"  {label} named while reading: " + ", ".join(
                    f"{n} ({entries[n].get('uses') or 0})" for n in names[:most])
                    + (f", and {len(names) - most} more" if len(names) > most else ""))
        return out


def _document_in(out: str | Path | None) -> dict[str, Any]:
    if not out or not Path(out).expanduser().exists():
        return {}
    try:
        from ml_stack.graph.store import GraphStore

        with GraphStore(out, read_only=True) as store:
            held = store.get_doc(DOC)
    except Exception:  # noqa: BLE001 - a vocabulary is guidance; a run reads on without one
        return {}
    return dict(held) if isinstance(held, Mapping) else {}
