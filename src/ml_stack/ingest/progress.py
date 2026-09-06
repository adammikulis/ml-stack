"""How far a run has got: the progress file beside the store."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ml_stack.ingest.reads import Read, _read_json, _write_json

__all__ = ["GIVE_UP", "Progress"]


GIVE_UP = 2             # failed attempts before --resume leaves a unit alone


class Progress:
    """The record of a run, beside the store: which units are done, and what each cost.

    A file rather than the store itself, because it is written after every section and a
    store is opened for writing once per source. ``--resume`` reads it; ``status`` prints it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.state: dict[str, Any] = {"started": time.strftime("%FT%T"), "sources": {}}
        held = _read_json(self.path)
        if isinstance(held, dict) and isinstance(held.get("sources"), dict):
            self.state = held

    @staticmethod
    def beside(out: str | Path) -> Path:
        """Where a store's progress file goes."""
        return Path(str(Path(out).expanduser()) + ".ingest.json")

    def source(self, slug: str, *, title: str = "", path: str = "", sections: int = 0
               ) -> dict[str, Any]:
        held = self.state["sources"].setdefault(slug, {"title": title, "path": path,
                                                       "sections": sections, "done": {}})
        for key, value in (("title", title), ("path", path), ("sections", sections)):
            if value:
                held[key] = value
        return held

    def done(self, slug: str, unit: str) -> bool:
        """Finished and kept, or given up on -- a unit that failed is written down, so
        `status` can say so, and is read again by `--resume` until it has failed `GIVE_UP`
        times; after that it is left, because one APBiology unit ran to the ceiling on four
        resumes in a row at twelve minutes each."""
        entry = (self.state["sources"].get(slug, {}).get("done") or {}).get(unit)
        if not isinstance(entry, dict):
            return False
        if not entry.get("error"):
            return True
        return int(entry.get("attempts") or 1) >= GIVE_UP

    def note(self, slug: str, read: Read) -> None:
        """Write one finished unit down, at once: a run killed mid-source resumes from here.

        An unreachable server is not the unit's failure and does not count as one of its
        attempts: the read is written down so `status` can say what happened, and the
        next `--resume` reads it again as if for the first time."""
        before = (self.source(slug)["done"].get(read.unit) or {})
        # an entry from before attempts were counted is one attempt; zero is a number
        attempts = int(before["attempts"]) if "attempts" in before else (1 if before else 0)
        if not read.error.startswith("ServerUnreachable"):
            attempts += 1
        self.source(slug)["done"][read.unit] = {
            "seconds": read.seconds, "concepts": read.concepts, "relations": read.relations,
            "figures": read.figures, "images": read.images, "error": read.error,
            "attempts": attempts, "at": time.strftime("%FT%T")}
        self.save()

    def folded(self, slug: str, *, units: int, nodes: int, edges: int,
               seconds: float = 0.0) -> None:
        """Write down that the source is in the store as of ``units`` units read, and what
        the fold cost -- `_fold_interval` reads the seconds back to space the next ones."""
        held = self.source(slug)
        held["folded_at"] = int(units)
        held["folded_nodes"] = int(nodes)
        held["folded_edges"] = int(edges)
        held["folded_seconds"] = round(float(seconds), 2)
        held["folded"] = time.strftime("%FT%T")
        self.save()

    def save(self) -> None:
        _write_json(self.path, self.state)

    def totals(self) -> dict[str, Any]:
        """Sources, sections done of how many, seconds spent, and sections a minute."""
        done = seconds = wanted = failed = given_up = 0
        for one in self.state["sources"].values():
            entries = (one.get("done") or {}).values()
            done += len(entries)
            wanted += int(one.get("sections") or 0)
            seconds += sum(float(e.get("seconds") or 0.0) for e in entries)
            failed += sum(1 for e in entries if e.get("error"))
            given_up += sum(1 for e in entries if e.get("error")
                            and int(e.get("attempts") or 1) >= GIVE_UP)
        return {"sources": len(self.state["sources"]), "sections": done, "of": wanted,
                "failed": failed, "given_up": given_up, "seconds": round(seconds, 1),
                "per_section": round(seconds / done, 1) if done else 0.0,
                "started": self.state.get("started", "")}


def _folded_at(out: str | Path) -> dict[str, int]:
    """``{source: units read when it was last folded}``, from the progress file."""
    if not out:
        return {}
    held = _read_json(Progress.beside(out))
    listed = held.get("sources") if isinstance(held, dict) else None
    if not isinstance(listed, dict):
        return {}
    return {slug: int((one or {}).get("folded_at") or 0) for slug, one in listed.items()}
