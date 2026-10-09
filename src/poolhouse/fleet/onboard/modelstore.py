"""Serve the machine's own model store to paired devices (``fleet share --models``).

The files are what `hub.discover` finds (GGUF and safetensors), offered by file name with their
repository, size and sha256. Only the model roots: a file is listed, and again at every request,
only if its resolved path lies inside a root (`transfer.confined`). Nothing sentinel holds. A model
is ``owner`` unless its terms say otherwise: only the owner's own devices get it, and only once the
owner accepted its licence on this machine. The manifest is signed with the owner's key and its
serial never goes down (`next_serial`). Digests are kept by path, size and mtime (`Digests`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poolhouse import hub
from poolhouse.files import read_json, write_json
from poolhouse.hub.places import Place
from poolhouse.safenames import Unsafe, safe_filename

from .manifest import DEFAULT_CHUNK, MOST_ENTRIES, Entry, chunk_digests
from .sharing import OWNER
from .transfer import confined

__all__ = ["Candidate", "Digests", "Store", "Terms", "build", "candidates", "next_serial"]

logger = logging.getLogger(__name__)

SUFFIXES = (".gguf", ".safetensors")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A model file found in a root."""

    name: str
    path: Path
    size: int
    repo: str


@dataclass(frozen=True, slots=True)
class Terms:
    """What the owner says about files: by file name or by repository."""

    sharing: dict[str, str] = field(default_factory=dict)
    licence: dict[str, str] = field(default_factory=dict)
    """``name -> "id,url"``."""
    source: dict[str, str] = field(default_factory=dict)

    def pick(self, table: dict[str, str], c: Candidate) -> str:
        return table.get(c.name) or (table.get(c.repo, "") if c.repo else "")


@dataclass(slots=True)
class Store:
    """The result: entries for a manifest, where each file is, and why others were left out."""

    entries: list[Entry] = field(default_factory=list)
    paths: dict[str, Path] = field(default_factory=dict)
    roots: tuple[Path, ...] = ()
    skipped: list[str] = field(default_factory=list)


class Digests:
    """Whole-file and chunk digests kept between runs, by path, size and modification time."""

    def __init__(self, path: Path | None) -> None:
        self.path, self.dirty = path, False
        doc = read_json(path, {}) if path else {}
        self.rows: dict[str, Any] = dict(doc.get("rows", {})) if isinstance(doc, dict) else {}

    def of(self, file: Path, size: int, chunk_size: int) -> tuple[str, tuple[str, ...]]:
        stat = file.stat()
        key = f"{file.resolve()}|{size}|{stat.st_mtime_ns}|{chunk_size}"
        row = self.rows.get(key)
        if isinstance(row, dict) and isinstance(row.get("chunks"), list):
            return str(row["sha256"]), tuple(row["chunks"])
        whole, parts = chunk_digests(file, chunk_size)
        self.rows = {key: {"sha256": whole, "chunks": list(parts)}} | \
            {k: v for k, v in self.rows.items() if k.split("|")[0] != str(file.resolve())}
        self.dirty = True
        return whole, parts

    def save(self) -> None:
        if self.dirty and self.path:
            write_json(self.path, {"schema_version": 1, "rows": self.rows}, indent=None)


def roots_of(places: Iterable[Place]) -> tuple[Path, ...]:
    return tuple(p.path for p in places if p.path.is_dir())


def candidates(roots: Iterable[Path | str] | None = None) -> tuple[list[Candidate], tuple[Path, ...]]:
    """``(files, roots)``: the model files of the folders ``roots`` (the usual ones when None)
    and the folders searched. GGUF and safetensors only; an Ollama blob has no name to offer."""
    places = hub.standard(None if roots is None else [Path(r) for r in roots])
    searched = roots_of(places)
    found: list[Candidate] = []
    seen: set[Path] = set()
    for m in hub.discover([p for p in places if p.path.is_dir()], formats=("gguf", "safetensors")):
        if m.source == "ollama":
            continue
        for part in (hub.shards_beside(m.path) if m.format == "gguf" else [m.path]):
            if part.suffix.lower() not in SUFFIXES or part in seen:
                continue
            seen.add(part)
            try:
                found.append(Candidate(safe_filename(part.name), part, part.stat().st_size, m.repo))
            except (OSError, Unsafe):
                continue
    return found, searched


def build(found: Iterable[Candidate], roots: tuple[Path, ...], *, digests: Digests,  # noqa: PLR0913 - one call site, all keywords
          terms: Terms | None = None, veto: Callable[[Entry], str] = lambda _e: "",
          chunk_size: int = DEFAULT_CHUNK) -> Store:
    """Entries for ``found``: only files inside ``roots``, one per file name, none that ``veto``
    holds back. The default level is ``owner`` with the repository as the licence."""
    terms = terms or Terms()
    store = Store(roots=roots)
    for c in found:
        try:
            confined(c.path, roots)
        except Unsafe as exc:
            store.skipped.append(f"{c.name}: {exc}")
            continue
        if c.name in store.paths:
            store.skipped.append(f"{c.name}: another file of that name is already listed")
            continue
        if len(store.entries) >= MOST_ENTRIES:
            store.skipped.append(f"{c.name}: a manifest lists at most {MOST_ENTRIES} files")
            continue
        level = terms.pick(terms.sharing, c) or OWNER
        stated = terms.pick(terms.licence, c)
        lid, _, url = stated.partition(",")
        if not lid and level == OWNER:
            lid = c.repo or c.name
            url = f"https://huggingface.co/{c.repo}" if c.repo.count("/") == 1 else ""
        try:
            whole, parts = digests.of(c.path, c.size, chunk_size)
        except OSError as exc:
            store.skipped.append(f"{c.name}: cannot be read: {exc}")
            continue
        entry = Entry(c.name, c.size, whole, chunk_size, parts, kind="model", sharing=level,
                      licence=lid[:200], licence_url=url[:500],
                      source=terms.pick(terms.source, c)[:500], repo=c.repo)
        if reason := veto(entry):
            store.skipped.append(f"{c.name}: {reason}")
            continue
        store.entries.append(entry)
        store.paths[c.name] = c.path
    digests.save()
    return store


def next_serial(path: Path, now: Callable[[], float] = time.time) -> int:
    """A manifest serial above every one this machine issued before (the time, or one more than
    the last when the clock went back), written down before it is used."""
    doc = read_json(path, {})
    last = int(doc.get("serial", 0)) if isinstance(doc, dict) else 0
    serial = max(last + 1, int(now()))
    write_json(path, {"schema_version": 1, "serial": serial})
    return serial
