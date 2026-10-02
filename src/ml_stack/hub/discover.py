"""Every model installed on this machine, whichever tool put it there.

``discover`` reads the folders ``places`` names, collapses a model found twice into one row
with the copy to use first, and answers with `ModelInfo`. Headers are read from the front of
each file and kept in a cache keyed by path, size and modification time, so a second call
touches only what changed.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ml_stack import files, home, hub
from ml_stack.hub import header
from ml_stack.hub.naming import QUANT, _precision, base_words, pretty_name
from ml_stack.hub.places import Place
from ml_stack.hub.scan import Entry, scan

FORMATS = ("gguf", "safetensors", "mlx")

RANK = {"ml-stack": 0, "extra": 1, "huggingface": 2, "llama.cpp": 3, "lmstudio": 4,
        "jan": 5, "gpt4all": 6, "ollama": 7, "manual": 8, "modelscope": 9, "kagglehub": 10,
        "downloads": 11, "volume": 12}
"""Which copy of a model is used when it is installed in more than one place."""

_OF = re.compile(r"-(\d{5})-of-(\d{5})(?=\.gguf$)", re.IGNORECASE)
_VERSION = 1


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One installed model."""

    id: str
    name: str
    path: Path
    format: str
    size_bytes: int
    source: str
    quantization: str = ""
    parameters: int = 0
    architecture: str = ""
    context_length: int = 0
    mmproj: Path | None = None
    repo: str = ""
    mtime: float = 0.0
    is_complete: bool = True
    shards: int = 1
    verified: bool = True
    copies: tuple[Path, ...] = ()
    """Paths of the other copies of this model, which were collapsed into this row."""

    def as_dict(self) -> dict[str, object]:
        """The row as plain JSON types."""
        row = asdict(self)
        row["path"] = str(self.path)
        row["mmproj"] = str(self.mmproj) if self.mmproj else None
        row["copies"] = [str(c) for c in self.copies]
        return row

    @property
    def filename(self) -> str:
        return self.path.name


class _Headers:
    """Header summaries kept between runs, one JSON file under the cache root."""

    def __init__(self, *, use: bool) -> None:
        self.path = home.cache("models", "headers.json")
        self.rows: dict[str, dict] = {}
        self.dirty = False
        if use:
            record = files.read_json(self.path, {})
            if isinstance(record, dict) and files.version_of(record) == _VERSION:
                self.rows = dict(record.get("rows", {}))

    def of(self, path: Path, size: int, mtime: float) -> header.Header | None:
        try:
            key = f"{path.resolve()}|{size}|{int(mtime * 1e6)}"
        except OSError:
            return None
        if key in self.rows:
            row = self.rows[key]
            return header.Header.from_dict(row) if row else None
        got = header.read(path)
        self.rows[key] = got.as_dict() if got else {}
        self.dirty = True
        return got

    def save(self) -> None:
        if not self.dirty:
            return
        with contextlib.suppress(OSError):
            files.write_json(self.path, files.versioned({"rows": self.rows}, _VERSION))


def _as_places(roots: Iterable[Path | str | Place] | None) -> list[Place]:
    if roots is None:
        return hub.standard()
    return [r if isinstance(r, Place) else Place("extra", "auto", home.expand(r), True, 8)
            for r in roots]


def _stem(name: str) -> str:
    return _OF.sub("", name)


def _members(entries: list[Entry]) -> list[tuple[Entry, list[Entry]]]:
    """Entries with the other shards of a sharded build folded under its first shard."""
    groups: dict[tuple[Path, str], list[Entry]] = {}
    out: list[tuple[Entry, list[Entry]]] = []
    for one in entries:
        if one.format == "gguf" and _OF.search(one.name):
            groups.setdefault((one.path.parent, _stem(one.name)), []).append(one)
        else:
            out.append((one, [one]))
    for parts in groups.values():
        parts.sort(key=lambda e: e.name)
        out.append((parts[0], parts))
    return out


def _mmproj(one: Entry, side: dict[Path, list[Entry]], alone: bool) -> Path | None:
    if one.mmproj:
        return one.mmproj
    near = side.get(one.path.parent, [])
    words = set(base_words(one.name))
    fits = [m for m in near if alone or (set(base_words(m.name)) - {"mmproj"}) <= words]
    return min(fits, key=lambda m: _precision(m.name)).path if fits else None


def _hf_file(one: Entry) -> str:
    """The file's name relative to its snapshot, so an id carries the build folder."""
    parts = one.path.parts
    if one.revision and one.revision in parts:
        return "/".join(parts[parts.index(one.revision) + 1:])
    return one.sidecar_name or one.name


def _identity(one: Entry, name: str) -> str:
    if one.display:
        return f"ollama:{one.display}"
    if one.repo and one.format == "gguf":
        return f"hf:{one.repo}/{_hf_file(one) if one.revision else name}"
    if one.repo:
        return f"hf:{one.repo}"
    return f"file:{one.name}"


def _quant(name: str, head: header.Header | None) -> str:
    found = QUANT.search(name)
    if found:
        return found.group(1).upper()
    return head.quantization if head else ""


def _info(one: Entry, parts: list[Entry], head: header.Header | None,
          side: dict[Path, list[Entry]], alone: bool) -> ModelInfo:
    size = sum(p.size for p in parts)
    total = int(_OF.search(one.name).group(2)) if _OF.search(one.name) else 1
    shown = one.display or (head.name if head and head.name and one.format == "gguf"
                            and not one.repo else "") or pretty_name(_stem(one.name))
    if one.format != "gguf":
        shown = one.repo.split("/")[-1] if one.repo else one.name
    return ModelInfo(
        id=_identity(one, _stem(one.name) if total == 1 else one.name),
        name=shown, path=one.path, format=one.format, size_bytes=size, source=one.place,
        quantization=_quant(one.name, head) if one.format == "gguf" else "",
        parameters=head.parameters if head else 0,
        architecture=head.architecture if head else "",
        context_length=head.context_length if head else 0,
        mmproj=_mmproj(one, side, alone) if one.format == "gguf" else None,
        repo=one.repo or "", mtime=max(p.mtime for p in parts),
        is_complete=all(p.complete for p in parts) and len(parts) == total
        and (head is not None or one.format != "gguf"),
        shards=total, verified=one.verified)


def _best(group: list[ModelInfo]) -> ModelInfo:
    group.sort(key=lambda m: (not m.is_complete, RANK.get(m.source, 99), -m.mtime))
    first = group[0]
    others = tuple(m.path for m in group[1:] if m.path != first.path)
    return replace(first, copies=others)


def _collapse(infos: list[tuple[ModelInfo, header.Header | None]]) -> list[ModelInfo]:
    groups: dict[object, list[ModelInfo]] = {}
    for info, head in infos:
        if info.format != "gguf":
            key: object = (info.format, info.repo or info.filename)
        else:
            key = (info.size_bytes, head.architecture, head.name) if head else (
                info.size_bytes, info.filename.lower())
        groups.setdefault(key, []).append(info)
    return [_best(g) for g in groups.values()]


def discover(roots: Iterable[Path | str | Place] | None = None,
             formats: Iterable[str] = FORMATS, include: Iterable[str] | None = None, *,
             companions: bool = False, refresh: bool = False) -> list[ModelInfo]:
    """Every installed model in ``formats``, one row each, newest first.

    ``roots`` replaces the standard folders (see `places`) with the folders given, each
    searched as whatever layout it holds. ``include`` keeps only models from those sources
    (``huggingface``, ``ollama``, ``lmstudio``, ...). ``companions`` adds vision projectors
    and draft heads as rows of their own. ``refresh`` ignores the header cache.
    """
    wanted = set(formats)
    kept = set(include) if include is not None else None
    chosen = _as_places(roots)
    allowed = [p.path.resolve() for p in chosen if p.path.exists()]
    headers = _Headers(use=not refresh)
    entries: list[Entry] = []
    for place in chosen:
        if kept is None or place.label in kept:
            entries += list(scan(place, allowed))
    side: dict[Path, list[Entry]] = {}
    for one in entries:
        if one.aux and one.name.lower().startswith("mmproj"):
            side.setdefault(one.path.parent, []).append(one)
    primaries = [e for e in entries if companions or not e.aux]
    solo: dict[Path, int] = {}
    for one in primaries:
        if not one.aux:
            solo[one.path.parent] = solo.get(one.path.parent, 0) + 1
    found: list[tuple[ModelInfo, header.Header | None]] = []
    for one, parts in _members(primaries):
        if one.format not in wanted:
            continue
        head = headers.of(one.path, one.size, one.mtime) if one.format == "gguf" else None
        found.append((_info(one, parts, head, side, solo.get(one.path.parent, 0) == 1), head))
    headers.save()
    return sorted(_collapse(found), key=lambda m: (-m.mtime, m.id))


_SPLIT = re.compile(r"[\s/_:]+")


def find(query: str, installed: list[ModelInfo] | None = None) -> list[ModelInfo]:
    """The installed models a name stands for, best match first.

    An exact id, name or file name wins, then a name with ``:latest`` implied, then every
    model whose id contains all the words of ``query``.
    """
    pool = installed if installed is not None else discover()
    text = query.strip().lower()
    if not text:
        return []
    exact = [m for m in pool if text in (m.id.lower(), m.name.lower(), m.filename.lower(),
                                         m.filename.lower().removesuffix(".gguf"))]
    if exact:
        return exact
    tagged = [m for m in pool if m.id.lower() == f"ollama:{text}:latest"]
    if tagged:
        return tagged
    words = [w for w in _SPLIT.split(text) if w]
    return [m for m in pool if all(w in f"{m.id} {m.name} {m.filename}".lower() for w in words)]


def installed_for(ref: str, installed: list[ModelInfo] | None = None) -> ModelInfo | None:
    """The installed copy of an ``hf:owner/repo/file.gguf`` reference, or ``None``.

    Needs the repository and the file name to match, so a different quantisation of the
    same repository is not mistaken for it.
    """
    if not ref.startswith("hf:"):
        return None
    parts = [p for p in ref[3:].split("/") if p]
    if len(parts) < 3:
        return None
    repo, name = "/".join(parts[:2]), parts[-1].lower()
    pool = installed if installed is not None else discover(formats=("gguf",),
                                                            companions=True)
    for m in pool:
        same_file = m.filename.lower() == name or _stem(m.filename).lower() == _stem(name)
        if m.is_complete and m.repo.lower() == repo.lower() and same_file:
            return m
    return None
