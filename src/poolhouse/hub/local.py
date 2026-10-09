"""Model files already on this machine: where they live, what they are, and which
repository one came from."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from poolhouse import home, hub
from poolhouse.hub import places, scan
from poolhouse.hub.naming import _SHARD, DRAFT_MARK, WEIGHT_SUFFIXES


def hub_cache() -> Path:
    """The Hub's model cache: ``$HF_HUB_CACHE``, ``$HF_HOME/hub``, or ``~/.cache/huggingface/hub``."""
    return next(p.path for p in places.places() if p.label == "huggingface")


def default_roots(root: Path | str) -> list[Path]:
    """Where model files live: every folder `places` names, the store's own under ``root``
    first, in search order."""
    return [hub.hub_cache() if p.label == "huggingface" else p.path
            for p in places.places(state=home.expand(root))]


def _walk(place: places.Place) -> list[Path]:
    """The weight files under one folder, to the folder's depth, links left unfollowed."""
    out: list[Path] = []
    base = len(place.path.parts)
    for here, dirs, names in os.walk(place.path):
        deep = len(Path(here).parts) - base
        dirs[:] = sorted(d for d in dirs if d not in scan.SKIP_DIRS) if deep < place.depth else []
        out.extend(p for n in sorted(names)
                   if (p := Path(here) / n).suffix.lower() in WEIGHT_SUFFIXES and p.is_file())
    return out


def standard(roots: Sequence[Path | str] | None = None) -> list[places.Place]:
    """The folders to search as `Place` rows: ``roots`` when given, else `default_roots`;
    a folder `places` knows keeps its tool's label and layout."""
    known = {p.path: p for p in places.places(state=home.home())}
    named = list(roots) if roots is not None else hub.default_roots(home.home())
    return [known.get(Path(r)) or places.Place("extra", "auto", Path(r), True, 8)
            for r in named]


def weight_paths(roots: Sequence[Path] | None = None) -> list[Path]:
    """Every weight file under ``roots`` (the model roots by default), in root order.

    A Hub cache keeps a snapshot of symlinks into ``blobs/``; the link is returned, never
    the blob it points at, and a caller reading a size ``stat()``s through it. An Ollama
    folder answers with the blob each manifest names.
    """
    out: list[Path] = []
    for place in standard(roots):
        if not place.path.is_dir():
            continue
        if scan.detect(place.path) == "ollama":
            out.extend(e.path for e in scan.scan_ollama(place))
        else:
            out.extend(_walk(place))
    return out


def on_disk(*, alongside: bool = False) -> dict[str, int]:
    """Every model file already on this machine, by filename, with its real size.

    Sizes are resolved through symlinks: a Hub cache is symlinks into `blobs/`, so
    `ls -l` reports 79 bytes for a 46G shard. ``alongside`` includes the draft heads and
    vision projectors that travel with a model; they are left out otherwise, since neither
    is a model anybody serves on its own.
    """
    out: dict[str, int] = {}
    for path in weight_paths():
        if path.name in out or (not alongside and hub.aside(path.name)):
            continue
        try:
            out[path.name] = path.resolve().stat().st_size
        except OSError:
            continue
    return out


def shards_beside(path: Path) -> list[Path]:
    """``path`` and every other shard of its build in the same directory, in order -- just
    ``[path]`` for a model that is one file."""
    path = Path(path)
    if not _SHARD.search(path.name):
        return [path]
    stem = _SHARD.sub("", path.name)
    found = sorted(p for p in path.parent.iterdir()
                   if _SHARD.search(p.name) and _SHARD.sub("", p.name) == stem)
    return found or [path]


def _big_enough(path: Path, floor: int) -> bool:
    if floor <= 0:
        return True
    try:
        return path.stat().st_size >= floor
    except OSError:
        return False


def located(name: str | Path, *, roots: Sequence[Path] | None = None, loose: bool = False,
            min_size: int = 0) -> Path | None:
    """The file a model name stands for -- an existing path, or a filename found under
    ``roots`` (the model roots by default) by exact match, then by a sharded build's first
    shard, then with ``loose`` by the first name containing it, weights before the
    projectors and heads beside them -- or ``None``.

    A draft head is passed over unless ``name`` carries ``.draft``; so is a file under
    ``min_size`` bytes; an ``hf:`` reference is not a local file and is ``None``.
    """
    text = str(name).strip()
    if not text or text.startswith("hf:"):
        return None
    where = home.expand(text)
    if where.is_file():
        return where
    if "/" in text or os.sep in text:
        return None
    wanted = Path(text).name
    drafts = DRAFT_MARK in Path(wanted).suffixes
    every = [p for p in weight_paths(roots)
             if (drafts or DRAFT_MARK not in p.suffixes) and _big_enough(p, min_size)]
    base = wanted[: -len(".gguf")] if wanted.lower().endswith(".gguf") else wanted
    # a Hub match is the snapshot symlink, never the blob: llama.cpp reads a sharded
    # model's other shards off that name and refuses a blob hash
    shard = re.compile(rf"^{re.escape(base)}-00001-of-\d+\.gguf$", re.IGNORECASE)
    for match in (lambda p: p.name == wanted, lambda p: bool(shard.match(p.name))):
        found = next((p for p in every if match(p)), None)
        if found is not None:
            return found
    needle = wanted.lower()
    # weights before the projectors and heads that travel with them
    ranked = sorted(every, key=lambda p: hub.aside(p.name))
    named = next((p for p in ranked if needle in p.name.lower()), None) if loose else None
    return named or _installed_as(wanted, roots, loose, min_size)


def _installed_as(wanted: str, roots: Sequence[Path] | None, loose: bool,
                  min_size: int) -> Path | None:
    """The file an installed model's own name stands for (``llama3:latest`` in Ollama)."""
    pool = [m for m in hub.discover(standard(roots), formats=("gguf",))
            if m.is_complete and _big_enough(m.path, min_size)]
    exact = hub.installed_find(wanted, pool)
    narrowed = exact if loose else [m for m in exact if wanted.lower() in (
        m.id.lower(), m.name.lower(), f"ollama:{wanted.lower()}")]
    return narrowed[0].path if narrowed else None


def repo_of(model: str | Path) -> str:
    """The Hub repository ``model`` came from, as ``owner/name``, or ''.

    An `hf:` reference names it outright. A path inside the Hub cache names it too --
    the cache keeps one directory per repository, `models--owner--name/snapshots/<rev>/`,
    so a file downloaded through `poolhouse-models fetch` or a lease still knows where it
    came from and can be asked what shipped with it. A bare filename is looked up with
    `located`. A path anywhere else came from nowhere the Hub can say.
    """
    text = str(model)
    if text.startswith("hf:"):
        return "/".join(text[3:].split("/")[:2])
    where = Path(text).expanduser()
    if "/" not in text and not where.is_file():
        found = hub.located(text)
        if found is None:
            return ""
        where = found
    for part in where.resolve().parts:
        if part.startswith("models--") and "--" in part[len("models--"):]:
            owner, name = part[len("models--"):].split("--", 1)
            return f"{owner}/{name}"
    return ""
