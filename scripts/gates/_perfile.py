"""Results of checks that read one file at a time, kept by the bytes of the file they read."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path
from typing import Any

from . import Finding, _store
from ._util import python_files, rel

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


@cache
def where(path: Path, root: Path) -> str:
    """The path relative to root, worked out once per pair."""
    return rel(path, root)


def cached_here(root: Path) -> bool:
    """True when root is the tree this package sits in, the only one whose results are kept."""
    return root.resolve() == REPO


def kept_name(owner: Callable) -> str:
    """The cache entry for a function: its module and its name."""
    return f"perfile-{owner.__module__.rsplit('.', 1)[-1].strip('_')}-{owner.__name__.strip('_')}"


def code_stamp(owner: Callable, salt: str) -> str:
    """Changes when the interpreter, the function's module or the shared parsing code does."""
    module = sys.modules[owner.__module__]
    files = [Path(module.__file__ or ""), HERE / "_util.py", HERE / "_perfile.py", HERE / "_store.py"]
    return _store.digest(f"{_store.stamp(*files)}\0{salt}".encode())


def each(root: Path, paths: Iterable[Path], compute: Callable[[Path], Any],
         owner: Callable | None = None, salt: str = "") -> list[Any]:
    """``compute(path)`` for each path, in order, reusing what was kept for the same bytes.

    Results are kept only for the repository this package sits in, as JSON, by the path
    relative to root and the hash of the file's bytes; ``salt`` is whatever else they depend on.
    """
    paths = list(paths)
    if not cached_here(root):
        return [compute(path) for path in paths]
    owner = owner or compute
    name, stamp = kept_name(owner), code_stamp(owner, salt)
    stored = {} if _store.forced() else _store.load(name)
    old = stored.get("files", {}) if stored.get("stamp") == stamp else {}
    new: dict[str, list] = {}
    out: list[Any] = []
    for path in paths:
        at, seen = where(path, root), _store.file_digest(path)
        have = old.get(at)
        value = have[1] if have and have[0] == seen else compute(path)
        new[at] = [seen, value]
        out.append(value)
    if new != old or stored.get("stamp") != stamp:
        _store.save(name, {"stamp": stamp, "files": new})
    return out


def each_file(root: Path, paths: Iterable[Path],
              scan: Callable[[Path, str], list[Finding]]) -> list[Finding]:
    """Every finding of ``scan(path, where)`` over the paths, as one list in path order."""
    def rows(path: Path) -> list[list]:
        return [[f.path, f.line, f.detail] for f in scan(path, rel(path, root))]

    return [Finding(*row) for found in each(root, paths, rows, owner=scan) for row in found]


def finder(roots: tuple[str, ...], scan: Callable[[Path, str], list[Finding]]):
    """The ``find(root)`` of a checker whose findings come from ``scan`` over each python file."""
    def find(root: Path) -> list[Finding]:
        return each_file(root, python_files(root, roots), scan)

    return find
