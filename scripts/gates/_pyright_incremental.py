"""Pyright's findings kept per file, so an edit re-checks only the files it can change."""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from collections.abc import Callable
from functools import cache
from pathlib import Path

from . import Finding, _imports, _pins, _store
from ._util import rel

BATCH = 100
"""Files named to one pyright run, so the command line stays short."""
FULL_ABOVE = 0.5
"""Past this share of files stale, one whole-project run is cheaper than naming them."""
UNSUPPORTED = {"exclude", "ignore", "extends", "executionEnvironments", "extraPaths",
               "stubPath", "venvPath", "venv", "pythonPath", "strict"}
"""Settings that change which files are checked or how an import resolves."""
ENVIRONMENT = ("import sys, importlib.metadata as m\n"
               "print(sys.version)\n"
               "print(sorted(f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions()))")

Run = Callable[[list[str] | None], list[Finding]]


def settings(root: Path) -> dict | None:
    """The [tool.pyright] table this checker can follow, None when the project needs a full run."""
    if (root / "pyrightconfig.json").exists():
        return None
    try:
        table = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    found = table.get("tool", {}).get("pyright")
    if not isinstance(found, dict) or UNSUPPORTED & set(found):
        return None
    include = found.get("include")
    if not isinstance(include, list) or any(not isinstance(i, str) or "*" in i for i in include):
        return None
    return found


def project_files(root: Path, include: list[str]) -> list[Path]:
    """Every source file pyright would check under these include paths, sorted."""
    found: set[Path] = set()
    for entry in include:
        place = root / entry
        if place.is_file():
            found.add(place.resolve())
        for here, dirs, names in os.walk(place):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")
                       and not (Path(here) / d / "pyvenv.cfg").exists()]
            found.update((Path(here) / n).resolve() for n in names if n.endswith(_imports.SUFFIXES))
    return sorted(found)


@cache
def interpreter_stamp(root: Path) -> str:
    """The interpreter pyright finds on the path and every distribution installed in it."""
    exe = shutil.which("python3") or shutil.which("python")
    if exe is None:
        return ""
    done = subprocess.run([exe, "-c", ENVIRONMENT], cwd=root, capture_output=True,
                          text=True, check=False)
    return _store.digest(f"{exe}\0{done.stdout}".encode())


def environment(root: Path) -> str:
    """Everything but the project's own files that the answers depend on."""
    cmd = _pins.tool("pyright") or ()
    here = Path(__file__).resolve().parent
    code = _store.stamp(here / "_pyright_incremental.py", here / "_imports.py", here / "_store.py",
                        root / "pyproject.toml", *(Path(c) for c in cmd[:1]))
    return _store.digest(f"{code}\0{_pins.version('pyright')}\0{interpreter_stamp(root)}".encode())


def graph_of(root: Path, files: list[Path], kept: dict) -> dict[Path, list[Path]]:
    """Each file reachable from files, with the files under root it imports; reuses ``kept`` specs."""
    resolver = _imports.Resolver(root)
    graph: dict[Path, list[Path]] = {}
    pending = list(files)
    while pending:
        path = pending.pop()
        if path in graph:
            continue
        where = rel(path, root)
        digest = _store.file_digest(path)
        have = kept.get(where)
        if have is None or have[0] != digest:
            text = path.read_text(encoding="utf-8", errors="replace")
            have = kept[where] = [digest, [list(s) for s in _imports.specs(text)]]
        found: set[Path] = set()
        for module, level, names in have[1]:
            found |= resolver.resolve((module, level, tuple(names)), path)
        graph[path] = sorted(found - {path})
        pending.extend(graph[path])
    return graph


def by_file(found: list[Finding]) -> dict[str, list[list]]:
    """Findings grouped as ``{path: [[line, detail], ...]}``."""
    grouped: dict[str, list[list]] = {}
    for item in found:
        grouped.setdefault(item.path, []).append([item.line, item.detail])
    return grouped


def findings(root: Path, run: Run) -> list[Finding]:
    """Pyright's findings for the project at root, re-running pyright only for stale files.

    A file is stale when its own text or any file it imports, transitively, changed since
    its rows were stored; ``run(None)`` is the whole-project run that stays authoritative.
    """
    root = root.resolve()
    config = settings(root)
    if config is None:
        return run(None)
    files = project_files(root, config["include"])
    name = "pyright-" + _store.digest(str(root).encode())[:16]
    kept = {} if _store.forced() else _store.load(name)
    env = environment(root)
    if kept.get("environment") != env:
        kept = {}
    specs_kept = kept.setdefault("specs", {})
    keys = _imports.closure_keys(graph_of(root, files, specs_kept), root)
    names = {path: rel(path, root) for path in files}
    stored = kept.get("files", {})
    stale = [names[p] for p in files if stored.get(names[p], {}).get("key") != keys[p]]
    changed = set(stale)
    rows = {n: stored[n]["rows"] for n in names.values() if n not in changed}
    whole = bool(stale) and len(stale) > len(files) * FULL_ABOVE
    if stale:
        wanted = list(names.values()) if whole else stale
        found = run(None) if whole else [f for at in range(0, len(stale), BATCH)
                                         for f in run(stale[at:at + BATCH])]
        grouped = by_file(found)
        if set(grouped) - set(wanted):
            return found if whole else run(None)
        rows.update({n: grouped.get(n, []) for n in wanted})
    if stale or set(stored) != set(names.values()):
        _store.save(name, {"environment": env, "specs": specs_kept,
                           "files": {names[p]: {"key": keys[p], "rows": rows[names[p]]}
                                     for p in files}})
    return [Finding(path, int(line), detail) for path, found in rows.items() for line, detail in found]
