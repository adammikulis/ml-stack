"""The import graph of the files in this repository, and a hash of each file's whole closure."""

from __future__ import annotations

import ast
from pathlib import Path

from . import _store

SUFFIXES = (".py", ".pyi")
Spec = tuple[str, int, tuple[str, ...]]
"""One import statement: the module as written, its relative level, the names it takes."""


def specs(source: str) -> list[Spec]:
    """Every import statement in the source, nested ones included; empty if it does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out: list[Spec] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(alias.name, 0, ()) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append((node.module or "", node.level, tuple(a.name for a in node.names)))
    return out


class Resolver:
    """Finds the files under a root that an import could bind, listing each directory once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._dirs: dict[Path, frozenset[str]] = {}

    def listing(self, path: Path) -> frozenset[str]:
        """The names in a directory, empty when it is not one."""
        if path not in self._dirs:
            try:
                self._dirs[path] = frozenset(p.name for p in path.iterdir())
            except OSError:
                self._dirs[path] = frozenset()
        return self._dirs[path]

    def modules(self, base: Path, parts: list[str], last_only: bool = False) -> list[Path]:
        """The module files and package ``__init__`` files under base that parts walk through."""
        found: list[Path] = []
        here = base
        for number, part in enumerate(parts, 1):
            names = self.listing(here)
            wanted = number == len(parts) or not last_only
            if wanted:
                found += [here / (part + s) for s in SUFFIXES if part + s in names]
            if part not in names:
                break
            here = here / part
            if wanted:
                inner = self.listing(here)
                found += [here / ("__init__" + s) for s in SUFFIXES if "__init__" + s in inner]
        return found

    def resolve(self, spec: Spec, importer: Path) -> set[Path]:
        """The files under the root this import could bind, trying every place a resolver looks."""
        module, level, names = spec
        parts = [p for p in module.split(".") if p]
        if level:
            anchor = importer.parent
            for _ in range(level - 1):
                anchor = anchor.parent
            bases = [anchor]
            found = [anchor / ("__init__" + s) for s in SUFFIXES if "__init__" + s in self.listing(anchor)]
        else:
            bases = [self.root, self.root / "src", importer.parent]
            found = []
        out: set[Path] = set(found)
        for base in bases:
            out.update(self.modules(base, parts))
            out.update(p for name in names if name != "*"
                       for p in self.modules(base, [*parts, name], last_only=True))
        return {p for p in out if self.root in p.parents}


def components(graph: dict[Path, list[Path]]) -> list[list[Path]]:
    """The strongly connected groups of the graph, each after every group it points at."""
    index: dict[Path, int] = {}
    low: dict[Path, int] = {}
    held: list[Path] = []
    on: set[Path] = set()
    out: list[list[Path]] = []
    for start in graph:
        if start in index:
            continue
        work = [(start, iter(graph.get(start, [])))]
        index[start] = low[start] = len(index)
        held.append(start)
        on.add(start)
        while work:
            node, edges = work[-1]
            for nxt in edges:
                if nxt not in index:
                    index[nxt] = low[nxt] = len(index)
                    held.append(nxt)
                    on.add(nxt)
                    work.append((nxt, iter(graph.get(nxt, []))))
                    break
                if nxt in on:
                    low[node] = min(low[node], index[nxt])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    group = []
                    while True:
                        member = held.pop()
                        on.discard(member)
                        group.append(member)
                        if member == node:
                            break
                    out.append(group)
    return out


def closure_keys(graph: dict[Path, list[Path]], root: Path) -> dict[Path, str]:
    """A hash per file of its own text and the text of everything it imports, transitively."""
    keys: dict[Path, str] = {}
    for group in components(graph):
        members = sorted(group)
        below = sorted({keys[dep] for m in members for dep in graph.get(m, []) if dep in keys})
        lines = [f"{m.relative_to(root).as_posix()}:{_store.file_digest(m)}:"
                 + ",".join(d.relative_to(root).as_posix() for d in graph.get(m, []))
                 for m in members]
        key = _store.digest("\n".join([*lines, "--", *below]).encode())
        for m in members:
            keys[m] = key
    return keys
