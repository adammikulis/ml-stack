"""Checkers that count the shapes this repository has a budget for."""

from __future__ import annotations

import importlib
import pkgutil
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

from ._util import remembered, tree_fingerprint


@dataclass(frozen=True)
class Finding:
    """One site a checker objects to."""

    path: str
    line: int
    detail: str = ""


def checkers() -> list[ModuleType]:
    """Every checker module in this package, by name."""
    found = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for attr in ("NAME", "OWNER", "describe", "find"):
            if not hasattr(module, attr):
                raise AttributeError(f"gates.{info.name} has no {attr}")
        found.append(module)
    return found


def hard() -> set[str]:
    """The metrics with no budget, where one finding fails."""
    return {c.NAME for c in checkers() if getattr(c, "HARD", False)}


def skipped(checker: ModuleType) -> str:
    """Why this checker cannot run here, empty when it can."""
    reason = getattr(checker, "skip", None)
    return reason() if reason is not None else ""


def notes(root: Path) -> list[str]:
    """Every checker's lines about what its count does not cover."""
    out: list[str] = []
    for checker in checkers():
        said = getattr(checker, "notes", None)
        if said is not None and not skipped(checker):
            out.extend(said(root))
    return out


def unrunnable() -> dict[str, str]:
    """Metric name -> why it cannot be counted here."""
    return {c.NAME: r for c in checkers() if (r := skipped(c))}


def findings(checker: ModuleType, root: Path) -> list[Finding]:
    """One checker's findings over the tree at root, in path order.

    The tree this package sits in is remembered by content, with the version of each
    external tool a checker may call, so an unchanged tree is not scanned twice.
    """
    def scan() -> list[list]:
        found = sorted(checker.find(root), key=lambda f: (f.path, f.line, f.detail))
        return [[f.path, f.line, f.detail] for f in found]

    if root.resolve() != Path(__file__).resolve().parents[2]:
        return [Finding(*row) for row in scan()]
    tools = ",".join(f"{t}={_stamp(t)}" for t in ("ruff", "pyright"))
    return [Finding(*row) for row in remembered(root, checker.NAME, scan, tools,
                                                 repo_fingerprint())]


@cache
def repo_fingerprint() -> str:
    """The fingerprint of the tree this package sits in, taken once per process."""
    return tree_fingerprint(Path(__file__).resolve().parents[2])


def _stamp(tool: str) -> str:
    """Where a tool is and when it was installed, empty when it is absent."""
    path = shutil.which(tool)
    return "" if path is None else f"{path}@{Path(path).resolve().stat().st_mtime_ns}"


def run(root: Path) -> dict[str, list[Finding]]:
    """Every runnable checker's findings over the tree at root, keyed by metric name."""
    return {c.NAME: findings(c, root) for c in checkers() if not skipped(c)}
