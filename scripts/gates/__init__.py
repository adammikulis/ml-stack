"""Checkers that count the shapes this repository has a budget for."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


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


def skipped(checker: ModuleType) -> str:
    """Why this checker cannot run here, empty when it can."""
    reason = getattr(checker, "skip", None)
    return reason() if reason is not None else ""


def unrunnable() -> dict[str, str]:
    """Metric name -> why it cannot be counted here."""
    return {c.NAME: r for c in checkers() if (r := skipped(c))}


def run(root: Path) -> dict[str, list[Finding]]:
    """Every runnable checker's findings over the tree at root, keyed by metric name."""
    return {
        c.NAME: sorted(c.find(root), key=lambda f: (f.path, f.line, f.detail))
        for c in checkers()
        if not skipped(c)
    }
