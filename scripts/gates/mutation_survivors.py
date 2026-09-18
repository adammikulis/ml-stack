"""Mutations of live code that no test notices, as recorded by scripts/mutate."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from . import Finding
from ._mutation import functions

NAME = "mutation-survivors"
OWNER = "scripts/mutate"
LEDGER = "scripts/gates/survivors.txt"
STATES = ("survivor", "equivalent")


@dataclass(frozen=True)
class Entry:
    """One recorded mutation: where it applies, which one it is, and what was said of it."""

    state: str
    site: str
    mutation: str
    note: str

    @property
    def path(self) -> str:
        return self.site.split("::")[0]

    @property
    def qualname(self) -> str:
        return self.site.split("::")[-1]


def describe() -> str:
    return ("A mutation of this function that every test it has kept green; the tests "
            "named do not measure what it returns. Write one that fails, then run "
            "scripts/mutate --verify.")


def entries(root: Path) -> list[Entry]:
    """Every row of the ledger, in file order."""
    ledger = root / LEDGER
    if not ledger.is_file():
        return []
    out = []
    for raw in ledger.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [field.strip() for field in line.split("\t") if field.strip()]
        if len(parts) < 3 or parts[0] not in STATES:
            continue
        out.append(Entry(parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else ""))
    return out


def located(root: Path, entry: Entry) -> int | None:
    """The line the entry's function starts on, or None when it is no longer there."""
    path = root / entry.path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for qualname, node in functions(tree):
        if qualname == entry.qualname:
            return node.lineno
    return None


def find(root: Path) -> list[Finding]:
    out = []
    for entry in entries(root):
        if entry.state != "survivor":
            continue
        line = located(root, entry)
        if line is None:
            continue
        where = f" -- green: {entry.note}" if entry.note else ""
        out.append(Finding(entry.path, line, f"{entry.qualname}: {entry.mutation}{where}"))
    return out
