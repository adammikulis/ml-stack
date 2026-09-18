"""Mutations of live code that no test notices, and the campaigns that looked."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from . import Finding
from ._mutation import candidates, functions

NAME = "mutation-survivors"
OWNER = "scripts/mutate"
LEDGER = "scripts/gates/survivors.txt"
STATES = ("survivor", "equivalent")
CAMPAIGN = "campaign"


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


@dataclass(frozen=True)
class Campaign:
    """One run of scripts/mutate: when it ran, what it sampled, what it found."""

    when: str
    commit: str
    sampled: str
    found: str

    def row(self) -> str:
        return "\t".join([CAMPAIGN, self.when, self.commit, self.sampled, self.found])


def describe() -> str:
    return ("A mutation of this function that every test it has kept green; the tests "
            "named do not measure what it returns. Write one that fails, then run "
            "scripts/mutate --verify.")


def campaigns(root: Path) -> list[Campaign]:
    """Every campaign the ledger records, oldest first."""
    ledger = root / LEDGER
    if not ledger.is_file():
        return []
    out = []
    for raw in ledger.read_text(encoding="utf-8").splitlines():
        parts = [f.strip() for f in raw.split("\t")]
        if len(parts) == 5 and parts[0] == CAMPAIGN:
            out.append(Campaign(*parts[1:]))
    return out


def record(root: Path, campaign: Campaign) -> None:
    """Append one campaign to the ledger."""
    ledger = root / LEDGER
    text = ledger.read_text(encoding="utf-8") if ledger.is_file() else ""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(text + ("" if text.endswith("\n") or not text else "\n")
                      + campaign.row() + "\n", encoding="utf-8")


def notes(root: Path) -> list[str]:
    """What the count does and does not cover, for anyone reading a zero."""
    ran = campaigns(root)
    last = (f"    The last ran on {ran[-1].when} at {ran[-1].commit}: {ran[-1].sampled}."
            if ran else "    No campaign is recorded, so the count covers nothing.")
    return [f"{NAME} counts recorded survivors, not every mutation the tests would miss.",
            f"    A campaign reads a sample; {len(candidates(root))} functions here are "
            f"mutable.",
            last,
            "    Another commit samples others, and a deeper run finds more mutations of "
            "the same function.",
            f"    {LEDGER} lists every campaign and what it found."]


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
