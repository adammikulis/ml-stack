"""Malware scanning of a downloaded file: a `Scanner` protocol, the verdict of one or several
scanners, and the policy that says whether a file nobody could scan may be kept."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from poolhouse import files, home

__all__ = ["Outcome", "ScanPolicy", "ScanResult", "Scanner", "Summary", "category", "scan_file",
           "summarise"]

UNSCANNED_ENV = "POOLHOUSE_NET_UNSCANNED"
SCAN_MODELS_ENV = "POOLHOUSE_NET_SCAN_MODELS"
CATEGORIES = ("executable", "archive", "model", "data")
ACTIONS = ("refuse", "warn", "allow")


class Outcome(StrEnum):
    """What one scanner (or all of them together) says about a file."""

    CLEAN = "clean"
    INFECTED = "infected"
    ERROR = "error"
    NO_SCANNER = "no scanner available"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ScanResult:
    """One scanner's answer: its name, the outcome, the signature found and any warnings."""

    scanner: str
    outcome: Outcome
    detail: str = ""
    signature: str = ""
    warnings: tuple[str, ...] = ()


class Scanner(Protocol):
    """Something that looks at one file. ``available`` is False when its tool is missing."""

    name: str

    def available(self) -> bool:
        """Whether this scanner can run on this machine."""
        ...

    def scan(self, path: Path) -> ScanResult:
        """Scan ``path``; never raises for a scanner fault, it answers `Outcome.ERROR`."""
        ...


@dataclass(frozen=True, slots=True)
class Summary:
    """All scanners together: the outcome, every result, and the status line to show."""

    outcome: Outcome
    results: tuple[ScanResult, ...] = field(default=())

    @property
    def line(self) -> str:
        """`scanned: clean (clamav)`, `scanned: no scanner available` and so on."""
        if self.outcome == Outcome.NO_SCANNER:
            return "scanned: no scanner available"
        if self.outcome == Outcome.SKIPPED:
            return "scan skipped: model weights (a virus scanner cannot judge them; format checked)"
        names = ", ".join(r.scanner for r in self.results if r.outcome == self.outcome)
        return f"scanned: {self.outcome.value} ({names})" if names else f"scanned: {self.outcome.value}"


def summarise(results: Iterable[ScanResult]) -> Summary:
    """Combine results: infected if any says so, else clean if one real scan said clean, else
    error if one failed, else no scanner."""
    rows = tuple(results)
    for outcome in (Outcome.INFECTED, Outcome.CLEAN, Outcome.ERROR):
        if any(r.outcome == outcome for r in rows):
            return Summary(outcome, rows)
    return Summary(Outcome.NO_SCANNER, rows)


def scan_file(path: Path, scanners: Sequence[Scanner]) -> Summary:
    """Run every available scanner over ``path``. No scanner at all is `NO_SCANNER`, never clean."""
    results = []
    for scanner in scanners:
        try:
            if not scanner.available():
                continue
            results.append(scanner.scan(path))
        except OSError as exc:
            results.append(ScanResult(scanner.name, Outcome.ERROR, f"{type(exc).__name__}: {exc}"))
    return summarise(results)


def category(kind: str) -> str:
    """The policy category of a file kind: executable, archive, model or data."""
    if kind == "executable":
        return "executable"
    if kind in ("zip", "tar", "archive", "gzip"):
        return "archive"
    if kind in ("gguf", "safetensors", "model"):
        return "model"
    return "data"


@dataclass(frozen=True, slots=True)
class ScanPolicy:
    """Whether a file may be kept when it was not scanned: ``refuse``, ``warn`` or ``allow``
    per category. An infected file is never kept whatever the policy says."""

    executable: str = "refuse"
    archive: str = "refuse"
    model: str = "warn"
    data: str = "warn"
    scan_models: bool = False

    @classmethod
    def load(cls) -> ScanPolicy:
        """The policy a person saved with `save`, with the environment applied over it."""
        saved = files.read_json(home.state("net", "scan-policy.json"), {})
        base = cls(**{k: v for k, v in saved.items()
                      if k in ("executable", "archive", "model", "data", "scan_models")
                      and (isinstance(v, bool) if k == "scan_models" else v in ACTIONS)})
        return base.from_env()

    def save(self) -> None:
        """Keep this policy for every later run."""
        files.write_json(home.state("net", "scan-policy.json"),
                         files.versioned(asdict(self), 1))

    def from_env(self) -> ScanPolicy:
        """This policy with ``POOLHOUSE_NET_UNSCANNED=archive:warn,model:allow`` applied."""
        values: dict[str, Any] = {}
        for part in os.environ.get(UNSCANNED_ENV, "").split(","):
            name, _, action = part.partition(":")
            name, action = name.strip(), action.strip()
            if name in CATEGORIES and action in ACTIONS:
                values[name] = action
        if os.environ.get(SCAN_MODELS_ENV) == "1":
            values["scan_models"] = True
        return replace(self, **values)

    def scan(self, path: Path, kind: str, scanners: Sequence[Scanner]) -> Summary:
        """The scan of ``path``; model weights are skipped unless ``scan_models``."""
        if category(kind) == "model" and not self.scan_models:
            return Summary(Outcome.SKIPPED)
        return scan_file(path, scanners)

    def action(self, kind: str) -> str:
        """The policy for a file of ``kind``."""
        return str(getattr(self, category(kind)))

    def decide(self, kind: str, summary: Summary, *, allow_unscanned: bool = False
               ) -> tuple[bool, str]:
        """``(keep, why)`` for a file of ``kind`` that scanned as ``summary``."""
        if summary.outcome == Outcome.INFECTED:
            return False, summary.line
        if summary.outcome in (Outcome.CLEAN, Outcome.SKIPPED):
            return True, summary.line
        action = "allow" if allow_unscanned else self.action(kind)
        if action == "refuse":
            return False, (f"{summary.line}; {category(kind)} files are not kept unscanned "
                           f"(install ClamAV, or `poolhouse-security scan-policy {category(kind)} warn`)")
        return True, summary.line + ("; weights cannot be judged by a virus scanner"
                                     if category(kind) == "model" else "")
