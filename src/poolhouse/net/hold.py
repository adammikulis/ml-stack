"""Where a download that failed a check goes: sentinel's quarantine store, with an event."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from poolhouse import files, home, sentinel as package
from poolhouse.sentinel.store import Holding

__all__ = ["Hold", "SentinelHold", "rejected_dir", "staging_dir"]

logger = logging.getLogger(__name__)


def staging_dir() -> Path:
    """The private directory downloads are written to before they are checked (mode 0700)."""
    where = home.state("net", "staging")
    where.mkdir(parents=True, exist_ok=True, mode=0o700)
    where.chmod(0o700)
    return where


def rejected_dir() -> Path:
    """Where a rejected file stays when sentinel cannot take it."""
    where = staging_dir() / "rejected"
    where.mkdir(parents=True, exist_ok=True, mode=0o700)
    return where


class Hold(Protocol):
    """Takes a file that failed a check and keeps it out of use."""

    def hold(self, path: Path, reason: str, evidence: dict[str, Any]) -> str:
        """Hold ``path`` and return where it is kept ('' when it was removed)."""
        ...


class SentinelHold:
    """Moves the file into sentinel's quarantine store as an ``artifact`` record and lets the
    store raise its event. The file stays in `rejected_dir` when the store cannot take it."""

    def __init__(self, sentinel: Any | None = None) -> None:
        self._sentinel = sentinel

    def hold(self, path: Path, reason: str, evidence: dict[str, Any]) -> str:
        try:
            sentinel = self._sentinel or package.default()
            key = str(evidence.get("sha256") or path.name)
            record = sentinel.store.quarantine(("artifact", f"download:{key}"), reason, evidence,
                                               Holding(path=path), actor="net")
            if record is not None and record.action:
                return str(record.action.get("held", ""))
        except (OSError, ImportError, ValueError) as exc:
            logger.warning("sentinel could not quarantine %s: %s", path.name, exc)
        if path.exists():
            target = rejected_dir() / f"{path.name}.rejected"
            files.promote(path, target)
            return str(target)
        return ""
