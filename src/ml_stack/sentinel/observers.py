"""The seam between the places that see a source behave (net, peers) and the reputation
ledger: calls that do nothing until a ledger is installed."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ml_stack import home

__all__ = ["FAILURES", "Gate", "Observer", "clean", "gate", "install", "installed", "line", "observe",
           "summary", "trait", "uninstall"]

logger = logging.getLogger("ml_stack.reputation")
FAILURES = (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError)
"""What an observation point swallows: a ledger that is locked, full or gone never breaks a fetch."""


@dataclass(frozen=True, slots=True)
class Gate:
    """Why a source is to be asked about or refused, and since when."""

    state: str
    reason: str
    since: float


class Observer(Protocol):
    def observe(self, kind: str, key: str, event: str, *, scale: float = 1.0) -> object: ...

    def trait(self, kind: str, key: str, name: str, value: str) -> None: ...

    def clean(self, kind: str, key: str) -> None: ...

    def gate(self, kind: str, key: str) -> Gate | None: ...


_ACTIVE: Observer | None = None


def install(observer: Observer) -> None:
    """Make ``observer`` the ledger every observation point reports to."""
    global _ACTIVE
    _ACTIVE = observer


def uninstall() -> None:
    global _ACTIVE
    _ACTIVE = None


def installed() -> Observer | None:
    return _ACTIVE


def _safely(call: str, *args: object, **kw: object) -> None:
    if _ACTIVE is None:
        return
    try:
        getattr(_ACTIVE, call)(*args, **kw)
    except FAILURES as exc:
        logger.warning("reputation %s failed: %s", call, type(exc).__name__)


def observe(kind: str, key: str, event: str, *, scale: float = 1.0) -> None:
    """Report a bad event the system saw. Never raises."""
    _safely("observe", kind, key, event, scale=scale)


def trait(kind: str, key: str, name: str, value: str) -> None:
    """Report a trait of a source. Never raises."""
    _safely("trait", kind, key, name, value)


def clean(kind: str, key: str) -> None:
    """Report one clean interaction. Never raises."""
    _safely("clean", kind, key)


def gate(kind: str, key: str) -> Gate | None:
    """What the ledger says against a source, else None. Never raises."""
    if _ACTIVE is None:
        return None
    try:
        return _ACTIVE.gate(kind, key)
    except FAILURES:
        return None


def summary() -> dict[str, Any]:
    """Counts of sources by state and waiting notices, from the plaintext summary beside the
    sealed store (counts only, no names); empty when no ledger has written one."""
    for path in sorted(home.state("reputation").glob("u-*/summary.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and all(isinstance(doc.get(k), int) for k in ("watch", "bad", "notices")):
            return doc
    return {}


def line() -> str:
    """One status line about reputation, or an empty string when there is nothing to say."""
    held = summary()
    if not held:
        return ""
    waiting = f", {held['notices']} notice(s) waiting" if held["notices"] else ""
    return (f"reputation: {held.get('established', 0)} established, {held['watch']} watched, "
            f"{held['bad']} bad{waiting}")
