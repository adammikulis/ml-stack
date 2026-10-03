"""How every source ml-stack deals with has behaved, on two timescales: an encrypted per-user
graph, a score, a notice when a reliable source changes, and the calls that tighten what is
asked of it. ``docs/reputation.md`` has the rules."""

from __future__ import annotations

import atexit
import time
from typing import TYPE_CHECKING

from ml_stack.reputation.model import EVENTS, KINDS, State, canonical
from ml_stack.sentinel import observers

__all__ = ["EVENTS", "KINDS", "State", "canonical", "install", "page_read"]

if TYPE_CHECKING:
    from ml_stack.reputation.store import Ledger

_REGISTERED: list[Ledger] = []


def install(ledger: Ledger | None = None) -> Ledger:
    """Make ``ledger`` (by default the person's own) what net and peers report to, and raise
    the divergence notice through sentinel's dialog. Idempotent without an argument."""
    if ledger is None and _REGISTERED and observers.installed() is _REGISTERED[-1]:
        return _REGISTERED[-1]
    from ml_stack import sentinel
    from ml_stack.reputation.store import Ledger
    from ml_stack.reputation.notice import Notifier

    held = ledger or Ledger(clock=time.time)
    node = sentinel.default()
    notifier = Notifier(node.root / "notified.json", clock=held.clock, ledger=held)
    held.on_notice = notifier.on_divergence
    observers.install(held)
    _REGISTERED.append(held)
    atexit.register(held.flush)
    return held


def page_read(url: str, text: str) -> None:
    """A page the agent read: an injection-shaped one is noted against the URL it was fetched
    from. The text only decides whether the system saw an instruction; it names no source."""
    from ml_stack.guard.untrusted import injection_markers
    from ml_stack.net.policy import host_of

    if not injection_markers(text):
        return
    observers.observe("url", url, "injection_flagged")
    if host := host_of(url):
        observers.observe("host", host, "injection_flagged", scale=0.5)
