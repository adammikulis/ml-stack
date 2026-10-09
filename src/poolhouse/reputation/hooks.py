"""Putting the ledger in place for a process and reading pages against it."""

from __future__ import annotations

import atexit
import time

from poolhouse import sentinel
from poolhouse.guard.untrusted import injection_markers
from poolhouse.net.policy import host_of
from poolhouse.reputation.notice import Notifier
from poolhouse.reputation.store import Ledger
from poolhouse.sentinel import observers

__all__ = ["install", "page_read"]

_REGISTERED: list[Ledger] = []


def install(ledger: Ledger | None = None) -> Ledger:
    """Make ``ledger`` (by default the person's own) what net and peers report to, and raise
    the divergence notice through sentinel's dialog. Idempotent without an argument."""
    if ledger is None and _REGISTERED and observers.installed() is _REGISTERED[-1]:
        return _REGISTERED[-1]
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
    if not injection_markers(text):
        return
    observers.observe("url", url, "injection_flagged")
    if host := host_of(url):
        observers.observe("host", host, "injection_flagged", scale=0.5)
