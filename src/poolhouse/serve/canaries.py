"""The scheduled behavioural canaries: a small fixed prompt set asked of each served model, scored
against the baseline recorded the first time the model answered, inside the sentinel's scan loop.

Cost is bounded: ``canary.CANARY_PROBES`` (six probes) times ``canary.RUNS`` (three), at
temperature 0 with at most ``MAX_TOKENS`` generated each, once per ``POOLHOUSE_SENTINEL_CANARY``
seconds (default one hour) per model. A server that is processing something is not asked, and the
requests go through ``poolhouse.http`` and so queue behind real traffic in the machine's request
gate. A drift is a watch; a hard drift, confirmed by a second run straight after, quarantines the
model (moved aside, its servers stopped, restored by a person).
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from poolhouse.client import Client
from poolhouse.http import ServerError
from poolhouse.sentinel import Sentinel, canary
from poolhouse.sentinel.events import Event, Severity
from poolhouse.sentinel.findings import Finding
from poolhouse.sentinel.watch import canary_cadence, opt_out
from poolhouse.serve.leases import recorded_servers
from poolhouse.serve.ports import DEFAULT_HOST
from poolhouse.serve.process import pid_exists
from poolhouse.serve.reclaim import busy_now

__all__ = ["MAX_TOKENS", "Canaries", "Target", "asker", "check", "lease_file_targets", "schedule"]

logger = logging.getLogger("poolhouse.sentinel")

MAX_TOKENS = 64
LEASE_WAIT_S = 15.0


@dataclass(frozen=True, slots=True)
class Target:
    """A served model: what it is called (the path or name it was started from), a way to ask
    it (``using`` is a context manager yielding the base URL, holding a lease while open)."""

    model: str
    using: Callable[[], AbstractContextManager[str]]


def asker(base_url: str) -> Callable[[str], str]:
    """``ask(prompt) -> reply`` over the server's chat endpoint, greedy (the client's default)
    and at most ``MAX_TOKENS`` long."""
    client = Client(base_url.rstrip("/"))

    def ask(prompt: str) -> str:
        reply = client.chat([{"role": "user", "content": prompt}], n_predict=MAX_TOKENS,
                            timeout=60.0)
        return str(reply.content or "")
    return ask


def _key(node: Sentinel, model: str) -> str:
    """The baseline key: the model, and its pin when it has one, so a replaced file starts a
    new baseline instead of being judged by the old one."""
    pin = node.manifest.pin_of(model)
    return f"{model}@{pin.sha256[:16]}" if pin else model


def check(node: Sentinel, target: Target) -> Finding | None:
    """One canary round for ``target``: records the baseline when there is none, else compares
    and hands the sentinel a watch or, once confirmed, a hard-drift finding. None when the model
    held steady, was just baselined, or could not be asked."""
    key = _key(node, target.model)
    try:
        with target.using() as base_url:
            if busy_now(base_url):
                return None
            ask = asker(base_url)
            base = node.baselines.get(key)
            if base is None:
                node.baselines.record(key, canary.run(ask, canary.CANARY_PROBES, canary.RUNS))
                node.bus.emit(Event("canary.baselined", Severity.INFO, "canary",
                                    f"model:{target.model}", {}, node.clock()))
                return None
            now = canary.run(ask, canary.CANARY_PROBES, canary.RUNS)
            level = canary.assess(base, now)
            if level == "hard":
                now = canary.run(ask, canary.CANARY_PROBES, canary.RUNS)
                level = canary.assess(base, now)
    except (ServerError, OSError, ValueError, RuntimeError) as exc:
        node.bus.emit(Event("canary.skipped", Severity.INFO, "canary", f"model:{target.model}",
                            {"why": type(exc).__name__}, node.clock()))
        return None
    if level == "ok":
        return None
    path = target.model if Path(target.model).expanduser().is_file() else ""
    found = canary.drift_finding(target.model, base, now, hard=level == "hard", path=path)
    if found is not None:
        node.handle(found)
    return found


class Canaries:
    """The scan loop's canary round: every model ``targets`` names that is due."""

    def __init__(self, node: Sentinel, targets: Callable[[], Iterable[Target]],
                 interval_s: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.node, self.targets, self.interval_s, self.clock = node, targets, interval_s, clock
        self._last: dict[str, float] = {}

    def __call__(self) -> list[Finding]:
        out = []
        for target in self.targets():
            now = self.clock()
            if target.model in self._last and now - self._last[target.model] < self.interval_s:
                continue
            if self.node.store.blocked("model", target.model):
                continue
            self._last[target.model] = now
            found = check(self.node, target)
            if found is not None:
                out.append(found)
        return out


def lease_file_targets(state_file: Path) -> Callable[[], list[Target]]:
    """The servers poolhouse recorded as running in ``state_file`` (the fleet daemon has no
    broker of its own), asked without a lease."""
    def targets() -> list[Target]:
        out = []
        for port, entry in recorded_servers(state_file).items():
            pid, model = entry.get("pid"), str(entry.get("model") or "")
            if model and isinstance(pid, int) and pid_exists(pid) and not entry.get("unmanaged"):
                url = f"http://{DEFAULT_HOST}:{port}"
                out.append(Target(model, lambda url=url: contextlib.nullcontext(url)))
        return out
    return targets


def schedule(node: Sentinel, targets: Callable[[], list[Target]]) -> Canaries | None:
    """The canary round for a scan loop at the configured cadence; None when the mode is off
    or the canaries were switched off with a reason (logged)."""
    if node.mode.value == "off":
        return None
    chosen = canary_cadence()
    if chosen.refused:
        logger.warning("sentinel: %s; canaries every %.0fs", chosen.refused, chosen.interval_s)
        node.bus.emit(Event("sentinel.canary_off_refused", Severity.WARNING, "watch", "",
                            {"why": chosen.refused}, node.clock()))
    if chosen.interval_s <= 0:
        opt_out(node, "the behavioural canaries", chosen.off_because)
        return None
    return Canaries(node, targets, chosen.interval_s)
