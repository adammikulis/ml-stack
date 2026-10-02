"""Windowed counts: failures and replays per peer, tool-call mix per session, abuse per caller."""

from __future__ import annotations

import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from ml_stack.sentinel.events import Severity
from ml_stack.sentinel.findings import HEURISTIC, HIGH, Finding, finding

__all__ = ["Abuse", "PeerLimits", "PeerWatch", "ToolMix", "Windows", "total_variation"]


class Windows:
    """Weighted counts of named things per key over a sliding window, bounded in keys."""

    def __init__(self, window_s: float = 60.0, *, most_keys: int = 4096,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s, self.most, self.clock = window_s, most_keys, clock
        self._seen: OrderedDict[tuple[str, str], deque[tuple[float, int]]] = OrderedDict()

    def add(self, key: str, what: str, n: int = 1) -> int:
        """Count ``n`` of ``what`` for ``key`` now; returns the total inside the window."""
        now = self.clock()
        slot = self._seen.pop((key, what), deque())
        slot.append((now, n))
        self._trim(slot, now)
        self._seen[(key, what)] = slot
        while len(self._seen) > self.most:
            self._seen.popitem(last=False)
        return sum(w for _, w in slot)

    def count(self, key: str, what: str) -> int:
        """The total for ``what`` under ``key`` inside the window."""
        slot = self._seen.get((key, what))
        if slot is None:
            return 0
        self._trim(slot, self.clock())
        return sum(w for _, w in slot)

    def _trim(self, slot: deque[tuple[float, int]], now: float) -> None:
        while slot and now - slot[0][0] > self.window_s:
            slot.popleft()


@dataclass(frozen=True, slots=True)
class PeerLimits:
    """Counts inside one window that make a peer suspicious (``watch_*``) or make its
    traffic forged beyond doubt (``forged``)."""

    watch_bad: int = 10
    forged: int = 30
    watch_replay: int = 1
    replay_forged: int = 3
    rate: int = 600
    flaps: int = 6


class PeerWatch:
    """Turns a peer's request outcomes into findings."""

    OUTCOMES = ("ok", "bad_sig", "replay", "clock", "locked", "oversize", "unsigned")

    def __init__(self, limits: PeerLimits | None = None, *, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits or PeerLimits()
        self.windows = Windows(window_s, clock=clock)
        self._reported: dict[tuple[str, str], int] = {}

    def note(self, peer: str, outcome: str) -> Finding | None:
        """Count one request outcome from ``peer``; a finding when a limit is crossed."""
        if outcome not in self.OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        requests = self.windows.add(peer, "request")
        bad = (self.windows.add(peer, "bad") if outcome in ("bad_sig", "unsigned", "locked")
               else self.windows.count(peer, "bad"))
        replays = (self.windows.add(peer, "replay") if outcome == "replay"
                   else self.windows.count(peer, "replay"))
        lim = self.limits
        counts = {"bad": bad, "replays": replays, "requests": requests}
        if replays >= lim.replay_forged or bad + replays >= lim.forged:
            return self._once(peer, "forged", finding(
                "peer.forged_traffic", Severity.CRITICAL, ("peer", peer), HIGH, counts))
        if bad >= lim.watch_bad or replays >= lim.watch_replay:
            return self._once(peer, "failing", finding(
                "peer.auth_failures", Severity.WARNING, ("peer", peer), HEURISTIC,
                counts))
        if requests >= lim.rate:
            return self._once(peer, "rate", finding(
                "peer.rate", Severity.WARNING, ("peer", peer), HEURISTIC, counts))
        return None

    def _once(self, peer: str, tag: str, found: Finding) -> Finding | None:
        """Report a condition once per window rather than once per request."""
        now = self.windows.clock()
        last = self._reported.get((peer, tag))
        if last is not None and now - last < self.windows.window_s:
            return None
        self._reported[(peer, tag)] = int(now)
        if len(self._reported) > 4096:
            self._reported.clear()
        return found

    def joined(self, peer: str) -> Finding | None:
        """Count a join or leave; a finding when a peer flaps."""
        n = self.windows.add(peer, "flap")
        if n >= self.limits.flaps:
            return self._once(peer, "flap", finding(
                "peer.flapping", Severity.NOTICE, ("peer", peer), HEURISTIC,
                {"changes": n}))
        return None

    def mismatch(self, peer: str, what: str, theirs: str, pinned: str) -> Finding | None:
        """A finding when a peer reports a ``version`` or ``binary`` that is not the pinned one."""
        if not pinned or theirs == pinned:
            return None
        return finding(f"peer.{what}_mismatch", Severity.WARNING, ("peer", peer), HIGH,
                       {"reported": theirs, "pinned": pinned})


def total_variation(a: Counter[str], b: Counter[str]) -> float:
    """Half the sum of absolute differences between two normalised count tables, 0 to 1."""
    sa, sb = sum(a.values()) or 1, sum(b.values()) or 1
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / sa - b[k] / sb) for k in keys)


class ToolMix:
    """Compares the tools a session calls now with the first calls it made."""

    def __init__(self, *, baseline_calls: int = 50, window: int = 30,
                 threshold: float = 0.5) -> None:
        self.baseline_calls, self.window, self.threshold = baseline_calls, window, threshold
        self._base: dict[str, Counter[str]] = {}
        self._recent: dict[str, deque[str]] = {}

    def note(self, session: str, tool: str) -> Finding | None:
        """Count one call; a finding when the latest window differs from the baseline."""
        base = self._base.setdefault(session, Counter())
        if sum(base.values()) < self.baseline_calls:
            base[tool] += 1
            return None
        recent = self._recent.setdefault(session, deque(maxlen=self.window))
        recent.append(tool)
        if len(recent) < self.window:
            return None
        distance = total_variation(base, Counter(recent))
        if distance < self.threshold:
            return None
        recent.clear()
        return finding("tools.mix_shift", Severity.NOTICE, ("session", session), HEURISTIC, {"distance": round(distance, 3), "calls": sum(base.values())})


@dataclass(frozen=True, slots=True)
class AbuseLimits:
    """What one caller may use inside one window."""

    calls: int = 120
    result_bytes: int = 8 * 1024 * 1024
    seconds: float = 600.0


class Abuse:
    """Counts calls, result bytes and wall seconds per caller."""

    def __init__(self, limits: AbuseLimits | None = None, *, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits or AbuseLimits()
        self.windows = Windows(window_s, clock=clock)

    def note(self, caller: str, *, result_bytes: int = 0, seconds: float = 0.0) -> Finding | None:
        """Count one call by ``caller``; a finding when a limit is crossed."""
        calls = self.windows.add(caller, "calls")
        size = self.windows.add(caller, "bytes", result_bytes) if result_bytes else \
            self.windows.count(caller, "bytes")
        spent = self.windows.add(caller, "ms", int(seconds * 1000)) if seconds else \
            self.windows.count(caller, "ms")
        lim = self.limits
        if calls == lim.calls or (size >= lim.result_bytes > size - result_bytes) \
                or (spent >= lim.seconds * 1000 > spent - seconds * 1000):
            return finding("abuse.resource", Severity.WARNING, ("caller", caller), HEURISTIC, {"calls": calls, "bytes": size, "seconds": spent / 1000})
        return None
