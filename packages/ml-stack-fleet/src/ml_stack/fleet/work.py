"""Spread a list of jobs over the peers that can run them."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .bench import BENCH_KIND
from .pool import Requires, Score, candidates, eligible
from .rates import Rates
from .remote import Peer

__all__ = ["Placement", "Unit", "run"]

QUARANTINE_AFTER = 3
"""Consecutive failures before a peer is set aside. Three, not one: a peer that fails"""


@dataclass(frozen=True, slots=True)
class Unit:
    """One piece of work, and what it needs from whatever runs it."""

    id: str
    argv: Sequence[str]
    name: str = ""
    cwd: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    requires: Requires = Requires()
    peer: str = ""
    """Pin to one named peer. Empty means place it automatically."""
    work: float = 1.0
    """How much work this is, in whatever unit ``kind``'s measured rate is in."""


@dataclass
class Placement:
    """Where a unit ended up, and how it went."""

    unit_id: str
    state: str = "pending"
    """pending | done | failed | stopped | unreachable | unplaceable"""
    peer: str = ""
    base_url: str = ""
    job_id: str = ""
    returncode: int | None = None
    attempts: int = 0
    tried: tuple[str, ...] = ()
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""

    @property
    def elapsed_s(self) -> float:
        return max(0.0, (self.finished_at or time.time()) - self.started_at)

    @property
    def ok(self) -> bool:
        return self.state == "done"


def run(units: Sequence[Unit], peers: Sequence[Peer], *, kind: str = "",
        score: Score | None = None, retries: int = 1, poll_s: float = 2.0,
        rates: Rates | None = None,
        held: Callable[[Peer, Unit], float] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        ) -> list[Placement]:
    """Run every unit on some peer, and report where each one went."""
    if not peers:
        raise ValueError("no peers to run on")

    rates = rates if rates is not None else Rates()
    results: dict[str, Placement] = {u.id: Placement(unit_id=u.id) for u in units}
    by_id = {u.id: u for u in units}
    pending = [u.id for u in units]
    strikes: dict[str, int] = {}
    cooldown: dict[str, float] = {}
    lock = threading.Lock()
    stop = threading.Event()
    live: dict[str, tuple[Peer, str]] = {}

    def emit(event: str, **fields: Any) -> None:
        if on_event is not None:
            try:
                on_event(event, fields)
            except Exception:                         # noqa: BLE001
                pass

    def take(name: str) -> str | None:
        """The next unit this peer may run, or None."""
        with lock:
            for uid in list(pending):
                unit = by_id[uid]
                if unit.peer and unit.peer != name:
                    continue
                if name not in admits.get(uid, ()):
                    continue
                if name in results[uid].tried and len(results[uid].tried) <= retries:
                    if any(o != name for o in _admitting(uid)):
                        continue
                pending.remove(uid)
                return uid
            return None

    def _admitting(uid: str) -> list[str]:
        return list(admits.get(uid, ()))

    admits: dict[str, tuple[str, ...]] = {}

    def worker(peer: Peer, name: str) -> None:
        while not stop.is_set():
            held_until = cooldown.get(name, 0.0)
            if held_until > time.time():
                if all(results[u].state != "pending" for u in by_id):
                    return
                time.sleep(min(1.0, held_until - time.time()))
                continue
            uid = take(name)
            if uid is None:
                if all(results[u].state != "pending" for u in by_id):
                    return
                time.sleep(0.1)
                continue
            unit = by_id[uid]
            place = results[uid]
            place.attempts += 1
            place.tried = tuple(dict.fromkeys(place.tried + (name,)))
            place.peer, place.base_url = name, peer.base_url
            place.started_at = place.started_at or time.time()
            emit("start", unit=uid, peer=name, attempt=place.attempts)
            try:
                job = peer.submit(list(unit.argv), name=unit.name or uid,
                                  cwd=unit.cwd, env=dict(unit.env))
                place.job_id = job["id"]
                with lock:
                    live[uid] = (peer, job["id"])
                final = peer.wait(job["id"], poll_s=poll_s)
                place.returncode = final.get("returncode")
                place.state = "done" if final.get("state") == "done" else "failed"
                place.error = "" if place.ok else f"exit {place.returncode}"
            except Exception as exc:                  # noqa: BLE001
                place.state = "unreachable"
                place.error = str(exc)
            finally:
                with lock:
                    live.pop(uid, None)
            place.finished_at = time.time()

            if place.ok:
                strikes[name] = 0
                rate = rates.record(name, kind, units=unit.work,
                                    seconds=place.elapsed_s)
                emit("done", unit=uid, peer=name, seconds=place.elapsed_s, rate=rate)
                continue

            strikes[name] = strikes.get(name, 0) + 1
            emit("fail", unit=uid, peer=name, state=place.state, error=place.error)
            if strikes[name] >= QUARANTINE_AFTER:
                wait = min(300.0, 30.0 * 2 ** (strikes[name] - QUARANTINE_AFTER))
                cooldown[name] = time.time() + wait
                emit("quarantine", peer=name, seconds=wait, strikes=strikes[name])

            if len(place.tried) <= retries and len(_admitting(uid)) > len(place.tried):
                place.state = "pending"
                with lock:
                    pending.append(uid)
            elif place.state == "pending":
                place.state = "failed"

    bench = {peer: score for (peer, k), score in rates.as_map().items()
             if k == BENCH_KIND}
    snapshot = candidates(peers, kind=kind, rates=rates.as_map(), bench=bench)
    for unit in units:
        kept, refused = eligible(list(snapshot), unit.requires)
        if unit.peer:
            kept = [c for c in kept if c.name == unit.peer]
        admits[unit.id] = tuple(c.name for c in kept)
        if not kept:
            place = results[unit.id]
            place.state = "unplaceable"
            place.error = "; ".join(f"{n}: {r}" for n, r in sorted(refused.items())) \
                or "no peer answered"
            place.finished_at = time.time()
            with lock:
                if unit.id in pending:
                    pending.remove(unit.id)
            emit("unplaceable", unit=unit.id, reasons=refused)

    threads = [threading.Thread(target=worker, args=(c.peer, c.name), daemon=True,
                                name=f"fanout-{c.name}-{i}")
               for c in snapshot for i in range(max(1, c.slots))]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except BaseException:
        stop.set()
        with lock:
            outstanding = list(live.items())
        for uid, (peer, job_id) in outstanding:
            try:
                peer.stop(job_id)
                results[uid].state = "stopped"
            except Exception:                         # noqa: BLE001
                pass
        raise
    finally:
        stop.set()
        try:
            rates.save()
        except OSError:
            pass

    for place in results.values():
        if place.state == "pending":
            place.state = "failed"
            place.error = place.error or "never ran"
    return [results[u.id] for u in units]
