"""Spread a list of jobs over the peers that can run them.

The daemon serialises jobs onto a device; this serialises units onto a peer. Same shape,
one level up: one worker thread per peer, all pulling from a shared queue, each peer
holding at most as many units as it has free slots.

The failures this is shaped around, none of which are hypothetical on a home LAN:

**A box whose card has fallen over accepts work and fails it in milliseconds.** Left
alone it will drain the entire queue faster than the healthy boxes can take anything --
every unit fails, on the fastest machine available, which is the one that is broken. So
a peer that fails several units in a row is quarantined for a growing cooldown, and
reported rather than silently dropped.

**A unit that is simply wrong looks exactly like a peer that is broken.** Distinguished
by counting *distinct* peers: a unit that has failed on several different machines is the
unit's fault, and retrying it forever just moves the failure around.

**Nothing eligible is not the same as nothing free.** A unit no peer can run must fail
immediately, carrying every peer's reason, instead of waiting for capacity that would
not help if it arrived.

**Ctrl-C must not leave six machines training.** Everything outstanding is stopped on the
way out -- SIGTERM, which a loop that checkpoints on TERM survives without losing work.
"""

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
"""Consecutive failures before a peer is set aside. Three, not one: a peer that fails
one unit is probably looking at a bad unit, and taking a good box out of the fleet for
that is the more expensive mistake."""


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
    """Run every unit on some peer, and report where each one went.

    ``retries`` is how many *additional distinct peers* a failing unit may be tried on.
    """
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
                # Eligibility is decided once up front, but it has to be enforced HERE
                # too: without this a worker happily takes any pending unit, and the
                # labels that were supposed to keep prep off the training boxes decide
                # nothing at all.
                if name not in admits.get(uid, ()):
                    continue
                if name in results[uid].tried and len(results[uid].tried) <= retries:
                    # Prefer a peer that has not already failed this unit, but do not
                    # strand the unit if this is the only peer that admits it.
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
                # Quarantined. Sleep rather than exit: the other peers may finish
                # everything meanwhile, and if they do the loop below notices.
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
                # Growing, because a box that has failed six things in a row is not
                # going to be fine in another thirty seconds.
                wait = min(300.0, 30.0 * 2 ** (strikes[name] - QUARANTINE_AFTER))
                cooldown[name] = time.time() + wait
                emit("quarantine", peer=name, seconds=wait, strikes=strikes[name])

            if len(place.tried) <= retries and len(_admitting(uid)) > len(place.tried):
                place.state = "pending"
                with lock:
                    pending.append(uid)
            elif place.state == "pending":
                place.state = "failed"

    # Which peers admit which units, decided once against a live snapshot. This is also
    # what makes "no peer can run this" an immediate answer rather than a wait.
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

    # One worker per SLOT, not per peer: a box told it has eight is a box that should
    # be running eight. One thread each would make --slots do nothing from here, which
    # is the sort of gap that looks like the daemon ignoring its own flag.
    threads = [threading.Thread(target=worker, args=(c.peer, c.name), daemon=True,
                                name=f"fanout-{c.name}-{i}")
               for c in snapshot for i in range(max(1, c.slots))]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except BaseException:
        # Ctrl-C, or anything else. Stop what is running before unwinding: a fan-out
        # that leaves six boxes training is worse than one that never started.
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
