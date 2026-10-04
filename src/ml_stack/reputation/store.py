"""The reputation ledger: sources as nodes of a sealed per-user graph, each bad event a node
tied to its source, scores kept in the source's attributes."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import activity, files
from ml_stack.graph.columns import column
from ml_stack.graph.store import GraphStore
from ml_stack.memory import vault
from ml_stack.reputation import model
from ml_stack.reputation.model import Rec, State
from ml_stack.reputation.sealed import SCHEMA_VERSION, SealedGraph
from ml_stack.sentinel.observers import Gate

__all__ = ["MAX_EVENTS", "MAX_SOURCES", "RECOVER_ENV", "Ledger", "Standing"]

logger = logging.getLogger("ml_stack.reputation")
MAX_SOURCES, MAX_EVENTS = 1000, 20
"""Sources kept in all, and bad-event nodes kept for each."""
RECOVER_ENV = "ML_STACK_REPUTATION_RECOVER"
FLUSH_S = 60.0
Step = tuple[str, str, Callable[[Rec, GraphStore], Any]]
_RANK = {State.UNKNOWN: 0, State.ESTABLISHED: 0, State.WATCH: 1, State.BAD: 2}


@dataclass(frozen=True, slots=True)
class Standing:
    """A source as it is now: its state, the two scores and what has been seen."""

    kind: str
    key: str
    state: str
    since: float
    short: float
    long: float
    clean: int
    streak: int
    notice: str
    first: float
    last: float
    traits: dict[str, list[str]]
    events: list[list[Any]]


def node_id(kind: str, key: str) -> str:
    return f"src:{kind}:{key}"


def recover_after() -> int:
    """Clean runs in a row that let a watched source back, from ``$ML_STACK_REPUTATION_RECOVER``."""
    try:
        return max(1, int(os.environ.get(RECOVER_ENV, model.RECOVER_CLEAN)))
    except ValueError:
        return model.RECOVER_CLEAN


class Ledger:
    """What is known about every source, encrypted at rest. ``on_notice`` is called, outside
    the store, after a source diverges. ``flush_s`` is how long clean runs wait to be written."""

    def __init__(self, path: Path | None = None, *, clock: Callable[[], float] = time.time,
                 keys: vault.Keys | None = None,
                 on_notice: Callable[[], None] | None = None, flush_s: float = FLUSH_S) -> None:
        self.sealed = SealedGraph(path, keys=keys)
        self.clock, self.on_notice, self.flush_s = clock, on_notice, flush_s
        self._pending: dict[tuple[str, str], float] = {}
        self._flushed = clock()

    @property
    def path(self) -> Path:
        return self.sealed.path

    def close(self) -> None:
        """Write waiting clean runs and release the in-memory graph."""
        self.flush()
        self.sealed.close()

    # -- reading ----------------------------------------------------------------------
    @staticmethod
    def _load(g: GraphStore, kind: str, key: str) -> Rec | None:
        rows = g.query("MATCH (n:Node {id:$id}) RETURN n.attrs AS attrs", {"id": node_id(kind, key)})
        return Rec.from_attrs(column(rows[0]["attrs"], "source")) if rows else None

    def _standing(self, rec: Rec, now: float) -> Standing:
        return Standing(rec.kind, rec.key, rec.state, rec.since, model.short_score(rec, now),
                        model.long_score(rec, now), rec.clean, rec.streak, rec.notice, rec.first,
                        rec.last, rec.traits, rec.recent)

    def standing(self, kind: str, key: str) -> Standing | None:
        """The source ``key`` of ``kind``, or None when it has never been seen."""
        g = self.sealed.graph()
        rec = self._load(g, kind, model.canonical(kind, key)) if g is not None else None
        return self._standing(rec, self.clock()) if rec else None

    def sources(self) -> list[Standing]:
        """Every source held, most recently seen first; empty when the store is locked."""
        g, now = self.sealed.graph(), self.clock()
        if g is None:
            return []
        rows = [Rec.from_attrs(n["attrs"]) for n in g.nodes("source")]
        return [self._standing(r, now) for r in sorted(rows, key=lambda r: (-r.last, r.kind, r.key))]

    def gate(self, kind: str, key: str) -> Gate | None:
        """What to say about a watched or bad source, else None. Never raises."""
        try:
            held = self.standing(kind, key)
        except (ValueError, vault.KeyUnavailable, RuntimeError):
            return None
        if held is None or held.state not in (State.WATCH, State.BAD):
            return None
        why = ("it has misbehaved before and no clean record since"
               if held.state == State.BAD else "it was reliable and has just changed or misbehaved")
        return Gate(held.state, f"{kind} {key} is {held.state}: {why}", held.since)

    # -- writing ----------------------------------------------------------------------
    def _write(self, rec: Rec, g: GraphStore) -> None:
        g.upsert_node({"id": node_id(rec.kind, rec.key), "kind": "source", "label": rec.key,
                       "mentions": rec.clean, "attrs": rec.attrs()})

    def _event_node(self, g: GraphStore, rec: Rec, event: str, now: float, weight: float) -> None:
        self._write(rec, g)
        ident = f"ev:{rec.kind}:{rec.key}:{int(now * 1000)}:{event}"
        g.upsert_node({"id": ident, "kind": "event", "label": event,
                       "attrs": {"ts": now, "weight": weight}})
        g.upsert_edge({"source": node_id(rec.kind, rec.key), "target": ident, "rel": "event"})
        rec.events = [*rec.events, ident][-MAX_EVENTS:]
        g.drop([e["id"] for e in g.neighbours(node_id(rec.kind, rec.key))
                if e["id"].startswith("ev:") and e["id"] not in rec.events], force=True)

    def _evict(self, g: GraphStore) -> None:
        rows = [Rec.from_attrs(n["attrs"]) for n in g.nodes("source")]
        for rec in sorted(rows, key=lambda r: (_RANK[State(r.state)], r.last))[:max(0, len(rows) - MAX_SOURCES)]:
            self._drop(g, rec.kind, rec.key)

    @staticmethod
    def _drop(g: GraphStore, kind: str, key: str) -> bool:
        here = node_id(kind, key)
        if not g.has(here):
            return False
        g.drop([here, *(e["id"] for e in g.neighbours(here) if e["id"].startswith("ev:"))], force=True)
        return True

    def _summary(self, g: GraphStore) -> None:
        held = [Rec.from_attrs(n["attrs"]) for n in g.nodes("source")]
        counts = {s.value: sum(r.state == s for r in held) for s in State}
        counts["notices"] = sum(r.notice == "queued" for r in held)
        files.write_json(self.path.with_name("summary.json"), files.versioned(counts, SCHEMA_VERSION),
                         indent=None)

    def _edit(self, steps: list[Step]) -> list[Any]:
        """Apply each ``(kind, key, change)`` to its source (made when new) in one sealed write."""
        diverged = False

        def run(g: GraphStore) -> list[Any]:
            nonlocal diverged
            out = []
            for kind, key, change in steps:
                rec = self._load(g, kind, key) or Rec.new(kind, key, self.clock())
                out.append(change(rec, g))
                diverged = diverged or out[-1] is True
                self._write(rec, g)
            self._evict(g)
            self._summary(g)
            return out

        done = self.sealed.edit(run)
        if diverged and self.on_notice is not None:
            try:
                self.on_notice()
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("reputation notice failed: %s", type(exc).__name__)
        return done

    def _pending_steps(self) -> list[Step]:
        taken, self._pending = self._pending, {}
        recover = recover_after()
        return [(k, v, lambda r, _g, at=at: model.clean_run(r, at, recover))
                for (k, v), at in taken.items()]

    def flush(self) -> None:
        """Write the clean runs that are waiting."""
        if self._pending:
            self._edit(self._pending_steps())
        self._flushed = self.clock()

    def clean(self, kind: str, key: str) -> None:
        """Note one clean interaction; written with the next flush."""
        self._pending[(kind, model.canonical(kind, key))] = self.clock()
        if self.clock() - self._flushed >= self.flush_s:
            self.flush()

    def _bad(self, rec: Rec, g: GraphStore, event: str, scale: float) -> bool:
        now = self.clock()
        diverged = model.misbehaved(rec, event, now, scale)
        if rec.recent and rec.recent[-1][0] == now and rec.recent[-1][1] == event:
            self._event_node(g, rec, event, now, float(rec.recent[-1][2]))
        return diverged

    def observe(self, kind: str, key: str, event: str, *, scale: float = 1.0) -> Standing:
        """Note one bad ``event`` that the system itself saw, and return where the source stands."""
        name = model.canonical(kind, key)
        self._edit([*self._pending_steps(),
                    (kind, name, lambda r, g: self._bad(r, g, event, scale))])
        held = self.standing(kind, name)
        if held is None:
            raise RuntimeError("the source was not written")
        activity.record("reputation.observed", actor="system", subject=f"{kind}:{name}",
                        outcome=held.state, meta={"event": event, "scale": scale,
                                                   "short": round(held.short, 2)})
        return held

    def trait(self, kind: str, key: str, name: str, value: str) -> None:
        """Record one trait of a source (``cert``, ``ip_range``, ``redirect``, ``hash``, ``shape``).
        A value outside the baseline already held is an event for the traits that have one."""
        def note(rec: Rec, g: GraphStore) -> bool:
            event = model.observe_trait(rec, name, value)
            return self._bad(rec, g, event, 1.0) if event else False

        self._edit([*self._pending_steps(), (kind, model.canonical(kind, key), note)])

    def block(self, kind: str, key: str) -> None:
        """Put a source in bad now (the dialog's Block button)."""
        def to_bad(rec: Rec, _g: GraphStore) -> None:
            rec.state, rec.since, rec.notice, rec.streak = State.BAD, self.clock(), "done", 0

        self._edit([(kind, model.canonical(kind, key), to_bad)])

    def settle_notice(self, kind: str, key: str, notice: str) -> None:
        """Set where a source's notice stands (``queued``, ``shown`` or ``done``)."""
        def mark(rec: Rec, _g: GraphStore) -> None:
            rec.notice = notice

        self._edit([(kind, model.canonical(kind, key), mark)])

    def queued(self) -> list[Standing]:
        """The sources with a divergence notice waiting, oldest first."""
        return sorted((s for s in self.sources() if s.notice == "queued"), key=lambda s: s.since)

    # -- the person's ------------------------------------------------------------------
    def forget(self, kind: str, key: str) -> bool:
        """Delete one source and the events tied to it; true when it was held."""
        name = model.canonical(kind, key)
        if self.standing(kind, name) is None:
            return False

        def run(g: GraphStore) -> bool:
            gone = self._drop(g, kind, name)
            self._summary(g)
            return gone

        return bool(self.sealed.edit(run))

    def forget_all(self) -> int:
        """Delete the whole store and its summary; returns how many sources were held."""
        held = len(self.sources())
        self._pending = {}
        self.sealed.delete()
        self.path.with_name("summary.json").unlink(missing_ok=True)
        return held

    def export(self) -> dict[str, Any]:
        """Every source with its traits and recent events, as plain data."""
        return {"schema_version": SCHEMA_VERSION,
                "sources": [{"kind": s.kind, "key": s.key, "state": s.state, "clean": s.clean,
                             "first": s.first, "last": s.last, "short": round(s.short, 3),
                             "long": round(s.long, 3), "traits": s.traits, "events": s.events}
                            for s in self.sources()]}

    def stats(self) -> dict[str, Any]:
        """Counts by state, limits and integrity."""
        held = self.sources()
        return {"status": self.sealed.status, "path": str(self.path), "sources": len(held),
                "limit": MAX_SOURCES, "by_state": {s.value: sum(h.state == s for h in held) for s in State},
                "notices": sum(h.notice == "queued" for h in held), "encrypted": True}
