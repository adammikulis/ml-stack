"""Filtering records, and the relations between them built at query time."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from poolhouse.activity.schema import Entry

__all__ = ["Filter", "duration", "graph", "select", "touching"]

_SPAN = re.compile(r"(\d+(?:\.\d+)?)\s*([smhdw])")
_UNIT = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def duration(text: str) -> float:
    """Seconds in ``2h``, ``90m``, ``1d12h``; ``ValueError`` for anything else."""
    parts = _SPAN.findall(text.strip().lower())
    if not parts or _SPAN.sub("", text.strip().lower()).strip():
        raise ValueError(f"not a span of time: {text!r} (try 30m, 2h, 1d)")
    return sum(float(n) * _UNIT[u] for n, u in parts)


@dataclass(frozen=True, slots=True)
class Filter:
    """Which records to show; every field left empty matches everything."""

    since: float = 0.0
    until: float = 0.0
    agent: str = ""
    kind: str = ""
    session: str = ""
    grep: str = ""

    def keeps(self, e: Entry) -> bool:
        return (e.ts >= self.since and (not self.until or e.ts <= self.until)
                and _agent(e, self.agent) and _kind(e.kind, self.kind)
                and (not self.session or e.session.startswith(self.session))
                and (not self.grep or self.grep.lower() in flat(e).lower()))


def _agent(e: Entry, name: str) -> bool:
    if not name:
        return True
    named = [e.actor, *(str(e.refs.get(k, "")) for k in ("agent", "from", "to"))]
    return name in {*named, *(n.partition(":")[2] for n in named)}


def _kind(kind: str, wanted: str) -> bool:
    if not wanted:
        return True
    return kind == wanted or kind.startswith(wanted.rstrip("*").rstrip(".") + ".")


def flat(e: Entry) -> str:
    """Everything searchable in a record as one line of text."""
    parts = [e.kind, e.actor, e.session, e.subject, e.outcome]
    parts += [f"{k}={v}" for k, v in {**e.refs, **e.meta}.items()]
    return " ".join(parts)


def select(entries: Iterable[Entry], flt: Filter) -> list[Entry]:
    """The records ``flt`` keeps, in the order they were written."""
    return [e for e in entries if flt.keeps(e)]


def graph(entries: Iterable[Entry]) -> dict[str, Any]:
    """The records as a graph mapping (``nodes`` and ``edges``, see docs/graph.md): an
    ``event`` node for each, tied to its actor, session, subject and every reference."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def node(kind: str, key: str) -> str:
        ident = f"{kind}:{key}"
        nodes.setdefault(ident, {"id": ident, "kind": kind, "label": key})
        return ident

    for e in entries:
        here = node("event", e.id)
        nodes[here]["attrs"] = {"kind": e.kind, "ts": e.ts, "outcome": e.outcome}
        edges.append({"source": node("actor", e.actor), "rel": "did", "target": here})
        edges.append({"source": here, "rel": "in", "target": node("session", e.session)})
        if e.subject:
            edges.append({"source": here, "rel": "on", "target": node("subject", e.subject)})
        for name, value in sorted(e.refs.items()):
            edges.append({"source": here, "rel": name, "target": node("subject", str(value))})
    return {"nodes": list(nodes.values()), "edges": edges}


def touching(entries: Iterable[Entry], agent: str, target: str) -> list[Entry]:
    """The records ``agent`` is party to that name ``target`` as subject or reference, found
    by walking the graph from both ends."""
    held = list(entries)
    links = graph(held)
    around: dict[str, set[str]] = defaultdict(set)
    for edge in links["edges"]:
        around[edge["source"]].add(edge["target"])
        around[edge["target"]].add(edge["source"])
    by_actor = {n for n in around if n.partition(":")[0] in ("actor", "subject")
                and _named(n.partition(":")[2], agent)}
    by_target = {n for n in around if n.startswith("subject:") and target in n[8:]}
    mine = set().union(*(around[a] for a in by_actor)) if by_actor else set()
    theirs = set().union(*(around[t] for t in by_target)) if by_target else set()
    both = {n for n in mine & theirs if n.startswith("event:")}
    return [e for e in held if f"event:{e.id}" in both]


def _named(actor: str, wanted: str) -> bool:
    return wanted in (actor, actor.partition(":")[2])


def midnight(now: float | None = None) -> float:
    """The start of the local day containing ``now``."""
    stamp = time.localtime(time.time() if now is None else now)
    return time.mktime((stamp.tm_year, stamp.tm_mon, stamp.tm_mday, 0, 0, 0, 0, 0, -1))
