"""Questions about an invented world whose answers are known, in the bench's format.

`ml_stack.graph.community.QUESTIONS` was written by hand against a graph small enough to
check by eye. A world of five thousand people cannot be, so its questions are generated from
the truth that made it: who reports to whom, who works on what, who is where. Each is
``{"q": ..., "expect": [ids]}``, the shape `ml_stack.graph.bench` reads, and the set is
spread across the kinds of answer the bench's own set covers -- people mostly, but
organisations, places, subjects, work, events, paths between two people, and a few whose
right answer is nobody -- so a model measured on it is measured on the same things.

Nothing here assumes a company. Every generator reads a relation, and a kind that lacks
that relation (a community has no ``reports_to``) simply contributes no such questions.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.world import World

__all__ = ["questions"]

# a relation into a unit or a person, and how to ask who is on the other end of it
_WHO_BY = {
    "reports_to": ("Who reports to {}?", "in", 2, 9),
    "advises": ("Whom does {} advise?", "out", 2, 12),
    "maintains": ("Who maintains {}?", "in", 1, 12),
    "moderates": ("Who moderates {}?", "in", 1, 6),
    "leads": ("Who leads {}?", "in", 1, 3),
    "chairs": ("Who chairs {}?", "in", 1, 2),
    "sits_on": ("Who sits on {}?", "in", 2, 12),
    "mentors": ("Whom does {} mentor?", "out", 1, 4),
}
# a relation into an organisation, and how to ask which organisations are on the other end
_ORGS_BY = {
    "customer_of": "Which companies are customers of {}?",
    "partner_of": "Which organisations partner with {}?",
    "funds": "Who funds {}?",
    "sponsors": "Which companies sponsor {}?",
}
# the same, asked in the other direction, when the bench would list them instead
_KINDS_LISTED = {
    "event": "What events come up?",
    "product": "What products are there?",
}
_NOBODY = (
    "Nobody here does underwater welding. Who could?",
    "Who is the court jester?",
    "hi",
    "Which person here is a licensed astronaut?",
)


class _Truth:
    """The graph indexed both ways by relation, and labels by id."""

    def __init__(self, graph: Mapping[str, Any]) -> None:
        self.nodes = {str(n["id"]): n for n in graph.get("nodes") or ()}
        self.out: dict[str, dict[str, list[str]]] = {}
        self.inc: dict[str, dict[str, list[str]]] = {}
        for e in graph.get("edges") or ():
            rel, s, t = str(e.get("rel") or ""), str(e["source"]), str(e["target"])
            self.out.setdefault(rel, {}).setdefault(s, []).append(t)
            self.inc.setdefault(rel, {}).setdefault(t, []).append(s)
        self.edges = list(graph.get("edges") or ())
        meta = (graph.get("meta") or {}).get("world") or {}
        self.unit_kind = str(meta.get("unit_kind") or "department")
        self.org = str(meta.get("organisation") or "")

    def label(self, node_id: str) -> str:
        return str(self.nodes.get(node_id, {}).get("label") or node_id)

    def kind(self, node_id: str) -> str:
        return str(self.nodes.get(node_id, {}).get("kind") or "")

    def of_kind(self, kind: str) -> list[str]:
        return [i for i, n in self.nodes.items() if n.get("kind") == kind]

    def people_in(self, unit: str) -> list[str]:
        return [p for p in self.inc.get("part_of", {}).get(unit, ()) if self.kind(p) == "person"]


def _q(text: str, expect: Sequence[str]) -> dict[str, Any]:
    return {"q": text, "expect": sorted(dict.fromkeys(expect))}


def _people_at(t: _Truth, rel: str, target: str, units: Sequence[str],
               phrase: str, narrowed: str, low: int = 1, high: int = 10) -> list[dict[str, Any]]:
    """"Who is based in C?" when that is few enough to list; else narrowed to one unit."""
    people = [p for p in t.inc.get(rel, {}).get(target, ()) if t.kind(p) == "person"]
    if low <= len(people) <= high:
        return [_q(phrase.format(t.label(target)), people)]
    out = []
    for unit in units:
        both = [p for p in people if p in set(t.people_in(unit))]
        if low <= len(both) <= high:
            out.append(_q(narrowed.format(t.label(unit), t.label(target)), both))
    return out


def _buckets(t: _Truth, rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    """Every question the truth supports, grouped by the kind of answer it wants."""
    b: dict[str, list[dict[str, Any]]] = {}
    people = t.of_kind("person")
    units = t.of_kind(t.unit_kind)

    # --- people, by what joins them to somebody or something --------------------------
    for rel, (phrase, side, low, high) in _WHO_BY.items():
        table = t.inc.get(rel, {}) if side == "in" else t.out.get(rel, {})
        for anchor, others in table.items():
            others = [o for o in others if t.kind(o) == "person"]
            if low <= len(others) <= high:
                b.setdefault("person", []).append(_q(phrase.format(t.label(anchor)), others))
    for project, crowd in t.inc.get("works_on", {}).items():
        if 2 <= len(crowd) <= 12:
            b.setdefault("person", []).append(_q(f"Who works on {t.label(project)}?", crowd))
    for place in t.of_kind("place"):
        b.setdefault("person", []).extend(_people_at(
            t, "based_in", place, units, "Who is based in {}?", "Who in {} is based in {}?"))
    for topic in t.of_kind("topic"):
        b.setdefault("person", []).extend(_people_at(
            t, "experienced_in", topic, units, "Who knows about {}?", "Who in {} knows about {}?"))
    for unit in units:
        crowd = t.people_in(unit)
        if 2 <= len(crowd) <= 12:
            b.setdefault("person", []).append(_q(f"Who is in {t.label(unit)}?", crowd))
    for opp, wants in t.out.get("wants", {}).items():
        able = [p for w in wants for p in t.inc.get("experienced_in", {}).get(w, ())]
        able = [p for p in able if t.kind(p) == "person"]
        if 1 <= len(able) <= 8:
            b.setdefault("person", []).append(_q(f"Who could take on {t.label(opp)}?", able))

    # --- organisations -------------------------------------------------------------------
    for rel, phrase in _ORGS_BY.items():
        for target, orgs in t.inc.get(rel, {}).items():
            orgs = [o for o in orgs if t.kind(o) == "org"]
            if 1 <= len(orgs) <= 12:
                b.setdefault("org", []).append(_q(phrase.format(t.label(target)), orgs))
    for who, orgs in t.out.get("works_at", {}).items():
        if t.kind(who) == "person" and orgs and orgs[0] != t.org:
            b.setdefault("org", []).append(_q(f"Where does {t.label(who)} work?", orgs))
    everyone_at = t.inc.get("works_at", {}).get(t.org, ())
    if t.org and not everyone_at:
        # the kinds where people work elsewhere: list the employers, if that is few enough
        employers = sorted({o for who in people for o in t.out.get("works_at", {}).get(who, ())})
        if 2 <= len(employers) <= 12:
            b.setdefault("org", []).append(_q("Which companies do people here work for?", employers))

    # --- places ----------------------------------------------------------------------------
    for who in people:
        places = t.out.get("based_in", {}).get(who, ())
        if places:
            b.setdefault("place", []).append(_q(f"Where is {t.label(who)} based?", places))
    for org, places in t.out.get("based_in", {}).items():
        if t.kind(org) == "org" and places:
            b.setdefault("place", []).append(_q(f"Which city is {t.label(org)} in?", places))
    for unit in units:
        crowd = t.people_in(unit)
        cities = sorted({c for p in crowd for c in t.out.get("based_in", {}).get(p, ())})
        if crowd and 1 <= len(cities) <= 6:
            b.setdefault("place", []).append(
                _q(f"Which places are the people in {t.label(unit)} based in?", cities))

    # --- subjects ---------------------------------------------------------------------------
    for who in people:
        topics = t.out.get("experienced_in", {}).get(who, ())
        if topics:
            b.setdefault("topic", []).append(_q(f"What is {t.label(who)} good at?", [who, *topics]))
    for opp, wants in t.out.get("wants", {}).items():
        b.setdefault("topic", []).append(_q(f"What does {t.label(opp)} need?", wants))
    for unit in units:
        crowd = t.people_in(unit)
        counted: dict[str, int] = {}
        for p in crowd:
            for topic in t.out.get("experienced_in", {}).get(p, ()):
                counted[topic] = counted.get(topic, 0) + 1
        top = sorted(counted, key=lambda k: (-counted[k], k))[:4]
        if len(crowd) >= 3 and top and counted[top[0]] >= 2:
            b.setdefault("topic", []).append(
                _q(f"What does {t.label(unit)} know most about?", top[:2]))

    # --- units: what a department, lab, group, repo or programme is ----------------------------
    for unit in units:
        b.setdefault("unit", []).append(_q(f"What does {t.label(unit)} do?", [unit]))
        for who in t.people_in(unit)[:2]:
            b.setdefault("unit", []).append(
                _q(f"Which {t.unit_kind} is {t.label(who)} in?", [unit]))
    if 2 <= len(units) <= 12:
        b.setdefault("unit", []).append(_q(f"Which {t.unit_kind}s are there?", units))

    # --- work going spare, and where people meet ----------------------------------------------
    opportunities = t.of_kind("opportunity")
    if 1 <= len(opportunities) <= 12:
        b.setdefault("opportunity", []).append(_q("What openings and projects are there?", opportunities))
    for owner, offered in t.out.get("offers", {}).items():
        if 1 <= len(offered) <= 12:
            b.setdefault("opportunity", []).append(
                _q(f"What is {t.label(owner)} running or offering?", offered))
    for topic, wanted_by in t.inc.get("wants", {}).items():
        if 1 <= len(wanted_by) <= 8:
            b.setdefault("opportunity", []).append(
                _q(f"Which projects or openings want {t.label(topic)}?", wanted_by))
    for who in people:
        mine = t.out.get("works_on", {}).get(who, ())
        if mine:
            b.setdefault("opportunity", []).append(_q(f"What is {t.label(who)} working on?", mine))
    events = t.of_kind("event")
    if 1 <= len(events) <= 12:
        b.setdefault("event", []).append(_q(_KINDS_LISTED["event"], events))
    for event in events:
        crowd = [p for p in t.inc.get("attended", {}).get(event, ()) if t.kind(p) == "person"]
        if 2 <= len(crowd) <= 12:
            b.setdefault("person", []).append(_q(f"Who was at {t.label(event)}?", crowd))
        for who in crowd[:1]:
            went = t.out.get("attended", {}).get(who, ())
            if 1 <= len(went) <= 6:
                b.setdefault("event", []).append(_q(f"Which events did {t.label(who)} go to?", went))
    products = t.of_kind("product")
    if products and t.org:
        b.setdefault("product", []).append(_q(f"What does {t.label(t.org)} make?", products))

    # --- the question names part of its own answer ----------------------------------------------
    for who in people:
        unit = [u for u in t.out.get("part_of", {}).get(who, ()) if t.kind(u) == t.unit_kind][:1]
        place = t.out.get("based_in", {}).get(who, ())[:1]
        b.setdefault("about", []).append(_q(f"Tell me about {t.label(who)}.", [who, *unit, *place]))

    # --- how two people connect, through something more specific than the whole organisation --
    from ml_stack.entities.paths import between

    narrow = [e for e in t.edges if e["target"] != t.org and e["source"] != t.org
              and t.kind(e["target"]) != t.unit_kind]
    pairs = rng.sample(people, min(len(people), 24))
    for a, c in zip(pairs[::2], pairs[1::2]):
        path = between(narrow, a, c)
        if 3 <= len(path) <= 6:
            b.setdefault("path", []).append(_q(f"How is {t.label(a)} connected to {t.label(c)}?", path))

    # --- the right answer is nobody ------------------------------------------------------------
    b["nobody"] = [_q(text, []) for text in _NOBODY]
    if "reports_to" in t.inc:
        leaves = [p for p in people if p not in t.inc["reports_to"]]
        for who in rng.sample(leaves, min(2, len(leaves))):
            b["nobody"].append(_q(f"Who reports to {t.label(who)}?", []))

    for bucket in b.values():
        rng.shuffle(bucket)
    return b


# how many of each bucket a cycle takes: people most, as the bench's own set has it
_ORDER = (("person", 3), ("org", 1), ("place", 1), ("topic", 1), ("unit", 1), ("path", 1),
          ("event", 1), ("opportunity", 1), ("product", 1), ("about", 1), ("nobody", 1))


def questions(world: World, n: int = 40, rng: random.Random | None = None) -> list[dict[str, Any]]:
    """``n`` questions about the world, each with the ids a good answer names.

    Spread over every kind of answer the world supports, rarest kinds first so a small
    ``n`` still asks about each; deterministic from ``world.seed`` unless an ``rng`` is
    given. Each is ``{"q", "expect"}`` and nothing else, so the bench reads it as it reads
    its own.
    """
    rng = rng or random.Random(f"questions/{world.seed}/{world.size}")
    buckets = _buckets(_Truth(world.graph), rng)
    taken: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(taken) < n and any(buckets.values()):
        for name, share in _ORDER:
            for _ in range(share):
                while buckets.get(name):
                    one = buckets[name].pop()
                    if one["q"] not in seen:
                        seen.add(one["q"])
                        taken.append(one)
                        break
                if len(taken) >= n:
                    return taken
    return taken
