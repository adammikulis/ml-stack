"""Questions about an invented world whose answers are known, in the bench's format.

`ml_stack.graph.community.QUESTIONS` was written by hand against a graph small enough to
check by eye. A world of five thousand people cannot be, so its questions are generated from
the truth that made it: who reports to whom, who works on what, who is where. Each is
``{"q": ..., "expect": [ids], "kind": ...}``, the shape `ml_stack.graph.bench` reads (it
ignores the tag), and the set is spread across the kinds of answer the bench's own set
covers -- people mostly, but organisations, places, subjects, work, events, paths between
two people, and a few whose right answer is nobody -- so a model measured on it is measured
on the same things. Four kinds are about the *question* rather than the answer, because the
bench's own set was short of them: ``aggregate`` (a count, scored as the people counted, or
the unit or employer with the most people), ``twohop`` (the people who work with whoever
knows a subject, or the units they sit in -- the far end, never the middle), ``trap`` (a
false premise about a real person; the right answer is the place as the graph has it) and
``quote`` (answerable from a person's own ``messages`` and nothing else). `KINDS` lists
every tag, and ``kinds=`` draws only some.

The relations the conversations state outright -- who reports to whom, who works with
whom, who works on what, who belongs to which unit, which `world.simulate` writes into the
messages and carries as gold -- are asked back as ``person`` questions ("Who does X report
to?") and as ``path`` questions between two people the same edge joins, so what a reader
of the corpus could have learnt is what it is asked.

Nothing here assumes a company. Every generator reads a relation, and a kind that lacks
that relation (a community has no ``reports_to``) simply contributes no such questions.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.world import World

__all__ = ["KINDS", "questions"]

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
# a relation a message states about one person, and how to ask for its far end back
_STATED_ASK = {
    "reports_to": "Who does {} report to?",
    "works_with": "Who works with {}?",
}
# relations whose shared far end joins two people: the path between them runs through it
_STATED_JOIN = ("part_of", "works_on", "reports_to", "works_with", "advises", "maintains")
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
        self.messages: Mapping[str, Any] = graph.get("messages") or {}
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

    # --- the relations the conversations state outright -------------------------------------
    # `world.simulate` writes "A reports to B" and "A is part of C" into the messages
    # themselves and carries each as gold; asked back here, the answer is the same edge,
    # so a graph built by reading the corpus can be asked exactly what the corpus said.
    for who in people:
        for rel, phrase in _STATED_ASK.items():
            others = [o for o in t.out.get(rel, {}).get(who, ()) if t.kind(o) == "person"]
            if rel == "works_with":
                others = sorted({*others, *[o for o in t.inc.get(rel, {}).get(who, ())
                                            if t.kind(o) == "person"]})
            if 1 <= len(others) <= 8:
                b.setdefault("person", []).append(_q(phrase.format(t.label(who)), others))

    # two people the stated relations put at the same end of one edge: the connection is
    # that shared end, and the path through it is the answer
    for rel in _STATED_JOIN:
        for anchor, crowd in t.inc.get(rel, {}).items():
            crowd = sorted({p for p in crowd if t.kind(p) == "person"})
            if anchor == t.org or not 2 <= len(crowd) <= 8:
                continue
            for a, c in zip(crowd[::2], crowd[1::2]):
                if anchor not in (a, c):
                    b.setdefault("path", []).append(
                        _q(f"How is {t.label(a)} connected to {t.label(c)}?", [a, anchor, c]))

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

    # --- counting and comparing: the count is scored as the people counted ----------------------
    for unit in units:
        crowd = t.people_in(unit)
        if 2 <= len(crowd) <= 15:
            b.setdefault("aggregate", []).append(
                _q(f"How many people are in {t.label(unit)}?", crowd))
    for topic in t.of_kind("topic"):
        b.setdefault("aggregate", []).extend(_people_at(
            t, "experienced_in", topic, units, "How many people know about {}?",
            "How many people in {} know about {}?", 2, 8))
    biggest = _unique_most({u: len(t.people_in(u)) for u in units})
    if biggest:
        b.setdefault("aggregate", []).append(
            _q(f"Which {t.unit_kind} has the most people in it?", [biggest]))
    employers = {o: len([p for p in crowd_at if t.kind(p) == "person"])
                 for o, crowd_at in t.inc.get("works_at", {}).items() if o != t.org}
    busiest = _unique_most(employers)
    if busiest and len(employers) >= 2:
        b.setdefault("aggregate", []).append(
            _q("Which company employs the most people here?", [busiest]))

    # --- two hops: the far end of works_with or part_of, never the person in the middle ---------
    beside: dict[str, set[str]] = {}
    for x, ys in t.out.get("works_with", {}).items():
        for y in ys:
            beside.setdefault(x, set()).add(y)
            beside.setdefault(y, set()).add(x)
    for topic in t.of_kind("topic"):
        knowers = [p for p in t.inc.get("experienced_in", {}).get(topic, ()) if t.kind(p) == "person"]
        if not knowers:
            continue
        # the people in the middle: everyone who knows it when that is few, else those in one unit
        anchors = [("", knowers)] if len(knowers) <= 3 else []
        for unit in units if len(knowers) > 3 else ():
            both = [p for p in knowers if p in set(t.people_in(unit))]
            if 1 <= len(both) <= 3:
                anchors.append((unit, both))
        for unit, middle in anchors:
            colleagues = sorted({c for p in middle for c in beside.get(p, ())} - set(middle))
            if 1 <= len(colleagues) <= 8:
                who = (f"someone in {t.label(unit)} who knows about" if unit
                       else "someone who knows about")
                b.setdefault("twohop", []).append(
                    _q(f"Who works with {who} {t.label(topic)}?", colleagues))
        homes = sorted({u for p in knowers for u in t.out.get("part_of", {}).get(p, ())
                        if t.kind(u) == t.unit_kind})
        if len(knowers) >= 2 and 1 <= len(homes) <= 4:
            b.setdefault("twohop", []).append(
                _q(f"Which {t.unit_kind}s are the people who know about {t.label(topic)} in?", homes))
        towns = sorted({c for p in knowers for c in t.out.get("based_in", {}).get(p, ())})
        if len(knowers) >= 2 and 1 <= len(towns) <= 6:
            b.setdefault("twohop", []).append(
                _q(f"Which places do the people who know about {t.label(topic)} live in?", towns))

    # --- a false premise about a real person: the place is exactly as the graph has it ----------
    settled = {p: [q for q in t.inc.get("based_in", {}).get(p, ()) if t.kind(q) == "person"]
               for p in t.of_kind("place")}
    towns = [p for p, here in settled.items() if here]
    for place in towns:
        elsewhere = [q for q in people if q not in settled[place] and t.out.get("based_in", {}).get(q)]
        others = [p for p in towns if p != place]
        if not elsewhere or not others:
            continue
        mover, to = rng.choice(elsewhere), rng.choice(others)
        if to in t.out.get("based_in", {}).get(mover, ()):
            continue
        premise = f"Since {t.label(mover)} moved to {t.label(to)}, "
        b.setdefault("trap", []).extend(_people_at(
            t, "based_in", place, units, premise + "who is left in {}?",
            premise + "who in {} is left in {}?"))

    # --- what people said, which only their own words carry -------------------------------------
    said_does: dict[str, list[str]] = {}
    labelled = {str(n.get("label") or ""): i for i, n in t.nodes.items()
                if n.get("kind") in ("opportunity", "product", "topic")}
    for who in people:
        for mid in t.nodes[who].get("messages") or ():
            text = str((t.messages.get(mid) or {}).get("text") or "")
            does = _DOES.search(text)
            if does:
                said_does.setdefault(does.group(1), []).append(who)
            lately = _LATELY.search(text)
            if lately and lately.group(1) in labelled:
                b.setdefault("quote", []).append(
                    _q(f"What did {t.label(who)} say takes most of their time?", [labelled[lately.group(1)]]))
    for does, whom in said_does.items():
        if 1 <= len(whom) <= 6:
            b.setdefault("quote", []).append(_q(f"Who said they {does}?", whom))

    for bucket in b.values():
        rng.shuffle(bucket)
    return b


_DOES = re.compile(r"(?:Mostly I|What I actually do: I|Day to day that means I) (.+?)\.$")
_LATELY = re.compile(r"Lately most of my time goes to (.+?)\.$")


def _unique_most(counted: Mapping[str, int]) -> str | None:
    """The key with the largest count, or nothing when two tie for it -- a comparative with
    a tied answer is not a comparative."""
    if not counted:
        return None
    top = sorted(counted.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0] if top[0][1] > 0 else None


# how many of each bucket a cycle takes: people most, as the bench's own set has it
_ORDER = (("person", 3), ("org", 1), ("place", 1), ("topic", 1), ("unit", 1), ("path", 1),
          ("event", 1), ("opportunity", 1), ("product", 1), ("about", 1), ("nobody", 1),
          ("aggregate", 1), ("twohop", 1), ("trap", 1), ("quote", 1))

KINDS: tuple[str, ...] = tuple(name for name, _share in _ORDER)
"""Every ``kind`` a question can carry: the bucket it was drawn from."""


def questions(world: World, n: int = 40, rng: random.Random | None = None, *,
              kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """``n`` questions about the world, each with the ids a good answer names.

    Spread over every kind of answer the world supports, rarest kinds first so a small
    ``n`` still asks about each; deterministic from ``world.seed`` unless an ``rng`` is
    given. Each is ``{"q", "expect", "kind"}`` -- the bench reads the first two exactly as
    it reads its own, and ``kind`` says which bucket in `KINDS` it came from. ``kinds``
    draws only those buckets; an unknown one is refused by name.
    """
    wanted = tuple(kinds) if kinds else KINDS
    unknown = sorted(set(wanted) - set(KINDS))
    if unknown:
        raise ValueError(f"unknown question kind(s) {unknown}; known: {', '.join(KINDS)}")
    rng = rng or random.Random(f"questions/{world.seed}/{world.size}")
    buckets = _buckets(_Truth(world.graph), rng)
    taken: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(taken) < n and any(buckets.get(name) for name in wanted):
        for name, share in _ORDER:
            if name not in wanted:
                continue
            for _ in range(share):
                while buckets.get(name):
                    one = buckets[name].pop()
                    if one["q"] not in seen:
                        seen.add(one["q"])
                        taken.append({**one, "kind": name})
                        break
                if len(taken) >= n:
                    return taken
    return taken
