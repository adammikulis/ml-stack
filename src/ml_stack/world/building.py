"""The accumulator a world is built into, and the pieces every kind is assembled from.

`_Build` collects nodes, edges and quotes in insertion order, so a seed reproduces a
world exactly. The rest are the moves each kind of organised group makes: sharing a
headcount out, running reporting lines, hiring somebody, giving them skills, work,
events, teammates, mentors, day jobs, partners and funders.
"""

from __future__ import annotations

import datetime as _dt
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ml_stack.world.catalogue import CITIES, DEPARTMENTS, IC_LEVELS, IC_WEIGHTS, INDUSTRIES, TODAY
from ml_stack.world.names import company_name, person_name, product_name, slug

# --- the accumulator ---------------------------------------------------------------------

class _Build:
    """Nodes, edges and quotes as they are made, in insertion order, so a seed reproduces them."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.joined: set[tuple[str, str, str]] = set()
        self.said: dict[str, list[str]] = {}
        self.people: list[str] = []
        self.names: set[tuple[str, str]] = set()

    def node(self, node_id: str, kind: str, label: str, **attrs: Any) -> str:
        if node_id in self.nodes:
            return node_id
        self.nodes[node_id] = {"id": node_id, "label": label, "kind": kind, "mentions": 0,
                               "attrs": {"member": kind == "person", **attrs}, "messages": []}
        return node_id

    def edge(self, source: str, rel: str, target: str, weight: int = 1) -> None:
        key = (source, rel, target)
        if key in self.joined or source == target:
            return
        self.joined.add(key)
        self.edges.append({"source": source, "rel": rel, "target": target, "weight": weight,
                           "messages": []})

    def person(self, **attrs: Any) -> str:
        """A new person with a name nobody in this world has yet."""
        while True:
            given, family = person_name(self.rng)
            if (given, family) not in self.names:
                break
        self.names.add((given, family))
        node_id = f"person:{slug(given)}-{slug(family)}"
        n = 2
        while node_id in self.nodes:
            node_id = f"person:{slug(given)}-{slug(family)}-{n}"
            n += 1
        self.node(node_id, "person", f"{given} {family}", **attrs)
        self.people.append(node_id)
        return node_id

    def place(self, city: str, country: str) -> str:
        return self.node(f"place:{slug(city)}", "place", city, country=country)

    def topic(self, name: str) -> str:
        return self.node(f"topic:{slug(name)}", "topic", name)

    def org(self, label: str, **attrs: Any) -> str:
        node_id = f"org:{slug(label)}"
        n = 2
        while node_id in self.nodes:
            node_id = f"org:{slug(label)}-{n}"
            n += 1
        return self.node(node_id, "org", label, **attrs)

    def label(self, node_id: str) -> str:
        return str(self.nodes[node_id]["label"])

    def attrs(self, node_id: str) -> dict[str, Any]:
        return self.nodes[node_id]["attrs"]


# --- helpers shared by every kind -----------------------------------------------------------

def _split(rng: random.Random, total: int, weights: Sequence[float], minimum: int = 1) -> list[int]:
    """``total`` shared out in proportion to ``weights``, nobody below ``minimum``."""
    if not weights:
        return []
    floor = min(minimum, total // len(weights))
    spare = total - floor * len(weights)
    whole = sum(weights)
    raw = [spare * w / whole for w in weights]
    out = [floor + int(r) for r in raw]
    left = total - sum(out)
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - int(raw[i])))[:left]:
        out[i] += 1
    return out


def _tree(rng: random.Random, members: list[str], head: str,
          low: int = 5, high: int = 9) -> dict[str, str]:
    """Reporting lines from ``members`` up to ``head``, every span between ``low`` and ``high``.

    Level by level: parents at one level take a span each from what is left; when what is
    left fits in one level, only as many parents as keep the spans sane are used and the rest
    of that level stay leaves. Returns ``child -> parent``.
    """
    parents = {}
    level = [head]
    left = list(members)
    while left:
        if len(left) <= high * len(level):
            k = max(1, min(len(level), -(-len(left) // ((low + high) // 2))))
            takers = level[:k]
            shares = _split(rng, len(left), [1] * k)
        else:
            takers = level
            shares = [rng.randint(low, high) for _ in level]
        nxt = []
        for parent, share in zip(takers, shares):
            for _ in range(share):
                if not left:
                    break
                child = left.pop(0)
                parents[child] = parent
                nxt.append(child)
        level = nxt
    return parents


def _reports(parents: Mapping[str, str]) -> dict[str, list[str]]:
    """``parent -> children``, in the order the children were given."""
    kids: dict[str, list[str]] = {}
    for child, parent in parents.items():
        kids.setdefault(parent, []).append(child)
    return kids


def _heights(parents: Mapping[str, str]) -> dict[str, int]:
    """How far above the leaves each node stands: 0 for a leaf."""
    kids = _reports(parents)
    heights: dict[str, int] = {}

    def height(node: str) -> int:
        if node not in heights:
            heights[node] = 1 + max((height(k) for k in kids.get(node, ())), default=-1)
        return heights[node]

    for node in list(parents) + list(kids):
        height(node)
    return heights


def _ic_level(rng: random.Random) -> tuple[str, str]:
    return rng.choices(IC_LEVELS, IC_WEIGHTS)[0]


def _started(rng: random.Random, founded: int, level_bias: float = 0.0) -> tuple[str, float]:
    """A start date between the founding and today, and the tenure in years it implies."""
    first = _dt.date(max(founded, 1900), 1, 1)
    span = (TODAY - first).days
    # senior people have usually been around longer; the bias pulls the date back
    frac = rng.random() ** (1 + level_bias)
    day = TODAY - _dt.timedelta(days=int(span * frac))
    return day.isoformat(), round((TODAY - day).days / 365.25, 1)


def _offices(b: _Build, n: int) -> list[str]:
    cities = b.rng.sample(CITIES, n)
    return [b.place(city, country) for city, country in cities]


def _cities(b: _Build) -> list[str]:
    """Every city there is, as place nodes."""
    return [b.place(city, country) for city, country in CITIES]


def _hiring(b: _Build, org: str, founded: int, offices: Sequence[str] = (),
            home: int = 0) -> Callable[..., str]:
    """A function that adds one person to ``org``, started some time since ``founded``.

    ``attrs`` become the person's own; ``bias`` among them pulls the start date back.
    With ``offices`` the person is also based in one of them, the first ``home`` times
    as likely as the rest.
    """
    def hire(**attrs: Any) -> str:
        started, tenure = _started(b.rng, founded, attrs.pop("bias", 0.0))
        who = b.person(started=started, tenure_years=tenure, **attrs)
        b.edge(who, "works_at", org)
        if offices:
            weights = [home] + [1] * (len(offices) - 1)
            b.edge(who, "based_in", b.rng.choices(list(offices), weights)[0])
        return who

    return hire


def _partners(b: _Build, org: str, n: int, cities: Sequence[str]) -> None:
    """``n`` invented organisations that partner with ``org``, each in a city."""
    for _ in range(n):
        p = b.org(company_name(b.rng), industry=b.rng.choice(INDUSTRIES)[0], type="partner")
        b.edge(p, "partner_of", org)
        b.edge(p, "based_in", b.rng.choice(list(cities)))


def _funders(b: _Build, n: int, cities: Sequence[str],
             funds: Callable[[], Sequence[str]]) -> None:
    """``n`` invented funders, each in a city, each funding what ``funds`` picks."""
    for _ in range(n):
        f = b.org(company_name(b.rng, kind=b.rng.choice(("Foundation", "Trust", "Council"))),
                  type="funder")
        b.edge(f, "based_in", b.rng.choice(list(cities)))
        for target in funds():
            b.edge(f, "funds", target)


def _skills(b: _Build, who: str, pool: Sequence[str], others: Sequence[str],
            n: tuple[int, int] = (2, 4)) -> list[str]:
    """Two to four skills from the pool, sometimes one from elsewhere. Returns topic ids."""
    chosen = b.rng.sample(list(pool), min(len(pool), b.rng.randint(*n)))
    if others and b.rng.random() < 0.25:
        chosen.append(b.rng.choice(list(others)))
    ids = [b.topic(s) for s in chosen]
    for t in ids:
        b.edge(who, "experienced_in", t, weight=b.rng.randint(1, 3))
    return ids


def _projects(b: _Build, n: int, *, naming: Sequence[str],
              skills_of: Mapping[str, Sequence[str]],
              size: tuple[int, int] = (4, 8)) -> list[str]:
    """``n`` pieces of work, each ``offers``-ed by one of ``skills_of``'s units and worked on
    by a handful of people drawn from more than one unit. Each wants one or two skills."""
    rng = b.rng
    out = []
    people, units = b.people, list(skills_of)
    by_unit: dict[str, list[str]] = {}
    for p in people:
        by_unit.setdefault(b.attrs(p).get("unit_id", ""), []).append(p)
    for i in range(n):
        name = f"{rng.choice(naming)} {product_name(rng).split()[0]}"
        pid = b.node(f"opportunity:{slug(name)}-{i}", "opportunity", name)
        owner = rng.choice(list(units)) if units else ""
        if owner:
            b.edge(owner, "offers", pid)
        want = set(skills_of.get(owner, ()) or ())
        team = []
        home = by_unit.get(owner, [])
        if home:
            team += rng.sample(home, min(len(home), max(2, rng.randint(*size) * 2 // 3)))
        while len(team) < rng.randint(*size) and len(team) < len(people):
            other = rng.choice(list(people))
            if other not in team:
                team.append(other)
        for who in team:
            b.edge(who, "works_on", pid, weight=rng.randint(1, 4))
            b.attrs(who).setdefault("projects", []).append(name)
        for s in rng.sample(sorted(want), min(len(want), rng.randint(1, 2))) if want else []:
            b.edge(pid, "wants", b.topic(s))
        # pairs on the same piece of work know each other
        for x in team:
            for y in team:
                if x < y and rng.random() < 0.6:
                    b.edge(x, "works_with", y, weight=rng.randint(1, 3))
        out.append(pid)
    return out


def _events(b: _Build, people: Sequence[str], names: Sequence[str], n: int) -> list[str]:
    """``n`` events off that list of names, each attended by a crowd the size of the place."""
    rng = b.rng
    out = []
    crowd = (6, 12) if len(people) < 200 else (10, 40)
    for i in range(n):
        name = names[i % len(names)]
        if i >= len(names):
            name = f"{name} {2020 + i // len(names)}"
        eid = b.node(f"event:{slug(name)}", "event", name, day=rng.randint(-365, 60))
        for who in rng.sample(list(people), min(len(people), rng.randint(*crowd))):
            b.edge(who, "attended", eid)
        out.append(eid)
    return out


def _teams(b: _Build, groups: Iterable[Sequence[str]], p: float = 0.5) -> None:
    """People in the same small group work with a few of the others."""
    rng = b.rng
    for group in groups:
        group = list(group)
        for x in group:
            for y in rng.sample(group, min(len(group), 3)):
                if x != y and rng.random() < p:
                    b.edge(min(x, y), "works_with", max(x, y), weight=rng.randint(1, 3))


def _mentors(b: _Build, seniors: Sequence[str], juniors: Sequence[str], share: float = 0.08) -> None:
    rng = b.rng
    if not seniors or not juniors:
        return
    for junior in juniors:
        if rng.random() < share:
            b.edge(rng.choice(list(seniors)), "mentors", junior)


def _employers(b: _Build, people: Sequence[str], n_orgs: int, offices: Sequence[str]) -> list[str]:
    """Day jobs at many invented organisations, for the kinds that have no single employer."""
    rng = b.rng
    orgs = [b.org(company_name(rng), industry=rng.choice(INDUSTRIES)[0]) for _ in range(n_orgs)]
    for o in orgs:
        b.edge(o, "based_in", rng.choice(list(offices)))
    at: dict[str, list[str]] = {}
    for who in people:
        # a few big employers and a long tail, the way a community actually is
        o = orgs[min(int(rng.expovariate(1.0) * n_orgs / 3), n_orgs - 1)]
        b.edge(who, "works_at", o)
        b.attrs(who)["employer"] = b.label(o)
        at.setdefault(o, []).append(who)
    _teams(b, at.values(), p=0.35)
    return orgs


def _day_job(b: _Build, who: str) -> tuple[str, str, str]:
    """A title, level and responsibility borrowed from the company catalogue."""
    dept = b.rng.choice(list(DEPARTMENTS))
    track, does = b.rng.choice(DEPARTMENTS[dept]["tracks"])
    level, word = _ic_level(b.rng)
    return f"{word}{track}", level, does
