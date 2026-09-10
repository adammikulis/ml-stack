"""An organised group that does not exist, as a graph, at any size.

`make(kind, size, seed)` invents one from a seed and returns a `World`: the graph in the
shape `ml_stack.graph.community` already uses, the people in it, and a persona for each.
`load` reads one back off disk, `summary` counts what is in it, and `role_catalogue`
lists every title the company builder can hand out.

`world.catalogue` holds the vocabulary, `world.building` the pieces, `world.kinds` the
five builders. Nothing here is a real person or organisation.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ml_stack.world import World, about
from ml_stack.world.building import _Build, _offices
from ml_stack.world.catalogue import (
    C_LEVELS,
    DEPARTMENTS,
    IC_LEVELS,
    KINDS,
    SIZES,
    UNIT_KIND,
    VOICES,
)
from ml_stack.world.kinds import BUILDERS

__all__ = ["KINDS", "SIZES", "UNIT_KIND", "load", "make", "role_catalogue", "summary"]


def _quotes(b: _Build, org: str) -> None:
    """One to three things each person would say about their work, from role and projects,
    so `look_up`'s "said" voter has something to find."""
    skills: dict[str, list[str]] = {}
    for e in b.edges:
        if e["rel"] == "experienced_in":
            skills.setdefault(e["source"], []).append(b.label(e["target"]))
    rng = b.rng
    for who in b.people:
        a = b.attrs(who)
        unit, where = a.get("unit", ""), a.get("employer") or b.label(org)
        title, does = a.get("title", "here"), a.get("does", "keep busy")
        lines = [rng.choice((
            f"I'm {title} in {unit} at {where}. Mostly I {does}.",
            f"{title}, {unit}. What I actually do: I {does}.",
            f"My job is {title}. Day to day that means I {does}.",
        ))]
        projects = a.get("projects") or []
        if projects and rng.random() < 0.8:
            lines.append(f"Lately most of my time goes to {projects[0]}."
                         if len(projects) == 1 or rng.random() < 0.5 else
                         f"Ask me about {projects[0]} -- I'm on it, along with {projects[-1]}.")
        mine = skills.get(who, [])
        if mine and rng.random() < 0.6:
            lines.append(f"Happy to help with {' or '.join(mine[:2])}.")
        b.said[who] = lines[: rng.randint(1, 3)]


def _personas(b: _Build, org: str, kind: str, public: Sequence[str]) -> dict[str, dict[str, Any]]:
    """A voice, a system prompt and what each person would know: the graph two hops out,
    stepping through people and pieces of work but not through hubs, plus every public node."""
    rng = b.rng
    near: dict[str, list[str]] = {}
    for e in b.edges:
        near.setdefault(e["source"], []).append(e["target"])
        near.setdefault(e["target"], []).append(e["source"])
    walkable = {"person", "opportunity"}
    out = {}
    for who in b.people:
        a = b.attrs(who)
        voice = rng.choice(VOICES)
        first = near.get(who, [])
        knows = {who, *public, *first}
        for step in first:
            if b.nodes[step]["kind"] in walkable:
                knows.update(near.get(step, ()))
        bits = [f"You are {b.label(who)}, {a.get('title', 'a member')}"
                + (f" in {a['unit']}" if a.get("unit") else "")
                + f" at {a.get('employer') or b.label(org)}"
                + (f", which is a {kind.replace('-', ' ')}" if not a.get("employer") else
                   f", and a member of {b.label(org)}") + "."]
        if a.get("does"):
            bits.append(f"You {a['does']}.")
        if a.get("manager"):
            bits.append(f"You report to {a['manager']}.")
        if a.get("projects"):
            bits.append("You are working on " + ", ".join(a["projects"][:3]) + ".")
        skills = [b.label(t) for t in first if b.nodes[t]["kind"] == "topic"]
        if skills:
            bits.append("You know about " + ", ".join(skills) + ".")
        place = next((b.label(t) for t in first if b.nodes[t]["kind"] == "place"), "")
        bits.append(f"You are based in {place}." if place else "You work remotely.")
        bits.append(f"How you write: {voice}")
        bits.append("Speak as this person, in the first person, from what you know and "
                    "nothing else; when you do not know, say who would.")
        out[who] = {"voice": voice, "system": " ".join(bits), "knows": sorted(knows)}
    return out


def make(kind: str = "company", size: str = "small", seed: int = 0) -> World:
    """An organised group of that kind and size, invented from the seed: the same every time.

    ``kind`` is one of `KINDS`; ``size`` one of `SIZES` (50, 500 or 5000 people).
    """
    if kind not in BUILDERS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {', '.join(SIZES)}, not {size!r}")
    n = SIZES[size]
    b = _Build(random.Random(f"{kind}/{size}/{seed}"))
    offices = _offices(b, {"small": 3, "medium": 6}.get(size, 12))
    made = BUILDERS[kind](b, n, offices)
    org = made["organisation"]
    for who in b.people:
        b.attrs(who).pop("bias", None)
        b.attrs(who).pop("unit_id", None)
    _quotes(b, org)

    # what a reader of this graph sees: quotes as messages, mentions as degree
    messages: dict[str, dict[str, Any]] = {}
    n_msg = 0
    for who, lines in b.said.items():
        for line in lines:
            mid = f"m{n_msg}"
            messages[mid] = {"text": line, "ts": str(1_756_600_000 + n_msg * 900),
                             "channel": "#introductions", "sender": b.label(who)}
            b.nodes[who]["messages"].append(mid)
            n_msg += 1
    degree: dict[str, int] = {}
    for e in b.edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1
    for node_id, node in b.nodes.items():
        node["mentions"] = max(1, degree.get(node_id, 0))

    unit_kind = UNIT_KIND[kind]
    public = [i for i, node in b.nodes.items()
              if node["kind"] in ("place", unit_kind, "product", "body") or i == org]
    personas = _personas(b, org, kind, public)
    graph = {"nodes": list(b.nodes.values()), "edges": b.edges, "messages": messages,
             "stats": {"messages": len(messages), "people": len(b.people)},
             "meta": {"community": "invented",
                      "world": {"kind": kind, "size": size, "seed": seed, "organisation": org,
                                "root": made.get("root", ""), "unit_kind": unit_kind}}}
    return World(graph=graph, people=list(b.people), personas=personas, calendar=[],
                 seed=seed, size=size, kind=kind)


def summary(world: World) -> dict[str, Any]:
    """People, units and edges by relation: what `make` made, in one mapping."""
    meta = (world.graph.get("meta") or {}).get("world") or {}
    by_rel: dict[str, int] = {}
    for e in world.graph["edges"]:
        by_rel[e["rel"]] = by_rel.get(e["rel"], 0) + 1
    by_kind: dict[str, int] = {}
    for n in world.graph["nodes"]:
        by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
    return {"kind": meta.get("kind", ""), "size": world.size, "seed": world.seed,
            "organisation": meta.get("organisation", ""), "people": len(world.people),
            "units": by_kind.get(meta.get("unit_kind", ""), 0), "nodes": len(world.graph["nodes"]),
            "edges": len(world.graph["edges"]), "nodes_by_kind": dict(sorted(by_kind.items())),
            "edges_by_relation": dict(sorted(by_rel.items()))}


def role_catalogue() -> dict[str, list[str]]:
    """Every title the company builder can hand out, by department: what the org chart may say."""
    out = {}
    for dept, held in DEPARTMENTS.items():
        titles = [f"{word}{track}" for track, _ in held["tracks"] for _, word in IC_LEVELS]
        titles += [f"{dept.title()} Manager", f"Director of {dept.title()}",
                   f"VP {dept.title()}", f"Head of {dept.title()}"]
        out[dept] = titles
    out["executive"] = ["Chief Executive Officer", *(t for t, _, _ in C_LEVELS)]
    return out


def load(where: str | Path) -> World:
    """The world `ml-stack-world make --out DIR` wrote, read back.

    ``world.json`` (kind, size, seed, people) is read when it is there, the way
    `world.simulate.run` reads it; without it the graph's own ``meta`` says the same.
    """
    where = Path(where).expanduser()
    graph = json.loads((where / "graph.json").read_text(encoding="utf-8"))
    personas = json.loads((where / "personas.json").read_text(encoding="utf-8"))
    calendar_file = where / "calendar.json"
    calendar = json.loads(calendar_file.read_text(encoding="utf-8")) if calendar_file.exists() else []
    meta = (graph.get("meta") or {}).get("world") or {}
    said = about.read(where)
    people = [str(p) for p in said.get("people") or ()] or \
        [str(n["id"]) for n in graph.get("nodes") or () if n.get("kind") == "person"]
    return World(graph=graph, people=people, personas=personas, calendar=calendar,
                 seed=int(said.get("seed", meta.get("seed", 0))),
                 size=str(said.get("size") or meta.get("size") or "small"),
                 kind=str(said.get("kind") or meta.get("kind") or "company"))
