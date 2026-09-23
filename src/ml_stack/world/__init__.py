"""An invented company, for demos that can be shown and for measurements at any size.

A demo of a graph read out of a community needs a community, and a real one cannot be shown.
`world` invents one from a seed -- a company, its offices and customers, its people with
reasonable jobs and a voice each -- as a graph in the shape `ml_stack.graph.community`
already uses, so the store, the bench, the page and the ask loop take it unchanged. The
people then talk (`world.simulate`), grounded in that graph as their memory, and what they
say is written out the way each product exports it (`world.emit`) and read back by
`ml_stack.sources`.

Nothing here is a real person or organisation. Names come from syllable tables, companies
from word lists, and the only real things are cities.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["World", "Writer"]

Writer = Callable[[Mapping[str, Any], str, Mapping[str, Any]], str]
"""What writes one message: ``(persona, prompt, context) -> str``."""


@dataclass
class World:
    """A company as a graph, plus what the graph cannot hold: how each person writes.

    `graph` is `{"nodes": [...], "edges": [...]}` in `ml_stack.graph.community`'s schema
    (node: id, kind, label, mentions, attrs; edge: source, target, rel, plus whatever
    the community writes). `people` lists the person ids. `personas[id]` is
    `{"voice": str, "system": str, "knows": [ids]}` -- the voice in a sentence, the system
    prompt a writer speaks with, and the subgraph that person would know. `calendar` is a
    list of `{"day": int, "kind": str, "who": [ids], "about": str}` arcs the simulation
    schedules conversations around. `seed` reproduces all of it. `kind` is what sort of
    organised group this is -- "company", "community", "university", "open-source" or
    "nonprofit" -- which decides what there is to talk about.
    """

    graph: dict[str, Any]
    people: list[str]
    personas: dict[str, dict[str, Any]]
    calendar: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    size: str = "small"
    kind: str = "company"
