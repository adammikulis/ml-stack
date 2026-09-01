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

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Message", "World"]


@dataclass(frozen=True)
class Message:
    """One thing somebody said, in whichever product they said it.

    The one shape every emitter writes and every reader returns, so a corpus of Slack,
    email and Teams is one list. Ids are the world's (`person:<slug>`); `ts` is unix seconds
    as Slack writes it ("1725148800.000100") and emitters convert. `channel` is a Slack
    channel name, `dm:<a>,<b>` for a direct message, an email subject line, or a Teams chat
    id. `thread` is the id of the root message, or None for a root. `kind` is "message",
    "reply" or "reaction". `attrs` carries what one product has and the others do not.
    """

    id: str
    source: str  # "slack" | "email" | "teams"
    channel: str
    sender: str
    ts: str
    text: str
    recipients: tuple[str, ...] = ()
    thread: str | None = None
    kind: str = "message"
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class World:
    """A company as a graph, plus what the graph cannot hold: how each person writes.

    `graph` is `{"nodes": [...], "edges": [...]}` in `ml_stack.graph.community`'s schema
    (node: id, kind, label, mentions, attrs; edge: source, target, relation, plus whatever
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
