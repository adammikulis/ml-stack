"""What ``ml-stack-world`` does, as functions that take values and return them.

`invent` writes a world, `ask` draws questions off its truth, `export` writes what was
said the way a product exports it, and `read_messages` reads a simulation's messages back.
`ml_stack.world.cli` parses and prints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.files import write_json
from ml_stack.world import Message, emit
from ml_stack.world.organisation import load, make, summary
from ml_stack.world.questions import questions

__all__ = ["EXPORTS", "Export", "ask", "export", "invent", "read_messages"]

EXPORTS = ("slack-export", "mbox", "teams", "rows")
"""What ``emit --as`` can write: a Slack export directory, an mbox, chatMessage JSON as the
Graph API returns it, or a scraper's rows as JSONL."""

_SOURCES = {"slack-export": "slack", "mbox": "email", "teams": "teams", "rows": "slack"}


@dataclass(frozen=True, slots=True)
class Export:
    """Where the talk is, which product's shape to write it in, and where it goes."""

    talk: str
    shape: str
    out: str
    world: str = ""
    domain: str = "example.com"
    every: bool = False


def invent(*, kind: str, size: str, seed: int, out: str) -> dict[str, Any]:
    """Write a world's graph, personas and calendar under ``out``; returns its summary."""
    world = make(kind, size, seed)
    where = home.expand(out)
    write_json(where / "graph.json", world.graph)
    write_json(where / "personas.json", world.personas)
    write_json(where / "calendar.json", world.calendar)
    write_json(where / "world.json",
               {"kind": world.kind, "size": world.size, "seed": world.seed,
                "people": world.people,
                "organisation": world.graph["meta"]["world"]["organisation"]})
    made = summary(world)
    made["out"] = str(where)
    return made


def ask(world: str, count: int, kinds: str = "") -> list[dict[str, Any]]:
    """Questions with known answers off ``world``'s truth, in the shape the bench reads."""
    wanted = [k.strip() for k in (kinds or "").split(",") if k.strip()]
    return questions(load(world), count, kinds=wanted or None)


def read_messages(path: str | Path) -> list[Message]:
    """The ``messages.jsonl`` a simulation wrote, as `Message`s again."""
    out = []
    for line in home.expand(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["recipients"] = tuple(row.get("recipients") or ())
            out.append(Message(**row))
    return out


def _people_of(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(n["id"]): {"label": str(n.get("label") or "")}
            for n in graph.get("nodes") or () if n.get("kind") == "person"}


def _people_for(talk: Path, world: str, messages: list[Message]) -> dict[str, dict[str, Any]]:
    """The people's names, from the simulation's graph, the world's, or the messages."""
    for where in (talk / "graph.json",
                  home.expand(world) / "graph.json" if world else None):
        if where and where.exists():
            return _people_of(json.loads(where.read_text(encoding="utf-8")))
    return {m.sender: {} for m in messages}


def export(asked: Export) -> tuple[Path, int]:
    """Write what was said the way `Export.shape` names; where it went and how many."""
    where_from = home.expand(asked.talk)
    messages = read_messages(where_from / "messages.jsonl" if where_from.is_dir()
                             else where_from)
    people = _people_for(where_from, asked.world, messages)
    target = home.expand(asked.out)
    source = None if asked.every else _SOURCES[asked.shape]
    writer = {"slack-export": emit.slack_export, "mbox": emit.mbox,
              "teams": emit.teams}.get(asked.shape)
    if writer is not None:
        return (writer(messages, people, target, source=source, domain=asked.domain),
                len(messages))
    rows = emit.rows(messages, people, source=source, domain=asked.domain)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")
    return target, len(messages)
