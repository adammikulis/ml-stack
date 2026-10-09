"""The things a fact is about: models, builds, settings, tasks and topics, checked and given
one plain identity so two spellings of a name are one node."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from poolhouse.hub.naming import pretty_name
from poolhouse.memory.facts import Refused, check

__all__ = ["ENTITY_KINDS", "MAX_ENTITIES", "MAX_NAME_CHARS", "Entity", "parse"]

ENTITY_KINDS = ("model", "build", "setting", "task", "topic")
MAX_ENTITIES = 6
MAX_NAME_CHARS = 80
_BUILD = re.compile(r"(?:llama\.cpp\s+|build\s+)?(b\d{3,6})", re.I)


@dataclass(frozen=True, slots=True)
class Entity:
    """``kind`` and a display ``name``; ``key`` is the identity two spellings share."""

    kind: str
    name: str

    @property
    def key(self) -> str:
        return "-".join(self.name.casefold().split())

    @property
    def id(self) -> str:
        return f"entity:{self.kind}:{self.key}"

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


def _normal(kind: str, name: str) -> str:
    if kind == "model":
        return " ".join(pretty_name(name).split())
    if kind == "build":
        found = _BUILD.fullmatch(name.strip())
        return found.group(1).lower() if found else name
    return name


def _one(raw: Any) -> Entity:
    if isinstance(raw, Mapping):
        kind, name = str(raw.get("kind", "")), str(raw.get("name", ""))
    else:
        kind, _, name = str(raw).partition(":")
    kind = kind.strip().casefold()
    if kind not in ENTITY_KINDS:
        raise Refused(f"an entity is written kind:name, with kind one of {', '.join(ENTITY_KINDS)}")
    name = check(name, person=False)
    if len(name) > MAX_NAME_CHARS:
        raise Refused(f"an entity name is at most {MAX_NAME_CHARS} characters")
    name = _normal(kind, name)
    if not name or len(name) > MAX_NAME_CHARS:
        raise Refused("an entity needs a name")
    return Entity(kind, name)


def parse(raw: Iterable[Any] | str | None) -> list[Entity]:
    """The checked, normal, de-duplicated entities in ``raw`` (``kind:name`` strings or
    ``{"kind", "name"}`` mappings); ``Refused`` for a bad kind or name or more than
    ``MAX_ENTITIES``."""
    if raw is None or raw == "":
        return []
    items = [raw] if isinstance(raw, (str, Mapping)) else list(raw)
    out: dict[str, Entity] = {}
    for item in items:
        entity = _one(item)
        out.setdefault(entity.id, entity)
    if len(out) > MAX_ENTITIES:
        raise Refused(f"a fact names at most {MAX_ENTITIES} entities")
    return list(out.values())
