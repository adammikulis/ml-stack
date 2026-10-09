"""Trust levels, labels, and a value that carries its label."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

__all__ = ["Label", "Labelled", "Level", "join"]


class Level(IntEnum):
    """How far text can be trusted: untrusted < user < system."""

    UNTRUSTED = 0
    USER = 1
    SYSTEM = 2


@dataclass(frozen=True, slots=True)
class Label:
    """A trust level and the id of where the text came from, such as ``tool:web_fetch#3``."""

    level: Level
    origin: str = ""


def join(*labels: Label) -> Label:
    """The label of text built from all of ``labels``: its lowest level, the origins joined."""
    if not labels:
        return Label(Level.SYSTEM)
    low = min(label.level for label in labels)
    origins = dict.fromkeys(label.origin for label in labels if label.origin)
    return Label(low, ",".join(origins))


@dataclass(frozen=True, slots=True)
class Labelled[T]:
    """``value`` and the label it carries."""

    value: T
    label: Label
