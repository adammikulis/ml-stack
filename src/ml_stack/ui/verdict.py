"""The verdict vocabulary shared with the browser: green, yellow, red, none."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Verdict = Literal["green", "yellow", "red", "none"]

_DEFINITION = json.loads((Path(__file__).parent / "assets" / "verdict.json").read_text("utf-8"))
NAMES: tuple[str, ...] = tuple(_DEFINITION["names"])
LABELS: dict[str, str] = dict(_DEFINITION["labels"])


@dataclass(frozen=True)
class Thresholds:
    """The share of capacity at which a bar turns yellow and red."""

    yellow_at: float = _DEFINITION["yellow_at"]
    red_at: float = _DEFINITION["red_at"]


THRESHOLDS = Thresholds()


def verdict_of(used: float, capacity: float, thresholds: Thresholds = THRESHOLDS) -> Verdict:
    """``green`` below ``yellow_at`` of capacity, ``red`` from ``red_at``, ``yellow`` between;
    ``none`` when there is no capacity to measure against."""
    if capacity <= 0 or used < 0:
        return "none"
    share = used / capacity
    if share >= thresholds.red_at:
        return "red"
    return "yellow" if share >= thresholds.yellow_at else "green"
