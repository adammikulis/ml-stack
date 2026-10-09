"""Learning-rate schedules as plain functions of the step."""

from __future__ import annotations

import math
from collections.abc import Callable

Schedule = Callable[[int], float]


def constant(lr: float) -> Schedule:
    def schedule(step: int) -> float:
        return lr

    return schedule


def warmup_cosine(
    peak: float,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    final_fraction: float = 0.1,
) -> Schedule:
    """Linear warmup, then cosine decay to ``peak * final_fraction``."""
    floor = peak * final_fraction

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return peak * (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return peak
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))

    return schedule


def warmup_stable_decay(
    peak: float,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    decay_fraction: float = 0.2,
    final_fraction: float = 0.1,
) -> Schedule:
    """Warmup, then a long constant stretch, then a linear decay at the end."""
    floor = peak * final_fraction
    decay_start = int(total_steps * (1.0 - decay_fraction))

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return peak * (step + 1) / warmup_steps
        if step < decay_start:
            return peak
        if total_steps <= decay_start:
            return floor
        progress = (step - decay_start) / (total_steps - decay_start)
        return peak + (floor - peak) * min(max(progress, 0.0), 1.0)

    return schedule


def linear_warmup(peak: float, *, warmup_steps: int) -> Schedule:
    """Warmup to ``peak``, then hold. For runs with no planned end."""

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return peak * (step + 1) / warmup_steps
        return peak

    return schedule
