"""Learning-rate schedules as plain functions of the step.

``schedule(step) -> float``, computed in Python, returning a float. Not a framework
schedule object.

That is a deliberate constraint rather than a stylistic one. A schedule object held by an
optimizer becomes part of the optimizer's state, and under a graph-compiling backend the
compiled function captures the learning rate as it was at trace time -- so the rate never
changes again and the model trains at a constant LR for the rest of the run, with nothing
reporting it. A plain float assigned from outside the compiled region cannot do that.
"""

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
    """Warmup, then a long constant stretch, then a linear decay at the end.

    Worth preferring over cosine when the total step count might change. Cosine bakes the
    horizon into every step's value, so extending a run means every rate after the original
    end is wrong, and shortening it means the decay never happens. WSD only needs to know
    the horizon during its final ``decay_fraction``, so the stable stretch can be extended
    without invalidating what came before.
    """
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
