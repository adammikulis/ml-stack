"""Splitting data so the held-out score means something."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

GUARD = 8
"""Rows dropped either side of a split boundary, so no training row is adjacent to a"""


class LeakageError(ValueError):
    """The requested split cannot be made without leaking."""


@dataclass(frozen=True, slots=True)
class Split:
    train: list[Any]
    holdout: list[Any]
    dropped: int = 0
    """Rows discarded at the boundary. Reported rather than hidden -- if it is a large"""

    def __str__(self) -> str:
        return f"{len(self.train)} train / {len(self.holdout)} holdout ({self.dropped} dropped)"


def contiguous_tail(rows: Sequence[T], fraction: float = 0.005, *, guard: int = GUARD) -> Split:
    """Hold out a contiguous block from the end, with a guard band before it."""
    total = len(rows)
    count = max(1, int(total * fraction)) if total else 0
    if count + guard >= total:
        raise LeakageError(
            f"holding out {count} of {total} rows with a {guard}-row guard leaves no "
            "training data"
        )
    cut = total - count
    return Split(train=list(rows[: cut - guard]), holdout=list(rows[cut:]), dropped=guard)


def by_group(
    rows: Sequence[T],
    groups: Sequence[Any],
    fraction: float = 0.1,
    *,
    seed: int = 0,
) -> Split:
    """Hold out whole groups, never rows within a group."""
    if len(rows) != len(groups):
        raise LeakageError(f"{len(rows)} rows but {len(groups)} group labels")

    import random

    unique = sorted({str(g) for g in groups})
    if len(unique) < 2:
        raise LeakageError(
            f"all rows are in one group ({unique[0] if unique else 'none'}), so no "
            "group-wise split exists. Either the grouping is wrong or this data cannot "
            "support a held-out set."
        )

    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    held = set(shuffled[: max(1, int(len(unique) * fraction))])

    train = [r for r, g in zip(rows, groups) if str(g) not in held]
    holdout = [r for r, g in zip(rows, groups) if str(g) in held]
    return Split(train=train, holdout=holdout)


def assert_no_duplicates(rows: Sequence[Any], *, key=repr, label: str = "dataset") -> None:
    """Fail if the data contains repeated examples."""
    seen: dict[str, int] = {}
    for row in rows:
        k = key(row)
        seen[k] = seen.get(k, 0) + 1

    repeated = {k: n for k, n in seen.items() if n > 1}
    if repeated:
        worst = sorted(repeated.items(), key=lambda kv: -kv[1])[:3]
        raise LeakageError(
            f"{label} contains {len(repeated)} repeated example(s); the most frequent "
            f"appears {worst[0][1]} times. Holding one out leaves its copies in training. "
            "Deduplicate before splitting."
        )


def spread_order(n: int) -> list[int]:
    """Indices ordered so that every prefix spans the whole range."""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    order = [0, n - 1]
    seen = {0, n - 1}
    frontier = [(0, n - 1)]
    while len(order) < n:
        next_frontier: list[tuple[int, int]] = []
        for lo, hi in frontier:
            mid = (lo + hi) // 2
            if mid not in seen:
                seen.add(mid)
                order.append(mid)
                if len(order) == n:
                    return order
            if mid - lo > 1:
                next_frontier.append((lo, mid))
            if hi - mid > 1:
                next_frontier.append((mid, hi))
        if not next_frontier:
            break
        frontier = next_frontier

    order.extend(i for i in range(n) if i not in seen)
    return order
