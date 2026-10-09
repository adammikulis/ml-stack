"""Scoring a ranked list of ids against the ids that were the right answer.

Three numbers, all binary (an id is right or it is not): `recall_at`, the share of the right
answers inside the first ``k``; `reciprocal_rank`, one over where the first right answer
sits; and `ndcg_at`, which also rewards putting the right answers early. A question whose
right answer is nobody has nothing to score and is the caller's to set aside.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at(ranked: Sequence[str], expect: Collection[str], k: int) -> float:
    """The share of ``expect`` found in the first ``k`` of ``ranked``."""
    if not expect:
        raise ValueError("a question with no expected answer has no recall")
    return len(set(ranked[:k]) & set(expect)) / len(set(expect))


def reciprocal_rank(ranked: Sequence[str], expect: Collection[str]) -> float:
    """One over the 1-based rank of the first right answer, 0 when none was returned."""
    wanted = set(expect)
    for rank, one in enumerate(ranked, start=1):
        if one in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked: Sequence[str], expect: Collection[str], k: int) -> float:
    """Normalised discounted cumulative gain over the first ``k``, gain 1 for a right answer.

    The discount is 1/log2(position + 1) with positions from 1; the ideal list holds
    ``min(len(expect), k)`` right answers first.
    """
    wanted = set(expect)
    if not wanted:
        raise ValueError("a question with no expected answer has no nDCG")
    gain = sum(1.0 / math.log2(i + 2) for i, one in enumerate(ranked[:k]) if one in wanted)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(wanted), k)))
    return gain / ideal
