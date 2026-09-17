"""The target's output head restricted to the tokens a drafter proposes from."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ["DRAFT_VOCAB", "SubHead"]

#: the token ids a drafter scores, most frequent in the target's own output first
DRAFT_VOCAB = Path(__file__).resolve().parent.parent / "data" / "draft_vocab.npy"


class SubHead:
    """``head(h)`` over a subset of rows; ``token(i)`` maps a row back to its token id."""

    def __init__(self, head: nn.Module, ids: np.ndarray) -> None:
        self.ids = mx.array(np.asarray(ids, dtype=np.int32))
        if isinstance(head, nn.QuantizedLinear):
            self.weight = head.weight[self.ids]
            self.scales = head.scales[self.ids]
            self.biases = head.biases[self.ids] if head.biases is not None else None
            self.group_size, self.bits, self.mode = head.group_size, head.bits, head.mode
            mx.eval(self.weight, self.scales)
        else:
            self.weight = head.weight[self.ids]
            self.scales = None
            mx.eval(self.weight)

    def __call__(self, hidden: mx.array) -> mx.array:
        if self.scales is None:
            return hidden @ self.weight.T
        return mx.quantized_matmul(hidden, self.weight, self.scales, self.biases, transpose=True,
                                   group_size=self.group_size, bits=self.bits, mode=self.mode)

    def token(self, rows: mx.array) -> mx.array:
        return self.ids[rows]


def sub_head(head: nn.Module, vocab: int) -> SubHead | None:
    """The shipped draft vocabulary over ``head``, or None when the target's vocabulary is smaller."""
    ids = np.load(DRAFT_VOCAB)
    if int(ids.max()) >= vocab:
        return None
    return SubHead(head, ids)
