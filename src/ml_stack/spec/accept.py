"""Which path of a verified tree is kept.

At each node the target draws a token from its processed distribution P. ``lossless`` follows
the child equal to the draw and stops at the first miss, so the output is distributed exactly
as plain sampling; greedy decoding is lossless under every rule. ``ratio`` may also take a
drafted child the draw missed, when the probability mass that moves, ``1 - P(children)``,
divided by the tokens that child's verified continuation unlocks, is at most ``theta``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ml_stack.spec.tree import Tree

__all__ = ["Accepted", "Rule", "walk"]

RULES = {"lossless": {}, "ratio": {"theta": 0.3, "tau": 0.0}}


@dataclass(frozen=True)
class Rule:
    """An acceptance rule and its settings, from ``lossless`` or ``ratio:theta=0.3,tau=0``."""

    name: str = "lossless"
    theta: float = 0.3
    tau: float = 0.0

    @classmethod
    def parse(cls, text: str) -> Rule:
        name, _, rest = str(text).partition(":")
        if name not in RULES:
            raise ValueError(f"unknown acceptance rule {name!r}; choose from {', '.join(RULES)}")
        settings: dict[str, float] = dict(RULES[name])
        for item in filter(None, rest.split(",")):
            key, _, value = item.partition("=")
            if key not in settings:
                raise ValueError(f"rule {name!r} has no setting {key!r}")
            settings[key] = float(value)
        return cls(name, **settings)

    def __str__(self) -> str:
        return self.name if self.name == "lossless" else f"{self.name}:theta={self.theta},tau={self.tau}"


@dataclass(frozen=True)
class Accepted:
    """The kept path (node indices from the root), the token after it, and how many were forced."""

    path: list[int]
    bonus: int
    forced: int = 0


def _continuation(tree: Tree, drawn: Sequence[int], node: int) -> int:
    length = 0
    while True:
        hit = [j for j in tree.children[node] if tree.tokens[j] == drawn[node]]
        if not hit:
            return length
        node, length = hit[0], length + 1


def walk(tree: Tree, drawn: Sequence[int], ids: np.ndarray | None, probs: np.ndarray | None,
         rule: Rule) -> Accepted:
    """The path ``rule`` keeps, given one draw per node and the distributions they came from."""
    node, path, forced = 0, [0], 0
    while True:
        children = tree.children[node]
        hit = [j for j in children if tree.tokens[j] == drawn[node]]
        if hit:
            node = hit[0]
            path.append(node)
            continue
        if not children or rule.name == "lossless" or ids is None or probs is None:
            return Accepted(path, int(drawn[node]), forced)
        mass = {int(t): float(p) for t, p in zip(ids[node], probs[node], strict=True) if p > 0}
        held = [mass.get(tree.tokens[j], 0.0) for j in children]
        top = float(probs[node][0])
        eligible = [i for i, p in enumerate(held) if p > 0 and p >= rule.tau * top]
        if not eligible:
            return Accepted(path, int(drawn[node]), forced)
        gains = {i: 1 + _continuation(tree, drawn, children[i]) for i in eligible}
        best = max(eligible, key=lambda i: (gains[i], held[i]))
        if (1.0 - sum(held)) / gains[best] > rule.theta:
            return Accepted(path, int(drawn[node]), forced)
        node = children[best]
        path.append(node)
        forced += 1
