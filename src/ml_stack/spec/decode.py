"""The decode loop: draft, verify, accept, commit -- and the session it keeps between calls."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

from ml_stack.spec.accept import Accepted, Rule, walk
from ml_stack.spec.drafters import Drafter
from ml_stack.spec.layout import Layout
from ml_stack.spec.sample import Sampling, sample_rows
from ml_stack.spec.tree import Tree
from ml_stack.spec.verify import TreeVerifier

__all__ = ["Asked", "Decoded", "Pass", "Session", "accept", "decode"]


@dataclass(frozen=True)
class Asked:
    """How far to decode and how each token is chosen."""

    max_tokens: int
    sampling: Sampling = field(default_factory=Sampling)
    rule: Rule = field(default_factory=Rule)
    eos: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Pass:
    """One verification pass: tokens it emitted, its tree size, its seconds, forced tokens."""

    emitted: int
    nodes: int
    seconds: float
    forced: int = 0


@dataclass
class Decoded:
    """What one call produced, and how."""

    tokens: list[int] = field(default_factory=list)
    passes: list[Pass] = field(default_factory=list)
    finish: str = "length"
    prompt_tokens: int = 0
    reused: int = 0
    prefill_s: float = 0.0

    @property
    def seconds(self) -> float:
        return sum(p.seconds for p in self.passes)

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds if self.seconds > 0 else 0.0

    @property
    def tokens_per_pass(self) -> float:
        return len(self.tokens) / len(self.passes) if self.passes else 0.0


@dataclass
class _Snapshot:
    count: int
    linear: dict[int, tuple[Any, Any]]
    drafter: object
    hidden: mx.array


class Session:
    """The target's caches and the drafter's context, kept across calls.

    A prompt that extends the tokens already held is restored to the longest snapshot inside
    the shared prefix and only the rest is prefilled. DeltaNet states are replaced, never
    mutated, so a snapshot holds references; attention rows below an offset are never
    rewritten, so an offset is enough. ``keep`` bounds how many snapshots are held.
    """

    def __init__(self, layout: Layout, drafter: Drafter, keep: int = 2) -> None:
        self.layout, self.drafter, self.keep = layout, drafter, keep
        self.reset()

    def reset(self) -> None:
        self.cache: list[Any] | None = None
        self.tokens: list[int] = []
        self._snapshots: list[_Snapshot] = []

    def snapshot(self, hidden: mx.array) -> None:
        """Remember the current state; ``hidden`` is the last cached token's pre-norm hidden."""
        if not self.keep or self.cache is None:
            return
        linear = {i: (c[0], c[1]) for i, c in enumerate(self.cache) if isinstance(c, ArraysCache)}
        taken = _Snapshot(len(self.tokens), linear, self.drafter.state(), hidden)
        self._snapshots = [s for s in self._snapshots if s.count < taken.count] + [taken]
        del self._snapshots[:-self.keep]

    def _restore(self, cache: list[Any], snap: _Snapshot) -> None:
        for index, (conv, state) in snap.linear.items():
            cache[index][0], cache[index][1] = conv, state
        for held in cache:
            if isinstance(held, KVCache):
                held.offset = snap.count
        self.drafter.restore(snap.drafter)
        del self.tokens[snap.count:]
        self._snapshots = [s for s in self._snapshots if s.count <= snap.count]

    def prepare(self, prompt: Sequence[int]) -> tuple[TreeVerifier, mx.array, int]:
        """Hold ``prompt[:-1]``; returns the verifier, the last hidden state and the reused count."""
        prefix = list(prompt[:-1])
        shared = 0
        for a, b in zip(prefix, self.tokens, strict=False):
            if a != b:
                break
            shared += 1
        best = max((s for s in self._snapshots if s.count <= shared), key=lambda s: s.count,
                   default=None)
        taps = self.drafter.taps
        if best is None or self.cache is None:
            self.reset()
            self.cache = self.layout.make_cache()
            verifier = TreeVerifier(self.layout, self.cache, taps)
            hidden = verifier.prefill(prefix)
            self.drafter.prefill(prompt, verifier.fused if taps else hidden)
            self.tokens = prefix
            last, reused = hidden[-1], 0
        else:
            self._restore(self.cache, best)
            verifier = TreeVerifier(self.layout, self.cache, taps)
            last, reused = best.hidden, best.count
            rest = prefix[best.count:]
            if rest:
                hidden = verifier.prefill(rest)
                self.drafter.accept(rest, verifier.fused if taps else hidden)
                self.tokens.extend(rest)
                last = hidden[-1]
        self.snapshot(last)
        return verifier, last, reused


def accept(tree: Tree, logits: mx.array, sampling: Sampling, rule: Rule) -> Accepted:
    """Draw from the target at every node and keep the path ``rule`` allows."""
    if sampling.greedy:
        drawn = logits.argmax(axis=-1).tolist()
        return walk(tree, drawn, None, None, rule)
    tokens, ids, probs = sample_rows(logits, sampling, stats=rule.name != "lossless")
    mx.eval(tokens, ids, probs)
    if rule.name == "lossless":
        return walk(tree, tokens.tolist(), None, None, rule)
    return walk(tree, tokens.tolist(), np.array(ids), np.array(probs), rule)


def decode(session: Session, prompt: Sequence[int], asked: Asked,
           on_tokens: Callable[[list[int]], None] | None = None,
           stop: Callable[[], bool] | None = None) -> Decoded:
    """Generate after ``prompt`` until an end token, ``asked.max_tokens`` or ``stop()``."""
    max_tokens, sampling, rule, eos = asked.max_tokens, asked.sampling, asked.rule, asked.eos
    began = time.perf_counter()
    verifier, hidden, reused = session.prepare(prompt)
    out = Decoded(prompt_tokens=len(prompt), reused=reused,
                  prefill_s=time.perf_counter() - began)
    drafter, root = session.drafter, int(prompt[-1])
    taps = drafter.taps
    while len(out.tokens) < max_tokens:
        if stop is not None and stop():
            out.finish = "interrupted"
            break
        started = time.perf_counter()
        tree = drafter.draft(root, hidden)
        logits, states = verifier.forward(tree)
        kept = accept(tree, logits, sampling, rule)
        verifier.commit(kept.path)
        rows = mx.array(kept.path)
        drafter.accept([tree.tokens[i] for i in kept.path],
                       verifier.fused[rows] if taps and verifier.fused is not None
                       else states[rows])
        session.tokens.extend(tree.tokens[i] for i in kept.path)
        hidden = states[kept.path[-1]]
        new = [tree.tokens[i] for i in kept.path[1:]] + [kept.bonus]
        emitted: list[int] = []
        for token in new[:max_tokens - len(out.tokens)]:
            emitted.append(token)
            if token in eos:
                out.finish = "stop"
                break
        out.tokens.extend(emitted)
        if on_tokens is not None:
            on_tokens(emitted)
        out.passes.append(Pass(len(emitted), tree.n, time.perf_counter() - started, kept.forced))
        if out.finish == "stop":
            break
        root = kept.bonus
    session.snapshot(hidden)
    return out
