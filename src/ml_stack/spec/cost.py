"""What verifying n tree nodes costs on this machine, relative to one, measured once and cached."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ml_stack import home
from ml_stack.files import read_json, write_json
from ml_stack.spec.layout import Layout
from ml_stack.spec.tree import Tree
from ml_stack.spec.verify import TreeVerifier

__all__ = ["SIZES", "calibration_tree", "load_curve", "measure"]

SIZES = tuple(range(1, 33))
CURVE_VERSION = 1


def low_power() -> bool:
    """Whether macOS Low Power Mode is on."""
    try:
        said = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5,
                              check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.split()[:2] == ["lowpowermode", "1"] for line in said.splitlines())


def calibration_tree(count: int, rng: np.random.Generator, vocab: int) -> Tree:
    """A chain of up to 16 nodes with the rest hanging off it, like a drafter's trees."""
    parent = [-1] + [i - 1 if i <= 16 else i - 16 for i in range(1, count)]
    return Tree([int(t) for t in rng.integers(0, vocab, count)], parent)


def _nondecreasing(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m1, c1), (m2, c2) = blocks[-2], blocks.pop()
            blocks[-1] = [(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2]
    return [mean for mean, size in blocks for _ in range(int(size))]


def measure(layout: Layout, sizes: tuple[int, ...] = SIZES, reps: int = 3,
            prefix: int = 64) -> dict[int, float]:
    """Milliseconds per verification pass at each node count, best of ``reps``, non-decreasing."""
    rng = np.random.default_rng(0)
    vocab = int(layout.args.vocab_size)
    verifier = TreeVerifier(layout, layout.make_cache())
    verifier.prefill([int(t) for t in rng.integers(0, vocab, prefix)])

    def timed(count: int) -> float:
        mx.synchronize()
        began = time.perf_counter()
        logits, hidden = verifier.forward(calibration_tree(count, rng, vocab))
        mx.eval(logits, hidden)
        seconds = time.perf_counter() - began
        verifier.commit([0])
        return seconds

    last = timed(max(sizes))
    for _ in range(8):
        now = timed(max(sizes))
        if abs(now - last) < 0.03 * last:
            break
        last = now
    times: dict[int, list[float]] = {n: [] for n in sizes}
    for _ in range(reps):
        for count in sizes:
            times[count].append(timed(count))
    ordered = sorted(sizes)
    best = _nondecreasing([1000 * min(times[n]) for n in ordered])
    return dict(zip(ordered, best, strict=True))


def curve_file(name: str) -> Path:
    info = mx.device_info()
    power = "-lowpower" if low_power() else ""
    return home.cache("spec", "cost", f"{info['architecture']}-mlx{mx.__version__}{power}-{name}.json")


def load_curve(layout: Layout, name: str, *, again: bool = False) -> dict[int, float]:
    """``{n: cost of n nodes / cost of 1}`` for this machine and model, measured on first use."""
    where = curve_file(name)
    kept = read_json(where, None) if not again else None
    if isinstance(kept, dict) and kept.get("version") == CURVE_VERSION:
        held = kept["ms"]
    else:
        held = {str(k): v for k, v in measure(layout).items()}
        where.parent.mkdir(parents=True, exist_ok=True)
        write_json(where, {"version": CURVE_VERSION, "device": mx.device_info()["device_name"],
                           "measured": time.strftime("%Y-%m-%dT%H:%M"), "ms": held})
    one = float(held["1"])
    return {int(k): float(v) / one for k, v in held.items()}
