"""Every operation the array API declares, run on two backends and compared."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_stack.testing.parity import assert_forward_parity, inputs
from ml_stack.train.backend.registry import BackendUnavailable, get_backend

Tensor = Any

ATOL = 1e-6
"""Tolerance for one operation on one input: fp32 rounding, not a different algorithm."""

_X = inputs((4, 5), seed=0)
_W = inputs((5, 3), seed=1)
_ROW_INDEX = np.array([2, 0, 3, 1], dtype=np.int64)
_COLUMN_INDEX = np.array([0, 2, 0, 1, 2], dtype=np.int64)
_NESTED_INDEX = np.array([[2], [0], [3], [1]], dtype=np.int64)
_MIXED = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
_HALVES = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)


def _x(ops: Any) -> Tensor:
    return ops.array(_X)


def _w(ops: Any) -> Tensor:
    return ops.array(_W)


CASES: dict[str, Callable[[Any, Any], Tensor]] = {
    # Factories
    "array": lambda b, o: o.array(_X),
    "zeros": lambda b, o: o.zeros((3, 4), dtype=o.float32),
    "ones": lambda b, o: o.ones((3, 4), dtype=o.float32),
    "zeros_like": lambda b, o: o.zeros_like(_x(o)),
    "ones_like": lambda b, o: o.ones_like(_x(o)),
    "arange": lambda b, o: o.arange(0, 10, dtype=o.float32),
    "eye": lambda b, o: o.eye(4),
    # Shape
    "reshape": lambda b, o: o.reshape(_x(o), (2, 10)),
    "stack(axis=0)": lambda b, o: o.stack([_x(o), _x(o)], 0),
    "stack(axis=1)": lambda b, o: o.stack([_x(o), _x(o)], 1),
    "stack(axis=-1)": lambda b, o: o.stack([_x(o), _x(o)], -1),
    "concatenate(axis=0)": lambda b, o: o.concatenate([_x(o), _x(o)], 0),
    "concatenate(axis=-1)": lambda b, o: o.concatenate([_x(o), _x(o)], -1),
    "tile": lambda b, o: o.tile(_x(o), (2, 3)),
    "broadcast_to": lambda b, o: o.broadcast_to(o.array(_X[:1]), (4, 5)),
    "swapaxes": lambda b, o: o.swapaxes(_x(o), 0, 1),
    # Indexing
    "take(axis=0)": lambda b, o: o.take(_x(o), o.array(_ROW_INDEX, dtype=o.int32), 0),
    "take(axis=1)": lambda b, o: o.take(_x(o), o.array(_COLUMN_INDEX, dtype=o.int32), 1),
    "take(nested index)": lambda b, o: o.take(_x(o), o.array(_NESTED_INDEX, dtype=o.int32), 0),
    "argsort(axis=-1)": lambda b, o: o.argsort(_x(o), -1),
    "argsort(axis=0)": lambda b, o: o.argsort(_x(o), 0),
    "argmin(axis=-1)": lambda b, o: o.argmin(_x(o), -1),
    "argmin(axis=0)": lambda b, o: o.argmin(_x(o), 0),
    "take_along_axis": lambda b, o: o.take_along_axis(_x(o), o.argsort(_x(o), -1), -1),
    "where": lambda b, o: o.where(_x(o) > 0, _x(o), o.zeros_like(_x(o))),
    "stop_gradient": lambda b, o: o.stop_gradient(_x(o)),
    # Elementwise
    "sigmoid": lambda b, o: o.sigmoid(_x(o)),
    "tanh": lambda b, o: o.tanh(_x(o)),
    "sin": lambda b, o: o.sin(_x(o)),
    "cos": lambda b, o: o.cos(_x(o)),
    "sqrt": lambda b, o: o.sqrt(o.abs(_x(o))),
    "rsqrt": lambda b, o: o.rsqrt(o.abs(_x(o)) + 0.5),
    "erf": lambda b, o: o.erf(_x(o)),
    "exp": lambda b, o: o.exp(_x(o)),
    "log": lambda b, o: o.log(o.abs(_x(o)) + 0.1),
    "log1p": lambda b, o: o.log1p(o.abs(_x(o))),
    "reciprocal": lambda b, o: o.reciprocal(o.abs(_x(o)) + 0.5),
    "abs": lambda b, o: o.abs(_x(o)),
    "floor": lambda b, o: o.floor(_x(o) * 3.0),
    "round": lambda b, o: o.round(_x(o) * 3.0),
    "round(halves)": lambda b, o: o.round(o.array(_HALVES)),
    "clip": lambda b, o: o.clip(_x(o), -0.5, 0.5),
    "maximum": lambda b, o: o.maximum(_x(o), o.zeros_like(_x(o))),
    "minimum": lambda b, o: o.minimum(_x(o), o.zeros_like(_x(o))),
    "nan_to_num": lambda b, o: o.nan_to_num(o.array(_MIXED)),
    # Linear algebra
    "matmul": lambda b, o: o.matmul(_x(o), _w(o)),
    "softmax(axis=-1)": lambda b, o: o.softmax(_x(o), -1),
    "softmax(axis=0)": lambda b, o: o.softmax(_x(o), 0),
    "einsum": lambda b, o: o.einsum("ij,jk->ik", _x(o), _w(o)),
    "repeat": lambda b, o: o.repeat(_x(o), 3, 1),
    # Reductions
    "sum": lambda b, o: o.sum(_x(o)),
    "sum(axis=-1)": lambda b, o: o.sum(_x(o), -1),
    "sum(axis=0,keepdims)": lambda b, o: o.sum(_x(o), 0, True),
    "mean": lambda b, o: o.mean(_x(o)),
    "mean(axis=-1)": lambda b, o: o.mean(_x(o), -1),
    "mean(axis=0,keepdims)": lambda b, o: o.mean(_x(o), 0, True),
    "max": lambda b, o: o.max(_x(o)),
    "max(axis=-1)": lambda b, o: o.max(_x(o), -1),
    "max(axis=0,keepdims)": lambda b, o: o.max(_x(o), 0, True),
    "min": lambda b, o: o.min(_x(o)),
    "min(axis=-1)": lambda b, o: o.min(_x(o), -1),
    "min(axis=0,keepdims)": lambda b, o: o.min(_x(o), 0, True),
    "logsumexp(axis=-1)": lambda b, o: o.logsumexp(_x(o), -1),
    "logsumexp(axis=0)": lambda b, o: o.logsumexp(_x(o), 0),
    # Beyond the protocol: what ArrayBackend carries beside ops
    "scatter_add(axis=0)": lambda b, o: b.scatter_add(
        o.zeros((4, 5), dtype=o.float32), o.array(_ROW_INDEX, dtype=o.int32), _x(o), 0),
    "scatter_add(axis=1)": lambda b, o: b.scatter_add(
        o.zeros((4, 5), dtype=o.float32), o.array(_COLUMN_INDEX, dtype=o.int32), _x(o), 1),
    "segment_sum(axis=0)": lambda b, o: b.segment_sum(
        _x(o), o.array(_ROW_INDEX, dtype=o.int32), 4, 0),
    "segment_sum(axis=1)": lambda b, o: b.segment_sum(
        _x(o), o.array(_COLUMN_INDEX, dtype=o.int32), 3, 1),
    "cumsum": lambda b, o: b.cumsum(_x(o), -1),
    "cumprod": lambda b, o: b.cumprod(_x(o), -1),
    "rfft_abs": lambda b, o: b.rfft_abs(_x(o), -1),
}
"""Operation name -> ``(backend, backend.ops) -> tensor``."""


@dataclass
class OpResult:
    """One operation on two backends: how far apart, and whether that is within ATOL."""

    name: str
    diff: float = 0.0
    ok: bool = True
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.name} {self.diff:.3e} {'ok' if self.ok else 'FAIL'}"


def as_numpy(value: Tensor) -> np.ndarray:
    """A backend tensor as a float64 NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(np.asarray(value), dtype=np.float64)


def check_op(name: str, first: Any, second: Any, *, atol: float = ATOL) -> OpResult:
    """Run one case on both backends and compare."""
    try:
        outputs = [as_numpy(CASES[name](b, b.ops)) for b in (first, second)]
    except (ArithmeticError, AttributeError, IndexError, KeyError, RuntimeError,
            TypeError, ValueError) as exc:
        return OpResult(name, float("nan"), False, f"{type(exc).__name__}: {exc}")

    try:
        diff = assert_forward_parity(*outputs, atol=atol, rtol=atol, label=name)
    except AssertionError as exc:
        a, b = outputs
        worst = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else float("nan")
        return OpResult(name, worst, False, str(exc).replace("\n", " "))
    return OpResult(name, diff, True)


def check_all(first: Any, second: Any, *, atol: float = ATOL) -> list[OpResult]:
    """Every case in ``CASES``, in declaration order."""
    return [check_op(name, first, second, atol=atol) for name in CASES]


def table(results: list[OpResult], *, first: str = "torch", second: str = "mlx") -> str:
    """The results as a plain-text table, in case order, with a count at the end."""
    width = max([len(r.name) for r in results] + [len("operation")])
    lines = [f"{'operation':<{width}}  {'max |Δ|':>10}  result",
             f"{'-' * width}  {'-' * 10}  ------"]
    for r in results:
        shown = "-" if r.diff != r.diff else f"{r.diff:.3e}"
        lines.append(f"{r.name:<{width}}  {shown:>10}  {'pass' if r.ok else 'FAIL'}")
        if r.detail:
            lines.append(f"{'':<{width}}  {'':>10}  {r.detail}")
    failed = [r for r in results if not r.ok]
    lines.append("")
    lines.append(f"{first} vs {second}: {len(results) - len(failed)}/{len(results)} agree "
                 f"within atol={ATOL:g}")
    if failed:
        lines.append(f"disagreeing: {', '.join(r.name for r in failed)}")
    return "\n".join(lines)


def report(*, say: Callable[[str], None] = print, first: str = "torch",
           second: str = "mlx") -> int:
    """Print the table for two named backends. 0 if all agree, 1 if any do not, 2 if one
    of the two cannot be built here."""
    built = []
    for name in (first, second):
        try:
            built.append(get_backend(name))
        except (BackendUnavailable, RuntimeError) as exc:
            say(f"{name} is not usable here, so there is nothing to compare: {exc}")
            return 2

    results = check_all(*built)
    say(table(results, first=first, second=second))
    return 1 if any(not r.ok for r in results) else 0
