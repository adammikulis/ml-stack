"""Prove two backends compute the same thing -- forward and backward."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FORWARD_ATOL = 1e-4
FORWARD_RTOL = 1e-4
GRAD_NORM_RTOL = 5e-2

ZERO_GRAD = 1e-7
"""Below this, a gradient norm is fp32 residue on an analytically dead parameter."""


class ParityError(AssertionError):
    """Two backends disagree by more than the tolerance allows."""


@dataclass
class ParityReport:
    """What was compared and how far apart it was."""

    max_forward_diff: float = 0.0
    grad_norms: dict[str, tuple[float, float]] = field(default_factory=dict)
    checked_parameters: int = 0

    def __str__(self) -> str:
        return (
            f"forward max|Δ|={self.max_forward_diff:.2e}, "
            f"{self.checked_parameters} parameter gradient(s) compared"
        )


def inputs(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Deterministic float32 test input. NumPy so both backends start from one array."""
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _normalize_mlx_name(name: str) -> str:
    """MLX's ``nn.Sequential`` nests parameters under ``layers``; torch's does not."""
    return name.replace(".layers.", ".")


def copy_torch_weights_to_mlx(torch_module: Any, mlx_module: Any) -> None:
    """Load a torch module's weights into an MLX module, matched by name."""
    from mlx.utils import tree_flatten, tree_unflatten

    torch_params = {k: v.detach().cpu().numpy() for k, v in torch_module.state_dict().items()}
    mlx_flat = tree_flatten(mlx_module.trainable_parameters())
    normalized = {_normalize_mlx_name(name): name for name, _ in mlx_flat}

    only_torch = sorted(set(torch_params) - set(normalized))
    only_mlx = sorted(set(normalized) - set(torch_params))
    if only_torch or only_mlx:
        raise ParityError(
            "parameter trees diverge, so no comparison is possible:\n"
            f"  only in torch: {only_torch}\n"
            f"  only in mlx:   {only_mlx}"
        )

    import mlx.core as mx

    updates = [(normalized[name], mx.array(value)) for name, value in torch_params.items()]
    mlx_module.update(tree_unflatten(updates))


def torch_grad_norms(module: Any) -> dict[str, float]:
    """Per-parameter gradient L2 norms after a backward pass."""
    return {
        name: float(p.grad.detach().norm().item())
        for name, p in module.named_parameters()
        if p.grad is not None
    }


def mlx_grad_norms(grads: Any) -> dict[str, float]:
    """Per-parameter gradient L2 norms from an MLX gradient tree."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    return {
        _normalize_mlx_name(name): float(mx.sqrt(mx.sum(value * value)).item())
        for name, value in tree_flatten(grads)
    }


def assert_forward_parity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    atol: float = FORWARD_ATOL,
    rtol: float = FORWARD_RTOL,
    label: str = "forward",
) -> float:
    """Compare two forward outputs. Returns the max absolute difference."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ParityError(f"{label}: shapes differ, {a.shape} vs {b.shape}")

    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        worst = np.unravel_index(int(np.argmax(np.abs(a - b))), a.shape) if a.size else ()
        raise ParityError(
            f"{label}: outputs differ by up to {diff:.3e} "
            f"(atol={atol}, rtol={rtol}); worst element at {worst}: "
            f"{a[worst] if a.size else '-'} vs {b[worst] if b.size else '-'}"
        )
    return diff


def assert_grad_parity(
    first: dict[str, float],
    second: dict[str, float],
    *,
    rtol: float = GRAD_NORM_RTOL,
    zero: float = ZERO_GRAD,
    names: tuple[str, str] = ("torch", "mlx"),
) -> dict[str, tuple[float, float]]:
    """Compare per-parameter gradient norms. Returns the pairs that were compared."""
    only_first = sorted(set(first) - set(second))
    only_second = sorted(set(second) - set(first))
    if only_first or only_second:
        raise ParityError(
            f"gradient trees diverge:\n"
            f"  only in {names[0]}: {only_first}\n"
            f"  only in {names[1]}: {only_second}"
        )

    compared: dict[str, tuple[float, float]] = {}
    for name in sorted(first):
        x, y = first[name], second[name]
        compared[name] = (x, y)

        if x < zero and y < zero:
            continue  # both analytically dead; the residue is noise, not signal
        if not np.isclose(x, y, rtol=rtol, atol=zero):
            raise ParityError(
                f"gradient norm for {name!r} differs: "
                f"{names[0]}={x:.6e} vs {names[1]}={y:.6e} (rtol={rtol})"
            )
    return compared


def run_pair(
    build_torch: Callable[[], Any],
    build_mlx: Callable[[], Any],
    torch_forward: Callable[[Any, np.ndarray], Any],
    mlx_forward: Callable[[Any, np.ndarray], Any],
    input_shape: tuple[int, ...],
    *,
    seed: int = 1,
    atol: float = FORWARD_ATOL,
    rtol: float = FORWARD_RTOL,
    grad_rtol: float = GRAD_NORM_RTOL,
) -> ParityReport:
    """The whole drill: one set of weights, both forwards, both backwards, compare."""
    import mlx.core as mx
    import mlx.nn as mlx_nn  # noqa: F401  (import proves MLX is usable before we build)
    import torch  # noqa: F401  (same, for torch: fail here, not inside build_torch)

    x = inputs(input_shape, seed=seed)

    torch_module = build_torch()
    mlx_module = build_mlx()
    copy_torch_weights_to_mlx(torch_module, mlx_module)

    torch_out = torch_forward(torch_module, x)
    mlx_out = mlx_forward(mlx_module, x)

    report = ParityReport()
    report.max_forward_diff = assert_forward_parity(
        torch_out.detach().cpu().numpy(),
        np.asarray(mlx_out),
        atol=atol,
        rtol=rtol,
    )

    torch_module.zero_grad(set_to_none=True)
    (torch_out**2).mean().backward()

    def mlx_loss(module: Any) -> Any:
        return mx.mean(mlx_forward(module, x) ** 2)

    _value, grads = mx.value_and_grad(mlx_loss)(mlx_module)
    mx.eval(grads)

    report.grad_norms = assert_grad_parity(
        torch_grad_norms(torch_module),
        mlx_grad_norms(grads),
        rtol=grad_rtol,
    )
    report.checked_parameters = len(report.grad_norms)
    return report
