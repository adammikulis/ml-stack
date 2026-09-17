"""Prove two backends compute the same thing -- forward and backward."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SEED = 42
"""Torch's global seed before a pair is built, so a pair's weights are the same every run."""

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


def parameter_name(name: str) -> str:
    """One spelling for both frameworks: MLX lists nest under ``layers``, torch's do not."""
    return name.replace(".layers.", ".").removeprefix("layers.")


def copy_torch_weights_to_mlx(torch_module: Any, mlx_module: Any) -> None:
    """Load a torch module's weights into an MLX module's trainable tree, matched by name."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    torch_params = {
        parameter_name(k): v.detach().cpu().numpy() for k, v in torch_module.state_dict().items()
    }
    mlx_flat = dict(tree_flatten(mlx_module.trainable_parameters()))
    by_name = {parameter_name(name): name for name in mlx_flat}

    only_torch = sorted(set(torch_params) - set(by_name))
    only_mlx = sorted(set(by_name) - set(torch_params))
    if only_torch or only_mlx:
        raise ParityError(
            "parameter trees diverge, so no comparison is possible:\n"
            f"  only in torch: {only_torch}\n"
            f"  only in mlx:   {only_mlx}"
        )
    wrong_shape = [
        f"{name}: torch {tuple(value.shape)} != mlx {tuple(mlx_flat[by_name[name]].shape)}"
        for name, value in torch_params.items()
        if tuple(value.shape) != tuple(mlx_flat[by_name[name]].shape)
    ]
    if wrong_shape:
        raise ParityError("parameter shapes diverge:\n  " + "\n  ".join(wrong_shape))

    mlx_module.update(tree_unflatten([(by_name[n], mx.array(v)) for n, v in torch_params.items()]))
    mx.eval(mlx_module.parameters())


def torch_grad_norms(module: Any) -> dict[str, float]:
    """Per-parameter gradient L2 norms after a backward pass; an untouched parameter is 0."""
    return {
        parameter_name(name): 0.0 if p.grad is None else float(p.grad.detach().norm().item())
        for name, p in module.named_parameters()
    }


def mlx_grad_norms(grads: Any) -> dict[str, float]:
    """Per-parameter gradient L2 norms from an MLX gradient tree."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    return {
        parameter_name(name): float(mx.sqrt(mx.sum(value * value)).item())
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
    """Compare two finite forward outputs. Returns the max absolute difference."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ParityError(f"{label}: shapes differ, {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ParityError(f"{label}: an output is not finite")

    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        worst = np.unravel_index(int(np.argmax(np.abs(a - b))), a.shape)
        raise ParityError(
            f"{label}: outputs differ by up to {diff:.3e} "
            f"(atol={atol}, rtol={rtol}); worst element at {worst}: {a[worst]} vs {b[worst]}"
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
    if not first:
        raise ParityError("no parameters, so no gradient was compared")

    compared: dict[str, tuple[float, float]] = {}
    failures: list[str] = []
    rows: list[str] = []
    for name in sorted(first):
        x, y = first[name], second[name]
        compared[name] = (x, y)
        rel = 0.0 if (x < zero and y < zero) else abs(x - y) / max(x, y)
        rows.append(f"  {name:36s} {names[0]}={x:12.6g} {names[1]}={y:12.6g} rel={rel:.3g}")
        if rel > rtol:
            failures.append(name)
    if failures:
        raise ParityError(
            f"gradient norms differ beyond rtol={rtol}: {failures}\n" + "\n".join(rows)
        )
    return compared


def run_pair(
    build_torch: Callable[[], Any],
    build_mlx: Callable[[], Any],
    torch_forward: Callable[[Any], Any],
    mlx_forward: Callable[[Any], Any],
    target_shape: tuple[int, ...],
    *,
    seed: int = 1,
    atol: float = FORWARD_ATOL,
    rtol: float = FORWARD_RTOL,
    grad_rtol: float = GRAD_NORM_RTOL,
    zero: float = ZERO_GRAD,
) -> ParityReport:
    """One set of weights, both forwards, both backwards of an MSE loss, compared.

    ``torch_forward(module)`` and ``mlx_forward(module)`` carry their own inputs and return
    the output compared. The loss is the mean squared error against a fixed target of
    ``target_shape``.
    """
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import torch

    torch.manual_seed(SEED)
    torch_module = build_torch()
    mlx_module = build_mlx()
    copy_torch_weights_to_mlx(torch_module, mlx_module)
    target = inputs(target_shape, seed=seed)

    torch_module.zero_grad(set_to_none=True)
    torch_out = torch_forward(torch_module)
    ((torch_out - torch.as_tensor(target, device=torch_out.device)) ** 2).mean().backward()

    def mlx_loss(module: Any, target_in: Any) -> Any:
        return mx.mean(mx.square(mlx_forward(module) - target_in))

    _value, grads = mlx_nn.value_and_grad(mlx_module, mlx_loss)(mlx_module, mx.array(target))
    mx.eval(grads)
    mlx_out = mlx_forward(mlx_module)
    mx.eval(mlx_out)

    report = ParityReport()
    report.max_forward_diff = assert_forward_parity(
        torch_out.detach().cpu().numpy(), np.asarray(mlx_out), atol=atol, rtol=rtol
    )
    report.grad_norms = assert_grad_parity(
        torch_grad_norms(torch_module), mlx_grad_norms(grads), rtol=grad_rtol, zero=zero
    )
    report.checked_parameters = len(report.grad_norms)
    return report
