"""Getting a backend, and choosing one when the caller has no preference.

The singletons here are guarded by an explicit re-entrancy check rather than
``functools.lru_cache``. The factories import framework modules, those imports can
traverse package ``__init__`` code that calls the factory again mid-build, and a re-entered
``lru_cache`` mints a *second* instance. Two instances is not merely wasteful: modules then
bind devices on different objects, and factories start creating tensors on the wrong
device. The post-build check keeps exactly one instance no matter who wins the race.
"""

from __future__ import annotations

import os
import sys

from mainspring.backend.ops import ArrayBackend

BACKENDS = ("mlx", "torch")

_MLX: ArrayBackend | None = None
_TORCH: ArrayBackend | None = None


class BackendUnavailable(RuntimeError):
    """The requested backend cannot be imported on this machine."""


def torch_backend() -> ArrayBackend:
    global _TORCH
    if _TORCH is not None:
        return _TORCH

    from mainspring.backend.torch_ops import (
        TorchArrayOps,
        build_scatter_add,
        build_segment_sum,
        require_torch,
    )

    torch, nn = require_torch()
    built = ArrayBackend(
        name="torch",
        ops=TorchArrayOps(torch),
        scatter_add=build_scatter_add(torch),
        segment_sum=build_segment_sum(torch),
        make_linear=lambda d_in, d_out: nn.Linear(int(d_in), int(d_out)),
        cumsum=lambda x, axis=-1: torch.cumsum(x, dim=axis),
        cumprod=lambda x, axis=-1: torch.cumprod(x, dim=axis),
        rfft_abs=lambda x, axis=-1: torch.fft.rfft(x, dim=axis).abs(),
    )
    if _TORCH is None:  # an inner, re-entrant call may have won
        _TORCH = built
    return _TORCH


def mlx_backend() -> ArrayBackend:
    global _MLX
    if _MLX is not None:
        return _MLX

    from mainspring.backend.mlx_ops import (
        build_make_linear,
        build_scatter_add,
        build_segment_sum,
        require_mlx,
    )

    mx, nn = require_mlx()
    built = ArrayBackend(
        name="mlx",
        ops=mx,
        scatter_add=build_scatter_add(mx),
        segment_sum=build_segment_sum(mx),
        make_linear=build_make_linear(nn),
        cumsum=lambda x, axis=-1: mx.cumsum(x, axis=axis),
        cumprod=lambda x, axis=-1: mx.cumprod(x, axis=axis),
        rfft_abs=lambda x, axis=-1: mx.abs(mx.fft.rfft(x, axis=axis)),
    )
    if _MLX is None:
        _MLX = built
    return _MLX


def available() -> list[str]:
    """Backend names that can actually be imported here, in preference order."""
    found: list[str] = []
    for name in BACKENDS:
        try:
            get_backend(name)
        except (BackendUnavailable, RuntimeError):
            continue
        found.append(name)
    return found


def detect_backend() -> str:
    """The backend to use when the caller has no preference.

    ``$MAINSPRING_BACKEND`` overrides everything, so a comparison run can pin one without
    editing code. Otherwise: MLX on Apple silicon, torch elsewhere.

    Raises rather than returning a name that cannot be imported. A backend chosen by
    platform guess and then found missing produces an import error somewhere far from
    here, which is a much worse place to read it.
    """
    override = os.environ.get("MAINSPRING_BACKEND")
    if override:
        if override not in BACKENDS:
            raise BackendUnavailable(
                f"MAINSPRING_BACKEND={override!r} is not one of {BACKENDS}"
            )
        return override

    order = ("mlx", "torch") if sys.platform == "darwin" else ("torch", "mlx")
    for name in order:
        try:
            get_backend(name)
        except (BackendUnavailable, RuntimeError):
            continue
        return name

    raise BackendUnavailable(
        "neither MLX nor PyTorch could be imported. Install one: "
        "`pip install torch`, or `pip install mlx` on Apple silicon."
    )


def get_backend(name: str | None = None) -> ArrayBackend:
    """The backend called ``name``, or the detected default."""
    if name is None:
        name = detect_backend()
    if name == "torch":
        return torch_backend()
    if name == "mlx":
        return mlx_backend()
    raise BackendUnavailable(f"unknown backend {name!r}; expected one of {BACKENDS}")


def reset() -> None:
    """Drop the cached singletons. For tests only."""
    global _MLX, _TORCH
    _MLX = None
    _TORCH = None
