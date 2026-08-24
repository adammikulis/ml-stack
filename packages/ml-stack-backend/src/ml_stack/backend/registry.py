"""Getting a backend, and choosing one when the caller has no preference."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from ml_stack.backend.ops import ArrayBackend

_FACTORIES: dict[str, "Callable[[], ArrayBackend]"] = {}
_BUILT: dict[str, ArrayBackend] = {}

REGISTRY_GROUP = "ml_stack.backend"
"""Entry-point group a third-party backend registers itself under."""


def register(name: str, factory: "Callable[[], ArrayBackend]", *,
             replace: bool = False) -> None:
    """Make a backend available under ``name``."""
    if name in _FACTORIES and not replace:
        raise BackendUnavailable(
            f"a backend named {name!r} is already registered; pass replace=True "
            "if shadowing it is deliberate")
    _FACTORIES[name] = factory
    _BUILT.pop(name, None)


def backends() -> tuple[str, ...]:
    """Every registered backend name, whether or not it can be imported here."""
    _load_plugins()
    return tuple(sorted(_FACTORIES))

class BackendUnavailable(RuntimeError):
    """The requested backend cannot be imported on this machine."""


_PLUGINS_LOADED = False


def _load_plugins() -> None:
    """Pull in any backend a third-party package registered. Once, and best effort."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        from importlib.metadata import entry_points
        found = entry_points(group=REGISTRY_GROUP)
    except Exception:                                 # noqa: BLE001
        return
    for ep in found:
        try:
            register(ep.name, ep.load())
        except Exception:                             # noqa: BLE001
            # A broken plugin costs its own backend, not every other one.
            continue


def torch_backend() -> ArrayBackend:

    from ml_stack.backend.torch_ops import (
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
    return built


def mlx_backend() -> ArrayBackend:

    from ml_stack.backend.mlx_ops import (
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
    return built


def available() -> list[str]:
    """Backend names that can actually be imported here, in preference order."""
    found: list[str] = []
    for name in backends():
        try:
            get_backend(name)
        except Exception:                             # noqa: BLE001
            continue
        found.append(name)
    return found


def detect_backend() -> str:
    """The backend to use when the caller has no preference."""
    override = os.environ.get("ML_STACK_BACKEND")
    if override:
        if override not in backends():
            raise BackendUnavailable(
                f"ML_STACK_BACKEND={override!r} is not one of {backends()}"
            )
        return override

    known = backends()
    preferred = ("mlx", "torch") if sys.platform == "darwin" else ("torch", "mlx")
    order = [n for n in preferred if n in known] + [n for n in known if n not in preferred]
    for name in order:
        try:
            get_backend(name)
        except (BackendUnavailable, RuntimeError):
            continue
        return name

    raise BackendUnavailable(
        f"none of {backends()} could be imported. Install one: `pip install torch` "
        "(which covers both CUDA and ROCm), or `pip install mlx` on Apple silicon."
    )


def get_backend(name: str | None = None) -> ArrayBackend:
    """The backend called ``name``, or the detected default."""
    if name is None:
        name = detect_backend()
    _load_plugins()
    built = _BUILT.get(name)
    if built is not None:
        return built
    factory = _FACTORIES.get(name)
    if factory is None:
        raise BackendUnavailable(
            f"unknown backend {name!r}; expected one of {backends()}")
    made = factory()
    if name not in _BUILT:
        _BUILT[name] = made
    return _BUILT[name]


def reset() -> None:
    """Drop the cached singletons. For tests only."""
    _BUILT.clear()


register("mlx", lambda: mlx_backend())
register("torch", lambda: torch_backend())

BACKENDS = ("mlx", "torch")
"""The backends that ship with this package. Prefer ``backends()``, which also sees"""
