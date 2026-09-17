"""Skip markers for tests that need a framework this platform may not have."""

from __future__ import annotations

import importlib.util

import pytest


def _probe(name: str, extra: str) -> tuple[bool, str]:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        found = False
    return found, "" if found else f"{name} is not importable: {extra}"


HAVE_TORCH, _TORCH_WHY = _probe("torch", "ml-stack[torch]")
HAVE_MLX, _MLX_WHY = _probe("mlx.core", "ml-stack[mlx], Apple silicon only")

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason=_TORCH_WHY)
needs_mlx = pytest.mark.skipif(not HAVE_MLX, reason=_MLX_WHY)
needs_a_backend = pytest.mark.skipif(
    not (HAVE_TORCH or HAVE_MLX),
    reason="neither torch nor mlx is importable; install ml-stack[torch] or ml-stack[mlx]",
)


def needs_both(fn):
    """Require torch *and* mlx. A parity test is meaningless without both."""
    return needs_torch(needs_mlx(fn))
