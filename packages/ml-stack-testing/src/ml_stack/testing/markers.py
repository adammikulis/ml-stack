"""Skip markers for tests that need a framework this platform may not have.

The distinction that matters: MLX ships **only** for Apple silicon, so its absence on
Linux is a platform fact, not a broken environment. Torch missing usually is a broken
environment. Both are skipped rather than failed, but the reason string says which, so a
CI log does not read as "these tests were never run" when it means "these tests cannot run
here".
"""

from __future__ import annotations

import importlib.util

import pytest


def _probe(name: str) -> tuple[bool, str]:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        found = False
    return found, "" if found else f"{name} is not importable on this platform"


HAVE_TORCH, _TORCH_WHY = _probe("torch")
HAVE_MLX, _MLX_WHY = _probe("mlx.core")

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason=_TORCH_WHY or "torch missing")
needs_mlx = pytest.mark.skipif(
    not HAVE_MLX,
    reason=_MLX_WHY or "mlx missing (it ships only for Apple silicon)",
)


def needs_both(fn):
    """Require torch *and* mlx. A parity test is meaningless without both."""
    return needs_torch(needs_mlx(fn))
