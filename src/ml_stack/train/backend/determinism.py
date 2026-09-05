"""Seed every random source that could affect a run."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedReport:
    """Which sources were actually seeded. Worth logging next to a result."""

    seed: int
    strict: bool
    python: bool = True
    numpy: bool = False
    mlx: bool = False
    torch: bool = False
    torch_deterministic: bool = False

    def __str__(self) -> str:
        seeded = [n for n in ("python", "numpy", "mlx", "torch") if getattr(self, n)]
        note = " (strict)" if self.torch_deterministic else ""
        return f"seed={self.seed} -> {', '.join(seeded)}{note}"


def set_seeds(seed: int = 42, *, strict: bool = False) -> SeedReport:
    """Seed everything available. Returns what was actually seeded."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    numpy_ok = _seed_numpy(seed)
    mlx_ok = _seed_mlx(seed)
    torch_ok, torch_strict = _seed_torch(seed, strict)

    return SeedReport(
        seed=seed,
        strict=strict,
        numpy=numpy_ok,
        mlx=mlx_ok,
        torch=torch_ok,
        torch_deterministic=torch_strict,
    )


def _seed_numpy(seed: int) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    np.random.seed(seed)
    return True


def _seed_mlx(seed: int) -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    seeder = getattr(getattr(mx, "random", None), "seed", None)
    if not callable(seeder):
        return False
    seeder(seed)
    return True


def _seed_torch(seed: int, strict: bool) -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not strict:
        return True, False

    deterministic = False
    try:
        torch.use_deterministic_algorithms(True)
        deterministic = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    return True, deterministic
