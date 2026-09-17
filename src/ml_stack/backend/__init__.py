"""One array API over MLX and PyTorch, so model math is written once."""

from __future__ import annotations

from ml_stack.backend.determinism import SeedReport, set_seeds
from ml_stack.backend.device import (
    DeviceProfile,
    Vendor,
    detect_device,
    resolve_torch_device,
)
from ml_stack.backend.ops import ArrayBackend, ArrayOps, bind_device
from ml_stack.backend.registry import (
    BACKENDS,
    REGISTRY_GROUP,
    BackendUnavailable,
    available,
    backends,
    detect_backend,
    get_backend,
    mlx_backend,
    register,
    torch_backend,
)

__all__ = [
    "BACKENDS",
    "REGISTRY_GROUP",
    "ArrayBackend",
    "ArrayOps",
    "BackendUnavailable",
    "DeviceProfile",
    "SeedReport",
    "Vendor",
    "available",
    "backends",
    "bind_device",
    "detect_backend",
    "detect_device",
    "get_backend",
    "mlx_backend",
    "register",
    "resolve_torch_device",
    "set_seeds",
    "torch_backend",
]
