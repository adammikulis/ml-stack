"""One array API over MLX and PyTorch, so model math is written once.

Lab tier: needs MLX or PyTorch.

Write shared math against ``ArrayOps`` and take an ``ArrayBackend`` as a parameter:

    def rms_norm(backend, x, weight, eps=1e-6):
        ops = backend.ops
        scale = ops.rsqrt(ops.mean(x * x, axis=-1, keepdims=True) + eps)
        return x * scale * weight

The same function then runs on both frameworks, and ``ml_stack.testing`` proves the two
agree numerically -- forward and backward.
"""

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
    BackendUnavailable,
    available,
    detect_backend,
    get_backend,
    mlx_backend,
    torch_backend,
)

__all__ = [
    "BACKENDS",
    "ArrayBackend",
    "ArrayOps",
    "BackendUnavailable",
    "DeviceProfile",
    "SeedReport",
    "Vendor",
    "available",
    "bind_device",
    "detect_backend",
    "detect_device",
    "get_backend",
    "mlx_backend",
    "resolve_torch_device",
    "set_seeds",
    "torch_backend",
]
