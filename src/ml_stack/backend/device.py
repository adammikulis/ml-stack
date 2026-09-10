"""What hardware is this, and how much of it can a job have?"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum

_GIB = 1024**3


class Vendor(StrEnum):
    APPLE = "apple"
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """What was detected. Log this next to any measurement."""

    vendor: Vendor
    name: str
    total_memory_gb: float
    unified_memory: bool
    """True when the GPU shares system RAM, so 'VRAM' and 'RAM' are the same pool."""

    def __str__(self) -> str:
        return f"{self.name} ({self.vendor}, {self.total_memory_gb:.1f} GB)"

    def budget_gb(self, *, fraction: float = 0.7, reserve_gb: float = 2.0) -> float:
        """How much memory a single job may claim."""
        return max(0.0, self.total_memory_gb * fraction - reserve_gb)


def detect_device() -> DeviceProfile:
    """Detect once. Cheap enough to call at startup, not per step."""
    for probe in (_detect_apple, _detect_nvidia, _detect_amd):
        found = probe()
        if found is not None:
            return found
    return DeviceProfile(Vendor.CPU, platform.processor() or "cpu", _system_ram_gb(), False)


def resolve_torch_device(prefer: str | None = None):
    """A ``torch.device`` for this machine."""
    from ml_stack.backend.torch_ops import require_torch

    torch, _ = require_torch()

    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _system_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / _GIB
    except ImportError:
        pass
    try:  # POSIX fallback, so this works without psutil
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GIB
    except (ValueError, OSError, AttributeError):
        return 0.0


def _sysctl(key: str) -> str | None:
    if platform.system() != "Darwin" or not shutil.which("sysctl"):
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _detect_apple() -> DeviceProfile | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    name = _sysctl("machdep.cpu.brand_string") or "Apple silicon"
    # Unified memory: the GPU has no separate pool, so system RAM is the budget.
    return DeviceProfile(Vendor.APPLE, name, _system_ram_gb(), unified_memory=True)


def _detect_nvidia() -> DeviceProfile | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None

    first = out.stdout.strip().splitlines()[0]
    name, _, mib = first.partition(",")
    try:
        total_gb = float(mib.strip()) / 1024
    except ValueError:
        total_gb = 0.0
    return DeviceProfile(Vendor.NVIDIA, name.strip(), total_gb, unified_memory=False)


def _detect_amd() -> DeviceProfile | None:
    if not shutil.which("rocm-smi"):
        return None
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    name = next(
        (ln.split(":", 1)[1].strip() for ln in out.stdout.splitlines() if ":" in ln),
        "AMD GPU",
    )
    return DeviceProfile(Vendor.AMD, name, _system_ram_gb(), unified_memory=False)
