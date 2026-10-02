"""What hardware is this, and how much of it can a job have?"""

from __future__ import annotations

import functools
import hashlib
import json
import os
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
    """Memory a model can live in: system RAM when unified, the card's own otherwise."""
    unified_memory: bool
    """True when the GPU shares system RAM, so 'VRAM' and 'RAM' are the same pool."""
    gpu_cores: int = 0
    compute_capability: str = ""
    """``metal4``, ``sm_89``, ``gfx``, or ``cpu``."""
    driver_version: str = ""
    cpu_model: str = ""
    ram_gb: float = 0.0

    def __str__(self) -> str:
        return f"{self.name} ({self.vendor}, {self.total_memory_gb:.1f} GB)"

    @property
    def label(self) -> str:
        """``Apple M4 Max (40-core GPU)``."""
        return f"{self.name} ({self.gpu_cores}-core GPU)" if self.gpu_cores else self.name

    @property
    def fingerprint(self) -> str:
        """Compute identity: the chip and its core count, or the card and the CPU beside it."""
        raw = (f"{self.name}|{self.gpu_cores}" if self.vendor is Vendor.APPLE
               else f"{self.name}|{self.cpu_model}")
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def budget_gb(self, *, fraction: float = 0.7, reserve_gb: float = 2.0) -> float:
        """How much memory a single job may claim."""
        return max(0.0, self.total_memory_gb * fraction - reserve_gb)

    def free_gb(self) -> float:
        """Memory a model could take right now, measured."""
        if self.vendor is Vendor.NVIDIA:
            free = _nvidia_query("memory.free")
            if free is not None:
                return float(free) / 1024
        if self.unified_memory or self.vendor is Vendor.CPU:
            return _available_ram_bytes() / _GIB
        return self.total_memory_gb


@functools.cache
def detect_device() -> DeviceProfile:
    """Detect once per process."""
    for probe in (_detect_apple, _detect_nvidia, _detect_amd):
        found = probe()
        if found is not None:
            return found
    ram = _system_ram_gb()
    cpu = _cpu_model()
    return DeviceProfile(Vendor.CPU, cpu, ram, False, compute_capability="cpu",
                         cpu_model=cpu, ram_gb=ram)


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
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GIB
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / _GIB
    except ImportError:
        return 0.0


def _available_ram_bytes() -> int:
    try:
        import metal_smi

        return int(metal_smi.system_stats()["memory_available"])
    except ImportError:
        pass
    import psutil

    return int(psutil.virtual_memory().available)


def _cpu_model() -> str:
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()


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


def _metal_version() -> str:
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "metal"
    if out.returncode != 0:
        return "metal"
    for display in json.loads(out.stdout).get("SPDisplaysDataType", []):
        family = str(display.get("spdisplays_mtlgpufamilysupport", ""))
        if family:
            return family.removeprefix("spdisplays_")
    return "metal"


def _detect_apple() -> DeviceProfile | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    cpu = _cpu_model()
    name, cores = cpu or "Apple silicon", 0
    try:
        import metal_smi

        gpu = metal_smi.system_gpu_stats()
        name = str(gpu.get("model") or name)
        cores = int(gpu.get("gpu_core_count") or 0)
    except ImportError:
        pass
    ram = _system_ram_gb()
    # Unified memory: the GPU has no separate pool, so system RAM is the budget.
    return DeviceProfile(Vendor.APPLE, name, ram, unified_memory=True, gpu_cores=cores,
                         compute_capability=_metal_version(), cpu_model=cpu, ram_gb=ram)


def _nvidia_query(fields: str) -> str | None:
    """The first GPU's answer to ``nvidia-smi --query-gpu``, or None without a card."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip().splitlines()[0]


def _detect_nvidia() -> DeviceProfile | None:
    first = _nvidia_query("name,memory.total,compute_cap,driver_version")
    if first is None:
        return None
    name, mib, cap, driver = [*(part.strip() for part in first.split(",")), "", "", ""][:4]
    try:
        total_gb = float(mib) / 1024
    except ValueError:
        total_gb = 0.0
    return DeviceProfile(Vendor.NVIDIA, name, total_gb, unified_memory=False,
                         compute_capability=f"sm_{cap.replace('.', '')}" if cap else "",
                         driver_version=driver, cpu_model=_cpu_model(),
                         ram_gb=_system_ram_gb())


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
    ram = _system_ram_gb()
    return DeviceProfile(Vendor.AMD, name, ram, unified_memory=False,
                         compute_capability="gfx", cpu_model=_cpu_model(), ram_gb=ram)
