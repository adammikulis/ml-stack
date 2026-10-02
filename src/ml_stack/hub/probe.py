"""What memory this machine has to give a model: RAM, a unified GPU pool, or a card's VRAM."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ml_stack import hub

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class Gpu:
    """One graphics card with its own memory."""

    name: str
    total_bytes: int
    free_bytes: int
    vendor: str = ""


@dataclass(frozen=True, slots=True)
class MachineMemory:
    """The memory a model can be placed in on this machine, in bytes.

    ``unified`` machines (Apple silicon) have one pool: ``gpu_limit_bytes`` is the most the
    GPU may wire out of it. Otherwise ``gpus`` lists each card's total and free memory and
    ``available_ram`` is what the CPU side can take.
    """

    total_ram: int = 0
    available_ram: int = 0
    unified: bool = False
    gpu_limit_bytes: int = 0
    gpus: tuple[Gpu, ...] = ()
    system: str = ""
    notes: tuple[str, ...] = ()

    @property
    def vram_free(self) -> int:
        return sum(g.free_bytes for g in self.gpus)

    @property
    def vram_total(self) -> int:
        return sum(g.total_bytes for g in self.gpus)

    def as_dict(self) -> dict[str, object]:
        return {"total_ram": self.total_ram, "available_ram": self.available_ram,
                "unified": self.unified, "gpu_limit_bytes": self.gpu_limit_bytes,
                "gpus": [{"name": g.name, "total_bytes": g.total_bytes,
                          "free_bytes": g.free_bytes, "vendor": g.vendor} for g in self.gpus],
                "system": self.system, "notes": list(self.notes)}


def _run(argv: list[str]) -> str:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def macos_available(text: str) -> int:
    """Bytes ``vm_stat`` output calls free, inactive, speculative or purgeable."""
    size = 4096
    pages: dict[str, int] = {}
    for line in text.splitlines():
        if "page size of" in line:
            size = int(line.split("page size of")[1].split()[0])
        name, _, count = line.partition(":")
        if count.strip().rstrip(".").isdigit():
            pages[name.strip()] = int(count.strip().rstrip("."))
    wanted = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    return size * sum(pages.get(k, 0) for k in wanted)


def linux_available(text: str) -> int:
    """Bytes ``MemAvailable`` names in /proc/meminfo text, or 0."""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def nvidia_gpus(text: str) -> tuple[Gpu, ...]:
    """Cards from ``nvidia-smi --query-gpu=name,memory.total,memory.free`` CSV in MiB."""
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            out.append(Gpu(parts[0], int(parts[1]) * 1024**2, int(parts[2]) * 1024**2,
                           "nvidia"))
    return tuple(out)


def amd_gpus(text: str) -> tuple[Gpu, ...]:
    """Cards from ``rocm-smi --showmeminfo vram --json`` output."""
    try:
        data = json.loads(text)
    except ValueError:
        return ()
    out = []
    for name, card in sorted(data.items() if isinstance(data, dict) else ()):
        if not isinstance(card, dict):
            continue
        total = card.get("VRAM Total Memory (B)")
        used = card.get("VRAM Total Used Memory (B)")
        if str(total).isdigit() and str(used).isdigit():
            out.append(Gpu(name, int(total), max(0, int(total) - int(used)), "amd"))
    return tuple(out)


def _available(system: str, total: int) -> int:
    if system == "Darwin":
        got = macos_available(_run(["vm_stat"]))
    elif system == "Linux":
        try:
            got = linux_available(Path("/proc/meminfo").read_text(encoding="utf-8"))
        except OSError:
            got = 0
    else:
        got = hub.free_memory() or 0
    if not got:
        got = hub.free_memory() or 0
    return min(got, total) if total else got


def machine_memory() -> MachineMemory:
    """Probe this machine. Every part is optional: what cannot be read is left at zero."""
    system = platform.system()
    total = hub.total_memory()
    notes: list[str] = []
    available = _available(system, total)
    unified = system == "Darwin" and platform.machine() == "arm64"
    gpus: tuple[Gpu, ...] = ()
    limit = 0
    if unified:
        limit = hub.machine_room()
    else:
        gpus = nvidia_gpus(_run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                 "--format=csv,noheader,nounits"]))
        gpus = gpus or amd_gpus(_run(["rocm-smi", "--showmeminfo", "vram", "--json"]))
        if not gpus:
            notes.append("no GPU memory could be read; sizing for the CPU")
    if not total:
        notes.append("installed memory could not be read")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "" and gpus:
        gpus = ()
        notes.append("CUDA_VISIBLE_DEVICES is empty; GPUs ignored")
    return MachineMemory(total, available, unified, limit, gpus, system or sys.platform,
                         tuple(notes))


__all__ = ["GIB", "Gpu", "MachineMemory", "amd_gpus", "linux_available", "machine_memory",
           "macos_available", "nvidia_gpus"]
