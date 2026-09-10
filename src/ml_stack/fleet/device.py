"""What this box is and what it is doing, as the beacon and ``/health`` carry it."""

from __future__ import annotations

import functools
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.hub import free_memory, total_memory

from .jobs import DaemonError

REPORT_GROUP = "ml_stack.device_report"
"""Entry-point group a higher tier registers a richer device probe under."""


def _ram_used_gb(total_gb: float) -> float | None:
    """Memory in use, from whatever this platform will say without a dependency."""
    try:
        import psutil
        return round(psutil.virtual_memory().used / 2**30, 2)
    except Exception:                                 # noqa: BLE001
        pass
    free = free_memory()
    if free is not None:
        return round(max(0.0, total_gb - free / 2**30), 2)
    try:
        if sys.platform.startswith("linux"):
            fields = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                fields[key] = float(rest.strip().split()[0]) / 2**20
            free = fields.get("MemAvailable", fields.get("MemFree", 0.0))
            return round(max(0.0, total_gb - free), 2)
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=5).stdout
            page = 4096
            first = out.splitlines()[0] if out else ""
            if "page size of" in first:
                page = int(first.split("page size of")[1].split()[0])
            pages = {}
            for line in out.splitlines()[1:]:
                key, _, rest = line.partition(":")
                pages[key.strip()] = int(rest.strip().rstrip(".") or 0)
            used = sum(pages.get(k, 0) for k in
                       ("Pages active", "Pages wired down", "Pages occupied by compressor"))
            return round(used * page / 2**30, 2)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return None


_cpu_primed = False


def _cpu_busy_pct() -> float | None:
    """How busy the processors are, as a percentage."""
    global _cpu_primed
    try:
        import psutil
        # The first reading has nothing to measure against and comes back 0.0, so
        # the load average answers until there is a second one.
        if not _cpu_primed:
            psutil.cpu_percent(interval=None)
            _cpu_primed = True
        else:
            return round(float(psutil.cpu_percent(interval=None)), 1)
    except Exception:                                 # noqa: BLE001
        pass
    try:
        load = os.getloadavg()[0]
        return round(min(100.0, 100.0 * load / (os.cpu_count() or 1)), 1)
    except (AttributeError, OSError):
        return None


def stdlib_device_report() -> dict[str, Any]:
    """What this box is, and what it is doing, from the standard library."""
    out: dict[str, Any] = {"backends": [], "cpus": os.cpu_count() or 1,
                           "arch": platform.machine(), "platform": sys.platform}
    ram = total_memory()
    if ram:
        out["ram_gb"] = round(ram / 2**30, 2)
        used = _ram_used_gb(out["ram_gb"])
        if used is not None:
            out["ram_used_gb"] = min(used, out["ram_gb"])
    busy = _cpu_busy_pct()
    if busy is not None:
        out["cpu_pct"] = busy
    return out


@functools.cache
def registered_reports() -> list[Callable[[], dict[str, Any]]]:
    """Every device probe a higher tier has registered under ``REPORT_GROUP``."""
    try:
        from importlib.metadata import entry_points
        found = entry_points(group=REPORT_GROUP)
    except Exception:                                 # noqa: BLE001
        return []
    out = []
    for ep in found:
        try:
            out.append(ep.load())
        except Exception:                             # noqa: BLE001
            continue
    return out


def resolve_report(spec: str) -> Callable[[], dict[str, Any]]:
    """Turn ``"ml_stack.train.accelerator:report"`` into the callable it names."""
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise DaemonError(
            f"bad report spec {spec!r} -- expected 'module.path:callable'")
    try:
        import importlib
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise DaemonError(f"cannot load report {spec!r}: {exc}") from None


def device_report(extra: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """What is on this box. Best effort, and additive."""
    out = stdlib_device_report()
    for fn in ([extra] if extra is not None else registered_reports()):
        try:
            out.update(fn() or {})
        except Exception:                             # noqa: BLE001
            pass
    return out

