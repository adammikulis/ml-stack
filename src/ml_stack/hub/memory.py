"""How much memory a model may use on this machine: what is installed, what is free, what
the graphics side will wire, and what ml-stack has been told it may take."""

from __future__ import annotations

import ctypes
import os
import sys

from ml_stack import hub


class _MemoryStatus(ctypes.Structure):
    """The shape GlobalMemoryStatusEx fills in. Windows has no sysconf."""

    _fields_ = (("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong))


def _windows_memory() -> tuple[int, int] | None:
    """``(installed, free)`` in bytes on Windows, or None anywhere else."""
    if sys.platform != "win32":
        return None
    try:
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (AttributeError, OSError, ValueError):
        return None


def total_memory() -> int:
    """Memory installed on this machine, in bytes, or 0 when it will not say."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        pass
    both = _windows_memory()
    if both is not None:
        return both[0]
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - a machine that will not say has no total
        return 0


def free_memory() -> int | None:
    """Bytes this machine could still give a model, or None when it will not say."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - no psutil here; ask the platform instead
        pass
    both = _windows_memory()
    return both[1] if both is not None else None


def machine_room() -> int:
    """How much memory this machine would let a model use, in bytes, or 0 when unknown.

    On a machine with unified memory this is not the whole of RAM: Metal will not wire more
    than `iogpu.wired_limit_mb`, and a model plus its KV cache has to fit under that. A
    limit of 0 is macOS saying nobody has set one, so Metal's own default share of
    `hw.memsize` stands. On a machine with a separate card it is that card's memory.
    """
    import subprocess

    try:
        got = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                             capture_output=True, text=True, timeout=5)
        said = got.stdout.strip()
        wired = int(said) if got.returncode == 0 and said.isdigit() else 0
        if wired > 0:
            return wired * 1024 * 1024
        got = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        if got.returncode == 0 and got.stdout.strip().isdigit():
            return int(int(got.stdout.strip()) * 0.75)   # Metal's own default share
    except Exception:  # noqa: BLE001 - not every machine answers, and that is not a failure
        pass
    return 0


def room() -> int:
    """How much memory a model may use here, in bytes, or 0 when unknown.

    `machine_room` is what the machine allows; this is that, capped by what this machine
    was told ml-stack may take (`ml_stack.limits`). Every preflight, fit and lease
    reads it, so a limit set once is honoured everywhere without anybody passing it on.
    """
    from ml_stack.limits import read

    return read().room(hub.machine_room())
