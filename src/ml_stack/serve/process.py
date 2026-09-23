"""Existence, termination and memory for the processes this machine is serving with."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from ml_stack.client.health import reported_models
from ml_stack.home import state
from ml_stack.units import human_bytes

logger = logging.getLogger(__name__)


def measuring_file(home: Path | None = None) -> Path:
    """Where the run holding this machine's measuring lock writes its pid, argv, log,
    start time and how it is asking."""
    return (home or state("bench")) / "measuring.json"


def measuring_lock_file(home: Path | None = None) -> Path:
    """Where the run holding this machine's measuring lock is named, whatever else it
    wrote."""
    return (home or state("bench")) / "measuring.lock"


def _locked_by(home: Path | None = None) -> int | None:
    """The pid written into the measuring lock, or None."""
    try:
        said = measuring_lock_file(home).read_text(encoding="utf-8").split()
    except OSError:
        return None
    return int(said[-1]) if said and said[-1].isdigit() else None


def measuring(home: Path | None = None) -> dict[str, Any] | None:
    """The measurement still running on this machine, or None. Read from
    `measuring_file`; a record marked ended, or one whose pid has gone, is a measurement
    that finished.

    A live lock with no record of its own is still a measurement, reported with the little
    the lock knows: a machine whose GPU is busy must never read as idle.
    """
    try:
        record = json.loads(measuring_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and not record.get("ended") and pid_exists(record.get("pid")):
        return record
    pid = _locked_by(home)
    if pid is None or not pid_exists(pid) \
            or (isinstance(record, dict) and record.get("pid") == pid):
        return None
    return {"pid": pid, "argv": [], "log": "", "how": {},
            "started": "", "lock_only": True}

# `proc_pid_rusage(pid, RUSAGE_INFO_V4, &buf)`, and where `ri_phys_footprint` sits in
# `rusage_info_v4`: sixteen bytes of uuid, then user and system time, two wakeup counts,
# pageins, wired size, resident size, and the footprint -- the eighth `uint64_t`. Verified
# against `ps -o rss` on this machine rather than counted off the header, because counting
# it off the header put it one slot late and read the process's start time as a footprint
# of eight terabytes.
RUSAGE_INFO_V4 = 4
PHYS_FOOTPRINT_AT = 16 + 7 * 8


def _rusage_footprint(pid: int) -> int:
    """macOS's phys_footprint for ``pid`` -- Activity Monitor's "Memory" -- or 0.

    0 for a process that is gone, a platform without the call, or any failure at all -- a
    memory reading is never worth a caller failing over.
    """
    if sys.platform != "darwin":
        return 0
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.dylib")
        buf = ctypes.create_string_buffer(1024)
        if lib.proc_pid_rusage(ctypes.c_int(int(pid)), ctypes.c_int(RUSAGE_INFO_V4),
                               ctypes.byref(buf)) != 0:
            return 0
        return int.from_bytes(buf.raw[PHYS_FOOTPRINT_AT:PHYS_FOOTPRINT_AT + 8], sys.byteorder)
    except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
        return 0


def footprint_of(process: Any) -> int:
    """One process's phys_footprint in bytes -- Activity Monitor's "Memory".

    psutil's own field where a build has one (some expose it in ``memory_full_info``), the
    `proc_pid_rusage` read where it does not, and the resident set on every platform that
    has no such distinction -- Linux and Windows charge a process for what is resident, so
    there the two figures are the same number and the table says so by printing it twice.
    """
    try:
        info = process.memory_info()
        for name in ("phys_footprint", "footprint"):
            got = int(getattr(info, name, 0) or 0)
            if got:
                return got
    except Exception:  # noqa: BLE001
        pass
    through_kernel = _rusage_footprint(int(getattr(process, "pid", 0) or 0))
    if through_kernel:
        return through_kernel
    try:
        return int(process.memory_info().rss)
    except Exception:  # noqa: BLE001
        return 0


def pid_exists(pid: int | None) -> bool:
    """Whether ``pid`` names a process that is still doing something."""
    if not pid or pid <= 0:
        return False
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def self_or_ancestor(pid: int | None) -> bool:
    """Whether ``pid`` is this process or one of the processes that started it."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil
    except ImportError:
        return pid == os.getppid()
    try:
        return any(parent.pid == pid for parent in psutil.Process(os.getpid()).parents())
    except psutil.Error:
        return pid == os.getppid()


def kill_pid(pid: int, *, grace_s: float = 1.0) -> None:
    """Terminate ``pid`` gracefully, escalating to kill after ``grace_s``."""
    if not pid_exists(pid):
        return
    import psutil

    try:
        proc = psutil.Process(pid)
    except Exception as exc:
        logger.debug("kill_pid(%s) lookup failed: %s", pid, exc)
        return
    try:
        proc.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except psutil.TimeoutExpired:
        pass
    except psutil.NoSuchProcess:
        return
    try:
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def kill_process_tree(pid: int, *, grace_s: float = 5.0) -> list[int]:
    """Terminate ``pid`` and every descendant, returning the pids acted on."""
    if not pid_exists(pid):
        return []
    import psutil

    try:
        parent = psutil.Process(pid)
        victims = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return []

    for proc in victims:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _gone, alive = psutil.wait_procs(victims, timeout=grace_s)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return [proc.pid for proc in victims]


def every_server() -> list[dict]:
    """Every llama-server process on this machine, leased or not: pid, port, model, the draft
    head it was started with, memory.

    A server nobody recorded -- a Homebrew one from before the managed build, a hand start
    -- holds memory a lease cannot see.
    """
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            argv = list(proc.info.get("cmdline") or [])
            name = str(proc.info.get("name") or "")
        except (psutil.Error, OSError):
            continue
        head = Path(argv[0]).name if argv else name
        if "llama-server" not in head and "llama-server" not in name:
            continue

        def after(flag: str, short: str = "") -> str:
            for i, a in enumerate(argv[:-1]):
                if a == flag or (short and a == short):
                    return argv[i + 1]
                if a.startswith(flag + "="):
                    return a.split("=", 1)[1]
            return ""

        mem = proc.info.get("memory_info")
        try:
            state = str(proc.status())
        except (psutil.Error, OSError):
            state = ""
        rss = int(getattr(mem, "rss", 0) or 0)
        if isinstance(proc, psutil.Process):
            rss = footprint_of(proc) or rss
        ahead = after("--spec-draft-n-max")
        out.append({"pid": int(proc.info["pid"]), "port": int(after("--port") or 8080),
                    "defunct": state == psutil.STATUS_ZOMBIE,
                    "model": after("--model", "-m") or after("-hf") or "",
                    "binary": argv[0] if argv else name,
                    "draft": after("--spec-draft-model", "-md") or after("--model-draft")
                             or after("-hfd"),
                    "spec_type": after("--spec-type"),
                    "draft_max": int(ahead) if ahead.isdigit() else None,
                    "rss": rss})
    return sorted(out, key=lambda r: r["port"])


def loaded_twice(servers: list[dict] | None = None) -> dict[str, list[int]]:
    """Each model a live llama-server reports serving on more than one port, with the ports.

    ``servers`` defaults to `every_server`; a server that does not answer ``/v1/models``
    reports nothing and counts for no model.
    """
    ports_of: dict[str, list[int]] = {}
    asked: set[int] = set()
    for server in every_server() if servers is None else servers:
        if server.get("defunct"):
            continue
        port = int(server["port"])
        # One port answers for one server, however many processes carry its command
        # line: a second copy of a model is a second port, never a second process.
        if port in asked:
            continue
        asked.add(port)
        for name in reported_models(f"http://127.0.0.1:{port}"):
            ports_of.setdefault(name, []).append(port)
    return {name: ports for name, ports in ports_of.items() if len(ports) > 1}


def machine_memory() -> dict | None:
    """What the machine holds: total, used, wired, free, the llama-servers' resident total,
    everything else's, and the five largest non-server processes -- None without psutil."""

    try:
        import psutil
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001
        return None
    servers = 0
    rest: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            mem = proc.info.get("memory_info")
            rss = int(getattr(mem, "rss", 0) or 0)
            argv = list(proc.info.get("cmdline") or [])
            head = Path(argv[0]).name if argv else str(proc.info.get("name") or "")
        except (psutil.Error, OSError):
            continue
        if "llama-server" in head:
            servers += rss
        elif rss:
            rest.append((rss, head))
    rest.sort(reverse=True)
    return {"total": int(vm.total), "used": int(vm.total - vm.available),
            "wired": int(getattr(vm, "wired", 0) or 0), "free": int(vm.available),
            "servers": servers, "others": sum(r for r, _ in rest),
            "largest": [f"{name} {human_bytes(r)}" for r, name in rest[:5]]}
