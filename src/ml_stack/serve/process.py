"""Existence, termination and memory for the processes this machine is serving with."""

from __future__ import annotations

import logging
from pathlib import Path

from ml_stack.units import human_bytes

logger = logging.getLogger(__name__)


def pid_exists(pid: int | None) -> bool:
    """Whether ``pid`` names a process that is still doing something."""
    if not pid or pid <= 0:
        return False
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


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
    """Every llama-server process on this machine, leased or not: pid, port, model, memory.

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
            from ml_stack.bench.measure import footprint_of

            rss = footprint_of(proc) or rss
        out.append({"pid": int(proc.info["pid"]), "port": int(after("--port") or 8080),
                    "defunct": state == psutil.STATUS_ZOMBIE,
                    "model": after("--model", "-m") or after("-hf") or "",
                    "binary": argv[0] if argv else name,
                    "rss": rss})
    return sorted(out, key=lambda r: r["port"])


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
