"""Linux bubblewrap namespaces, native availability and policy arguments."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from ml_stack.platform import start_process, terminate_process_group
from ml_stack.sandbox.backend import Availability, Wrapped, program_of
from ml_stack.sandbox.policy import NetMode, Policy, PolicyError, checked_path

__all__ = ["Bubblewrap", "arguments"]

SYSTEM_READ = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/ssl", "/etc/ld.so.cache",
               "/etc/resolv.conf", "/etc/hosts", "/etc/localtime", "/etc/passwd", "/etc/group")


def arguments(policy: Policy, program: str) -> list[str]:
    """The bwrap options for ``policy``: a new namespace for everything, network included
    unless the policy keeps loopback, the allow-listed trees bound read-only or read-write,
    and a fresh /proc, /dev and /tmp."""
    out = ["--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    if policy.net.mode != NetMode.DENY:
        out.append("--share-net")
    out += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]  # noqa: S108 - a mount point
    seen: set[str] = set()
    for path in [*(p for p in SYSTEM_READ if os.path.lexists(p)), *policy.read, *policy.exec,
                 program]:
        if path not in seen:
            seen.add(path)
            out += ["--ro-bind", path, path]
    for path in policy.write:
        out += ["--bind", path, path]
    if policy.cache:
        out += ["--bind", policy.cache, policy.cache]
    if policy.gpu:
        for path in ("/dev/dxg", "/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm",
                     "/dev/nvidia-uvm-tools", "/dev/dri"):
            if Path(path).exists():
                out += ["--dev-bind", path, path]
        for path in ("/usr/lib/wsl/lib", "/usr/lib/wsl/drivers"):
            if Path(path).is_dir():
                out += ["--ro-bind", path, path]
    for key, value in policy.env.items():
        out += ["--setenv", key, value]
    return out


_PROBE_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _probe(binary: str, identity: tuple, program: str) -> Availability:
    held = Policy("bubblewrap-probe", exec=(program,), env={}).validated()
    process = start_process([binary, *arguments(held, program), "--", program],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env={})
    try:
        _, error = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        terminate_process_group(process, force=True)
        process.communicate()
        return Availability(False, "bubblewrap namespace probe timed out")
    if process.returncode:
        return Availability(False, "bubblewrap namespace probe failed: " + error.decode(errors="replace")[:300].strip())
    return Availability(True)


class Bubblewrap:
    """Runs a command inside bubblewrap namespaces."""

    name = "bubblewrap"

    def available(self) -> Availability:
        if not sys.platform.startswith("linux"):
            return Availability(False, "bubblewrap runs on Linux only")
        found = shutil.which("bwrap")
        if not found:
            return Availability(False, "bwrap is not installed (apt install bubblewrap)")
        try:
            binary = checked_path(os.path.realpath(found), what="bubblewrap")
            program = checked_path(os.path.realpath("/usr/bin/true"), what="probe")
            stat = Path(binary).stat()
            namespace = Path("/proc/self/ns/user")
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                        stat.st_mode, os.geteuid(),
                        namespace.stat().st_ino if namespace.exists() else 0)
            with _PROBE_LOCK:
                return _probe(binary, identity, program)
        except (OSError, PolicyError) as exc:
            return Availability(False, str(exc))

    def wrap(self, argv: Sequence[str], policy: Policy) -> Wrapped:
        found = shutil.which("bwrap")
        if not found:
            raise PolicyError("bwrap is not installed")
        program = checked_path(program_of(argv, policy.env.get("PATH", "")), what="command", link=True)
        return Wrapped([found, *arguments(policy, program), "--", *argv])

    def denials(self, tag: str, since: float) -> list[dict[str, str]]:
        return []
