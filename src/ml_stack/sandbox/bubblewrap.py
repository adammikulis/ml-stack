"""The Linux backend's bubblewrap command and GPU mounts."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Sequence

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
            if os.path.exists(path):
                out += ["--dev-bind", path, path]
        for path in ("/usr/lib/wsl/lib", "/usr/lib/wsl/drivers"):
            if os.path.isdir(path):
                out += ["--ro-bind", path, path]
    for key, value in policy.env.items():
        out += ["--setenv", key, value]
    return out


class Bubblewrap:
    """Runs a command inside bubblewrap namespaces."""

    name = "bubblewrap"

    def available(self) -> Availability:
        if not sys.platform.startswith("linux"):
            return Availability(False, "bubblewrap runs on Linux only")
        if not shutil.which("bwrap"):
            return Availability(False, "bwrap is not installed (apt install bubblewrap)")
        return Availability(True)

    def wrap(self, argv: Sequence[str], policy: Policy) -> Wrapped:
        found = shutil.which("bwrap")
        if not found:
            raise PolicyError("bwrap is not installed")
        program = checked_path(program_of(argv, policy.env.get("PATH", "")), what="command", link=True)
        return Wrapped([found, *arguments(policy, program), "--", *argv])

    def denials(self, tag: str, since: float) -> list[dict[str, str]]:
        return []
