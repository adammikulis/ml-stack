"""The macOS backend: a deny-by-default Seatbelt profile run with ``sandbox-exec``.

``sandbox-exec`` is deprecated by Apple and is the only use of a deprecated tool in this
repository (docs/sandbox.md records the exception). Nothing outside this module names it.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from ml_stack.sandbox.backend import Availability, Wrapped, program_of
from ml_stack.sandbox.policy import NetMode, Policy, PolicyError, checked_path

__all__ = ["BINARY", "DEPRECATION", "ProfileError", "Seatbelt", "profile", "quote"]

logger = logging.getLogger("ml_stack.sandbox")

BINARY = "/usr/bin/sandbox-exec"
DEPRECATION = ("sandbox-exec is deprecated by Apple and has no command-line replacement on "
               "macOS; ml-stack uses it for confinement until a container backend can "
               "replace it (docs/sandbox.md)")

SYSTEM_READ = ("/usr/lib", "/usr/share", "/System/Library", "/System/Cryptexes",
               "/private/var/db/dyld", "/Library/Apple/System/Library", "/private/etc/ssl",
               "/private/etc/resolv.conf", "/private/etc/hosts", "/private/etc/localtime",
               "/private/var/db/timezone", "/usr/bin", "/bin", "/sbin", "/usr/sbin",
               "/usr/libexec", "/System/Volumes/Preboot/Cryptexes", "/private/var/select")
"""System trees every process reads: the dynamic linker, frameworks, locale and zoneinfo data."""

SYMLINKS = ("/etc", "/var", "/tmp")  # noqa: S108 - the system symlinks, not a temp file

DEVICES_READ = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/dtracehelper",
                "/dev/tty", "/dev/autofs_nowait")
DEVICES_WRITE = ("/dev/null", "/dev/dtracehelper", "/dev/tty")

SYSCTLS = ("hw.ncpu", "hw.activecpu", "hw.logicalcpu", "hw.logicalcpu_max", "hw.physicalcpu",
           "hw.physicalcpu_max", "hw.memsize", "hw.pagesize", "hw.pagesize_compat", "hw.machine", "hw.cputype",
           "hw.cpufamily", "hw.byteorder", "hw.cachelinesize", "hw.l1dcachesize",
           "hw.l2cachesize", "hw.nperflevels", "hw.packages", "hw.tbfrequency",
           "hw.vectorunit", "kern.argmax", "kern.hostname", "kern.maxfilesperproc",
           "kern.ostype", "kern.osrelease", "kern.osversion", "kern.osproductversion",
           "kern.version", "kern.usrstack64", "kern.secure_kernel", "kern.osvariant_status",
           "kern.bootargs", "kern.willshutdown", "kern.tcsm_available", "kern.tcsm_enable",
           "kern.ngroups", "kern.iossupportversion", "kern.maxfiles", "kern.maxproc", "vm.loadavg", "machdep.ptrauth_enabled",
           "security.mac.lockdown_mode_state", "sysctl.proc_cputype")
SYSCTL_PREFIXES = ("hw.optional.", "hw.perflevel", "machdep.cpu.", "kern.proc.pid.")

MACH_SERVICES = ("com.apple.system.opendirectoryd.libinfo", "com.apple.logd",
                 "com.apple.system.logger", "com.apple.system.notification_center",
                 "com.apple.distributed_notifications@Uv3", "com.apple.bsd.dirhelper")

GPU_IOKIT_CLASSES = ("IOGPUDeviceUserClient",)
GPU_MACH_SERVICES = ("com.apple.MTLCompilerService",)


class ProfileError(PolicyError):
    """A string that cannot be put in a profile."""


_BAD = re.compile(r"[\x00-\x1f\x7f]")


def quote(text: str) -> str:
    """``text`` as a Seatbelt string literal: quote and backslash escaped, control characters
    refused."""
    if _BAD.search(text):
        raise ProfileError(f"{text!r} holds a control character")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ancestors(paths: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {"/": None}
    for path in paths:
        parts = path.strip("/").split("/")
        for i in range(1, len(parts)):
            seen["/" + "/".join(parts[:i])] = None
    return list(seen)


def _filter(kind: str, paths: Sequence[str]) -> str:
    return " ".join(f"({kind} {quote(p)})" for p in paths)


def _existing(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if os.path.lexists(p) and os.path.realpath(p) == p]


def profile(policy: Policy, program: str, tag: str) -> str:
    """The profile text for ``policy`` running ``program``. Every path in it was checked and
    is quoted."""
    if not re.fullmatch(r"[A-Za-z0-9:_-]{1,64}", tag):
        raise ProfileError(f"tag {tag!r} is not a plain token")
    real = os.path.realpath(program)
    execs = list(dict.fromkeys([program, real, *policy.exec]))
    writes = list(policy.write)
    reads = list(dict.fromkeys([*_existing(SYSTEM_READ), *policy.read, *writes, *execs]))
    if policy.cache:
        reads.append(policy.cache)
        writes.append(policy.cache)
    out = [
        "(version 1)",
        f'(deny default (with message {quote("ml-stack-sandbox:" + tag)}))',
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow process-info* (target self))",
        "(allow process-info-pidinfo (target self))",
        "(allow user-preference-read)",
        f"(allow process-exec {_filter('subpath', execs)})",
        f"(allow file-read* {_filter('subpath', reads)})",
        f"(allow file-read-metadata {_filter('literal', _ancestors(reads + writes))})",
        '(allow file-read-data (literal "/"))',
        f"(allow file-read* {_filter('literal', DEVICES_READ)})",
        f"(allow file-write* {_filter('literal', DEVICES_WRITE)})",
        f"(allow file-ioctl {_filter('literal', DEVICES_WRITE)})",
        f"(allow file-read-metadata {_filter('literal', SYMLINKS)})",
        '(allow ipc-posix-shm-read-data (ipc-posix-name "apple.shm.notification_center"))',
        "(allow sysctl-read " + " ".join(
            [f"(sysctl-name {quote(s)})" for s in SYSCTLS]
            + [f"(sysctl-name-prefix {quote(s)})" for s in SYSCTL_PREFIXES]) + ")",
        "(allow mach-lookup " + " ".join(f"(global-name {quote(s)})" for s in MACH_SERVICES) + ")",
    ]
    if writes:
        out.append(f"(allow file-write* {_filter('subpath', writes)})")
    out += _network(policy)
    if policy.gpu:
        out += _gpu()
    return "\n".join(out) + "\n"


def _network(policy: Policy) -> list[str]:
    net = policy.net
    if net.mode == NetMode.DENY:
        return []
    if net.mode == NetMode.LOOPBACK:
        return ['(allow network-outbound (remote ip "localhost:*"))',
                '(allow network-bind (local ip "localhost:*"))',
                '(allow network-inbound (local ip "localhost:*"))']
    return [f'(allow network-outbound (remote ip "localhost:{p}"))' for p in net.ports]


def _gpu() -> list[str]:
    out = []
    if GPU_IOKIT_CLASSES:
        out.append("(allow iokit-open " + " ".join(
            f"(iokit-user-client-class {quote(c)})" for c in GPU_IOKIT_CLASSES) + ")")
    if GPU_MACH_SERVICES:
        out.append("(allow mach-lookup " + " ".join(
            f"(global-name {quote(s)})" for s in GPU_MACH_SERVICES) + ")")
    return out


NOISE = re.compile(
    r"\.CFUserTextEncoding$|/\.?GlobalPreferences[^/]*\.plist$|^com\.apple\.SystemConfiguration\.configd$"
    r"|^/Users/[^/]+$|^com\.apple\.analyticsd$|^com\.apple\.system\.opendirectoryd\.membership$"
    r"|^net\.routetable\.|^/private/etc/localtime$|^/Library/Preferences/com\.apple\.networkd\.plist$")
"""Targets every confined process probes at start-up and survives being refused."""

_WARNED = threading.Event()
DENIAL = re.compile(r"Sandbox: (?P<proc>[^(]+)\((?P<pid>\d+)\) deny\(\d+\) (?P<op>[\w*-]+)\s*(?P<target>.*)$")


class Seatbelt:
    """Runs a command under a generated Seatbelt profile."""

    name = "seatbelt"

    def available(self) -> Availability:
        if sys.platform != "darwin":
            return Availability(False, "Seatbelt exists on macOS only")
        if not (Path(BINARY).is_file() and os.access(BINARY, os.X_OK)):
            path = shutil.which("sandbox-exec")
            return Availability(False, f"{BINARY} is missing (found {path!r} on PATH)"
                                if path else f"{BINARY} is missing")
        return Availability(True)

    def wrap(self, argv: Sequence[str], policy: Policy) -> Wrapped:
        if not _WARNED.is_set():
            _WARNED.set()
            logger.warning("%s", DEPRECATION)
        ready = self.available()
        if not ready.ok:
            raise PolicyError(ready.reason)
        program = checked_path(program_of(argv, policy.env.get("PATH", "")), what="command", link=True)
        tag = secrets.token_hex(6)
        text = profile(policy, program, tag)
        return Wrapped([BINARY, "-p", text, "--", *argv], tag=tag)

    def denials(self, tag: str, since: float) -> list[dict[str, str]]:
        """What the sandbox refused a run tagged ``tag`` since ``since``, from the system log."""
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since - 1))
        try:
            done = subprocess.run(
                ["/usr/bin/log", "show", "--start", started, "--style", "compact",
                 "--predicate", f'sender == "Sandbox" AND eventMessage CONTAINS "{tag}"'],
                capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return []
        found: list[dict[str, str]] = []
        for line in done.stdout.splitlines():
            m = DENIAL.search(line)
            if m:
                target = m["target"].replace(f"ml-stack-sandbox:{tag}", "").strip()
                if NOISE.search(target):
                    continue
                found.append({"process": m["proc"].strip(), "operation": m["op"],
                              "target": target})
        return found


def deprecation_status() -> str:
    """The line ``ml-stack security status`` prints about the deprecated tool."""
    return DEPRECATION
