"""Read-only comparison of what is installed with what was prepared: current, stale, missing, drifted or not-prepared."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import home

from . import autostart_ledger
from .autostart_backends import Backend, backend_for
from .autostart_manifest import Manifest, Role, launcher_problem, parse, parse_role
from .autostart_prepare import platform_of, read_manifest, staging_root

__all__ = ["STATES", "Report", "check", "prepared", "state_word", "summary"]

STATES = ("current", "stale", "missing", "drifted", "not-prepared")
CACHE_S = 60.0
_LAST: list[Any] = [0.0, ""]


@dataclass(frozen=True, slots=True)
class Report:
    """The overall state, why, each installed role's state, and the manifest id it answers to."""

    state: str
    reasons: tuple[str, ...] = ()
    roles: tuple[tuple[str, str], ...] = ()
    manifest: str = ""


def prepared(now: float = 0.0, device: str = "", platform: str = "") -> Manifest | None:
    """The newest unexpired manifest staged for this device and platform, or None."""
    now = now or time.time()
    try:
        device = device or home.device_id()
    except RuntimeError:
        return None
    best: Manifest | None = None
    root = staging_root()
    for folder in root.iterdir() if root.is_dir() else ():
        try:
            found = parse(read_manifest(folder / "manifest.json"))
        except (OSError, ValueError):
            continue
        if (found.expires > now and found.device == device and found.platform == platform_of(platform)
                and (best is None or found.created > best.created)):
            best = found
    return best


def _role_reasons(role: Role, platform: str, backend: Backend) -> list[str]:
    out = []
    if platform != "win32":
        for unit in role.units:
            path = Path(unit.destination)
            if not path.is_file():
                out.append(f"{unit.destination} is missing")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != unit.sha256:
                out.append(f"{unit.destination} was edited after it was installed")
    problem = launcher_problem(Path(role.argv[0]))
    if problem:
        out.append(problem)
    if not backend.loaded(role):
        out.append(f"{role.label} is not loaded")
    return out


def _stale(installed: dict[str, Any], waiting: Manifest | None) -> bool:
    if waiting is None:
        return False
    for role in waiting.roles:
        have = installed.get(role.role)
        if have is None or have["manifest"] == waiting.id:
            continue
        if {u.sha256 for u in role.units} != {u["sha256"] for u in have["role"]["units"]}:
            return True
    return False


def check(*, runner: Any = None, now: float = 0.0, device: str = "", platform: str = "") -> Report:
    """The autostart state of this device; reads files and asks the service manager, changes nothing."""
    installed = autostart_ledger.read()["installed"]
    waiting = prepared(now, device, platform)
    if not installed:
        return Report("missing" if waiting else "not-prepared")
    reasons: list[str] = []
    roles = []
    for name, entry in sorted(installed.items()):
        backend = backend_for(entry["platform"], entry["scope"], runner)
        found = _role_reasons(parse_role(entry["role"]), entry["platform"], backend)
        roles.append((name, "drifted" if found else "current"))
        reasons += [f"{name}: {why}" for why in found]
    ids = sorted({e["manifest"] for e in installed.values()})
    if reasons:
        return Report("drifted", tuple(reasons), tuple(roles), ids[-1])
    if _stale(installed, waiting):
        return Report("stale", ("a newer prepared manifest differs from what is installed",), tuple(roles), ids[-1])
    return Report("current", (), tuple(roles), ids[-1])


def summary(*, runner: Any = None) -> dict[str, Any]:
    """The report as a small JSON-able object for a device profile or a status listing."""
    report = check(runner=runner)
    return {"state": report.state, "reasons": list(report.reasons[:3]), "manifest": report.manifest}


def state_word(now: float = 0.0) -> str:
    """The autostart state as one word, asked of the service manager at most once a minute."""
    now = now or time.monotonic()
    if not _LAST[1] or now - _LAST[0] > CACHE_S:
        try:
            _LAST[:] = [now, check().state]
        except (OSError, ValueError, RuntimeError, KeyError):
            _LAST[:] = [now, "unknown"]
    return _LAST[1]
