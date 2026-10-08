"""The record `prepare` stages and `install` checks: roles, units, destinations, exec argv, expiry."""

from __future__ import annotations

import ast
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import home, runtime

__all__ = [
    "DAEMON_PORT", "ENV_ALLOWED", "ROLES", "TTL_S", "VERSION", "Manifest", "ManifestError", "Role",
    "Unit", "allowed_dir", "destination_problem", "launcher_problem", "parse", "parse_role", "unit_names",
]

VERSION = 1
TTL_S = 24 * 3600
SKEW_S = 300
ROLES = ("pool-daemon", "runtime-ensure")
DAEMON_PORT = 8770
ENV_ALLOWED = ("HF_HOME", "HOME", "ML_STACK_CACHE", "ML_STACK_HOME", "PATH")
PLATFORMS = ("darwin", "linux", "win32")
SCOPES = ("user", "system")
LAUNCHER_HEAD = runtime.LAUNCHER.partition("{root!r}")[0]
ROLE_INFO: dict[str, dict[str, Any]] = {
    "pool-daemon": {"label": "com.ml-stack.traind", "service": "ml-stack-traind",
                    "launcher": "ml-stack-traind", "kinds": ("service",)},
    "runtime-ensure": {"label": "com.ml-stack.runtime-ensure", "service": "ml-stack-runtime-ensure",
                       "launcher": "ml-stack", "kinds": ("service", "timer")},
}
_ID = re.compile(r"[0-9a-f]{12}")
_HEX = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A manifest that is malformed, expired, for another device or tampered with."""


@dataclass(frozen=True, slots=True)
class Unit:
    """One file a role installs: its staged name, where it goes, and the hash of its bytes."""

    kind: str
    name: str
    destination: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Role:
    """What one role runs and how its platform keeps it running."""

    role: str
    label: str
    argv: tuple[str, ...]
    environment: dict[str, str]
    workdir: str
    restart: dict[str, Any]
    logs: dict[str, Any]
    health: dict[str, Any]
    units: tuple[Unit, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class Manifest:
    """A staged install: who prepared it, for which device and platform, valid until ``expires``."""

    id: str
    created: float
    expires: float
    device: str
    preparer: dict[str, str]
    platform: str
    scope: str
    roles: tuple[Role, ...]
    version: int = VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "id": self.id, "created": self.created, "expires": self.expires,
            "device": self.device, "preparer": self.preparer, "platform": self.platform,
            "scope": self.scope,
            "roles": [{**{k: getattr(r, k) for k in ("role", "label", "environment", "workdir",
                                                      "restart", "logs", "health")},
                       "argv": list(r.argv),
                       "units": [{k: getattr(u, k) for k in ("kind", "name", "destination", "sha256")}
                                 for u in r.units]} for r in self.roles],
        }


def _text(value: Any, what: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(c) < 32 for c in value):
        raise ManifestError(f"{what} must be short text without control characters")
    return value


def parse_role(raw: Any) -> Role:
    """The role in ``raw`` (a decoded JSON object), or `ManifestError`."""
    if not isinstance(raw, dict) or raw.get("role") not in ROLES:
        raise ManifestError(f"a role is one of {', '.join(ROLES)}")
    argv, env = raw.get("argv"), raw.get("environment")
    if not isinstance(argv, list) or not 1 <= len(argv) <= 64:
        raise ManifestError("argv is a list of one to sixty-four strings")
    if not isinstance(env, dict) or set(env) - set(ENV_ALLOWED):
        raise ManifestError(f"environment allows only {', '.join(ENV_ALLOWED)}")
    units = raw.get("units")
    if not isinstance(units, list) or not units:
        raise ManifestError("a role installs at least one unit")
    for key in ("restart", "logs", "health"):
        if not isinstance(raw.get(key), dict):
            raise ManifestError(f"{key} is an object")
    parsed = []
    for one in units:
        digest = one.get("sha256") if isinstance(one, dict) else None
        if not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise ManifestError("a unit carries the sha256 of its file")
        parsed.append(Unit(_text(one.get("kind"), "unit kind", 16), _text(one.get("name"), "unit name", 128),
                           _text(one.get("destination"), "destination"), digest))
    return Role(raw["role"], _text(raw.get("label"), "label", 128),
                tuple(_text(a, "an argv element") for a in argv),
                {_text(k, "environment name", 64): _text(v, "environment value") for k, v in env.items()},
                _text(raw.get("workdir"), "workdir"), raw["restart"], raw["logs"], raw["health"],
                tuple(parsed))


def parse(raw: Any) -> Manifest:
    """The manifest in ``raw`` (a decoded JSON object), or `ManifestError` naming what is wrong."""
    if not isinstance(raw, dict) or raw.get("version") != VERSION:
        raise ManifestError(f"a manifest of version {VERSION} is expected")
    ident = raw.get("id")
    if not isinstance(ident, str) or not _ID.fullmatch(ident):
        raise ManifestError("the manifest id is twelve hex digits")
    made, until = raw.get("created"), raw.get("expires")
    if not all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in (made, until)):
        raise ManifestError("created and expires are times")
    if until - made > TTL_S:
        raise ManifestError("a manifest lives at most twenty-four hours")
    preparer = raw.get("preparer")
    if not isinstance(preparer, dict) or not all(isinstance(v, str) for v in preparer.values()):
        raise ManifestError("the preparer is recorded")
    if raw.get("platform") not in PLATFORMS or raw.get("scope") not in SCOPES:
        raise ManifestError("the platform and scope are recorded")
    roles = raw.get("roles")
    if not isinstance(roles, list) or not 1 <= len(roles) <= len(ROLES):
        raise ManifestError("a manifest names one or two roles")
    parsed = tuple(parse_role(r) for r in roles)
    if len({r.role for r in parsed}) != len(parsed):
        raise ManifestError("a role appears once")
    return Manifest(ident, float(made), float(until), _text(raw.get("device"), "device", 128),
                    dict(preparer), raw["platform"], raw["scope"], parsed)


def unit_names(role: str, platform: str) -> list[tuple[str, str]]:
    """The (kind, file name) of every unit ``role`` installs on ``platform``."""
    info = ROLE_INFO[role]
    if platform == "darwin":
        return [("plist", f"{info['label']}.plist")]
    if platform == "win32":
        return [("task", f"{info['label']}.xml")]
    return [(kind, f"{info['service']}.{kind}") for kind in info["kinds"]]


def allowed_dir(platform: str, scope: str) -> Path | None:
    """The one directory a unit may be written to for ``platform`` and ``scope``; None for a Windows task."""
    if platform == "win32":
        return None
    if platform == "darwin":
        return home.user_home() / "Library" / "LaunchAgents" if scope == "user" else Path("/Library/LaunchDaemons")
    return home.user_home() / ".config" / "systemd" / "user" if scope == "user" else Path("/etc/systemd/system")


def destination_problem(destination: str, role: str, name: str, platform: str, scope: str) -> str:
    """Why ``destination`` is not where ``role``'s unit ``name`` may be installed, or an empty string."""
    if platform == "win32":
        want = f"task:{ROLE_INFO[role]['label']}"
        return "" if scope == "user" and destination == want else f"a Windows unit is the user task {want}"
    folder = allowed_dir(platform, scope)
    if folder is None:
        return "this platform has no unit directory"
    if ".." in Path(destination).parts or Path(destination) != folder / name:
        return f"{destination} is not {folder / name}"
    if scope == "user":
        base = Path(os.path.realpath(home.user_home()))
        if Path(os.path.realpath(folder)) != base / folder.relative_to(home.user_home()):
            return f"{folder} passes through a symbolic link"
    elif Path(os.path.realpath(folder)) != folder:
        return f"{folder} is a symbolic link"
    target = Path(destination)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        return f"{destination} exists and is not a regular file"
    return ""


def launcher_problem(path: Path) -> str:
    """Why ``path`` is not a launcher the runtime tooling wrote, or an empty string."""
    if not path.is_absolute():
        return "the exec target is not an absolute path"
    try:
        info = path.lstat()
    except OSError:
        return f"the exec target {path} is missing"
    if not stat.S_ISREG(info.st_mode):
        return f"{path} is not a regular file"
    if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o022):
        return f"{path} is not owned by this account or is writable by others"
    if any((up / ".git").exists() or (up / "pyproject.toml").exists() for up in path.parents):
        return f"{path} sits inside a source checkout"
    if path.suffix == ".exe":
        return ""
    try:
        text = path.read_text(encoding="utf-8")[:65536]
    except (OSError, UnicodeDecodeError):
        return f"{path} is unreadable"
    if not text.startswith(LAUNCHER_HEAD):
        return f"{path} is not a runtime launcher"
    try:
        root = next(n.value.value for n in ast.parse(text).body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "root"
                    and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
    except (SyntaxError, StopIteration):
        return f"{path} names no runtime root"
    if not Path(root).is_relative_to(home.state("runtimes")):
        return f"{path} launches a runtime outside {home.state('runtimes')}"
    return ""
