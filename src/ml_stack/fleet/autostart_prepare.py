"""`prepare`: render the units for chosen roles into a staging directory and write their manifest."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ml_stack import home
from ml_stack.files import write_json, writing

from .autostart_manifest import (
    DAEMON_PORT,
    ROLE_INFO,
    ROLES,
    TTL_S,
    Manifest,
    ManifestError,
    Role,
    Unit,
    allowed_dir,
    destination_problem,
    launcher_problem,
    parse,
    unit_names,
)
from .autostart_units import render

__all__ = ["PrepareError", "Prepared", "Spec", "environment_for", "launcher_path", "platform_of", "prepare",
           "read_manifest", "role_argv", "staging_root"]

LOG_BYTES = 5 * 1024 * 1024
LOG_KEEP = 3
_LABEL = re.compile(r"[A-Za-z0-9._-]{1,64}")


class PrepareError(ValueError):
    """Units that cannot be prepared, with the reason."""


@dataclass(frozen=True, slots=True)
class Spec:
    """What to prepare: the roles, where the stable launchers are, and the daemon's options."""

    roles: tuple[str, ...]
    scope: str = "user"
    launchers: Path | None = None
    slots: int = 1
    labels: tuple[str, ...] = ()
    report: str = ""
    agent: dict[str, str] = field(default_factory=dict)
    platform: str = ""
    device: str = ""
    now: float = 0.0


@dataclass(frozen=True, slots=True)
class Prepared:
    """A staged manifest, where it is, and the one command the person runs."""

    manifest: Manifest
    path: Path
    command: str


def staging_root() -> Path:
    """The directory prepared manifests are staged under."""
    return home.state("autostart", "staging")


def launcher_path(directory: Path, role: str, platform: str) -> Path:
    """Where ``role``'s stable launcher is in ``directory``."""
    return directory / (ROLE_INFO[role]["launcher"] + (".exe" if platform == "win32" else ""))


def environment_for(platform: str) -> dict[str, str]:
    """The small environment a unit may carry: paths only, no credentials."""
    base = home.user_home()
    hub = os.environ.get("HF_HOME") or str(base / ".cache" / "huggingface")
    env = {"HOME": str(base), "HF_HOME": hub, "ML_STACK_CACHE": str(home.cache()),
           "ML_STACK_HOME": str(home.home().absolute())}
    if platform != "win32":
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    return env


def role_argv(role: str, launcher: Path, spec: Spec) -> tuple[str, ...]:
    """The exec argv of ``role``: its launcher and options, each its own element."""
    if role == "runtime-ensure":
        return (str(launcher), "runtime", "ensure")
    if not isinstance(spec.slots, int) or not 1 <= spec.slots <= 64:
        raise PrepareError("slots is a number from 1 to 64")
    out = [str(launcher), f"--port={DAEMON_PORT}"]
    if spec.slots != 1:
        out.append(f"--slots={spec.slots}")
    for label in spec.labels:
        if not _LABEL.fullmatch(label):
            raise PrepareError(f"label {label!r} is letters, digits, dot, dash and underscore")
        out.append(f"--label={label}")
    if spec.report:
        if len(spec.report) > 512 or any(ord(c) < 32 for c in spec.report):
            raise PrepareError("report is one short line")
        out.append(f"--report={spec.report}")
    return tuple(out)


def _role(name: str, platform: str, spec: Spec, launchers: Path) -> Role:
    launcher = launcher_path(launchers, name, platform)
    problem = launcher_problem(launcher)
    if problem:
        raise PrepareError(problem)
    run = home.state("autostart", "run", name)
    logs = home.state("autostart", "logs", f"{name}.log")
    daemon = name == "pool-daemon"
    return Role(
        name, ROLE_INFO[name]["label"], role_argv(name, launcher, spec), environment_for(platform),
        str(run),
        {"policy": "on-failure" if daemon else "schedule", "backoff_s": 30 if daemon else 60,
         "burst": 5 if daemon else 1, "interval_s": 300 if daemon else 3600},
        {"mode": "journal" if platform == "linux" else "files",
         "stdout": "journal" if platform == "linux" else str(logs),
         "stderr": "journal" if platform == "linux" else str(logs), "max_bytes": LOG_BYTES,
         "keep": LOG_KEEP},
        ({"kind": "http", "url": f"http://127.0.0.1:{DAEMON_PORT}/health", "timeout_s": 30} if daemon
         else {"kind": "loaded", "timeout_s": 10}))


def platform_of(named: str = "") -> str:
    """``named``, or the platform this process runs on, as a unit format."""
    chosen = named or ("win32" if sys.platform == "win32" else "darwin" if sys.platform == "darwin"
                       else "linux")
    if chosen not in ("darwin", "linux", "win32"):
        raise PrepareError(f"{chosen} has no unit format")
    return chosen


def _write(directory: Path, name: str, data: bytes) -> None:
    with writing(directory / name) as temporary:
        temporary.write_bytes(data)
        temporary.chmod(0o600)


def _sweep(root: Path, now: float) -> None:
    """Remove staging directories whose manifests expired more than a week ago."""
    for old in root.iterdir() if root.is_dir() else ():
        try:
            expired = parse(read_manifest(old / "manifest.json")).expires
        except (OSError, ValueError):
            continue
        if now - expired > 7 * 86400 and old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)


def read_manifest(path: Path) -> object:
    """The decoded JSON in a staged manifest file (at most 256 KiB)."""
    if path.stat().st_size > 262144:
        raise ManifestError("the manifest is too large")
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(spec: Spec) -> Prepared:
    """Stage the units for ``spec`` and return the manifest with the person's install command."""
    names = tuple(dict.fromkeys(spec.roles))
    if not names or any(n not in ROLES for n in names):
        raise PrepareError(f"roles are chosen from {', '.join(ROLES)}")
    platform = platform_of(spec.platform)
    if spec.scope not in ("user", "system") or (spec.scope == "system" and platform == "win32"):
        raise PrepareError("scope is user, or system on macOS and Linux")
    launchers = spec.launchers
    if launchers is None:
        raise PrepareError("name the directory holding the stable launchers with --launchers")
    now = spec.now or time.time()
    user = getpass.getuser()
    roles, files = [], {}
    for name in names:
        built = _role(name, platform, spec, launchers)
        bodies = render(built, platform, spec.scope, user)
        units = []
        for kind, file in unit_names(name, platform):
            where = f"task:{built.label}" if platform == "win32" else str(allowed_dir(platform, spec.scope) / file)
            problem = destination_problem(where, name, file, platform, spec.scope)
            if problem:
                raise PrepareError(problem)
            files[file] = bodies[kind]
            units.append(Unit(kind, file, where, hashlib.sha256(bodies[kind]).hexdigest()))
        roles.append(replace(built, units=tuple(units)))
    ident = secrets.token_hex(6)
    who = spec.agent or {"agent": os.environ.get("ML_STACK_WORKSPACE_AGENT", ""),
                         "label": os.environ.get("ML_STACK_WORKSPACE_LABEL", "")}
    manifest = Manifest(ident, now, now + TTL_S, spec.device or home.device_id(),
                        {k: who.get(k, "") or "unknown" for k in ("agent", "label")},
                        platform, spec.scope, tuple(roles))
    parse(manifest.to_dict())
    root = staging_root()
    directory = root / ident
    directory.mkdir(mode=0o700, parents=True)
    _sweep(root, now)
    for role in roles:
        Path(role.workdir).mkdir(mode=0o700, parents=True, exist_ok=True)
        if role.logs["mode"] == "files":
            Path(role.logs["stdout"]).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for file, data in files.items():
        _write(directory, file, data)
    path = directory / "manifest.json"
    write_json(path, manifest.to_dict())
    path.chmod(0o600)
    command = shlex.join([sys.executable, "-m", "ml_stack.fleet.autostart", "install", "--manifest", str(path)]
                         + (["--system"] if spec.scope == "system" else []))
    return Prepared(manifest, path, command)
