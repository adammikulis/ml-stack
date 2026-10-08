"""Checks a staged manifest against the staged files, this device and the rules for what may be installed."""

from __future__ import annotations

import getpass
import hashlib
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path

from ml_stack import home

from .autostart_manifest import (
    DAEMON_PORT,
    ENV_ALLOWED,
    ROLE_INFO,
    SKEW_S,
    Manifest,
    Role,
    destination_problem,
    launcher_problem,
    unit_names,
)
from .autostart_prepare import environment_for, launcher_path
from .autostart_units import render

__all__ = ["Context", "problems", "staged_bytes"]

_POOL_ARG = re.compile(rf"--port={DAEMON_PORT}|--slots=\d{{1,2}}|--label=[A-Za-z0-9._-]{{1,64}}|--report=[^\x00-\x1f]{{1,512}}")


def _folder_problem(directory: Path) -> str:
    try:
        info = directory.lstat()
    except OSError:
        return f"the staging directory {directory} is missing"
    if not stat.S_ISDIR(info.st_mode):
        return f"{directory} is not a directory"
    if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        return f"{directory} is not private to this account"
    return ""


def staged_bytes(directory: Path, name: str) -> bytes | None:
    """The bytes of the regular, owned, unlinked staged file ``name``; None when it is not one."""
    path = directory / name
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or (os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o022)):
            return None
        return path.read_bytes()
    except OSError:
        return None


def _argv_problem(role: Role, platform: str) -> str:
    exe, rest = role.argv[0], role.argv[1:]
    want = launcher_path(Path(exe).parent, role.role, platform)
    if Path(exe) != want:
        return f"{exe} is not the {ROLE_INFO[role.role]['launcher']} launcher"
    problem = launcher_problem(want)
    if problem:
        return problem
    ok = rest == ("ensure", "--unattended") if role.role == "runtime-ensure" else all(_POOL_ARG.fullmatch(a) for a in rest)
    return "" if ok else f"{role.role} takes only its own options, not {list(rest)}"


def _role_problems(role: Role, platform: str, scope: str) -> list[str]:
    out = []
    if set(role.environment) - set(ENV_ALLOWED) or role.environment != environment_for(platform):
        out.append(f"{role.role}: the environment is not the allowed one")
    for where in (role.workdir, *([role.logs["stdout"]] if role.logs.get("mode") == "files" else [])):
        if ".." in Path(where).parts or not Path(where).is_relative_to(home.state()):
            out.append(f"{role.role}: {where} is outside the state root")
    problem = _argv_problem(role, platform)
    if problem:
        out.append(f"{role.role}: {problem}")
    return out


@dataclass(frozen=True, slots=True)
class Context:
    """When and where an install is attempted: the time, this device, this platform, and whether --system was given."""

    now: float
    device: str
    platform: str
    system: bool


def problems(manifest: Manifest, directory: Path, ctx: Context) -> list[str]:
    """Every reason ``manifest`` staged in ``directory`` may not be installed now, empty when none."""
    now, device, platform, system = ctx.now, ctx.device, ctx.platform, ctx.system
    out = []
    if now > manifest.expires:
        out.append("the manifest has expired; prepare it again")
    if manifest.created > now + SKEW_S:
        out.append("the manifest is dated in the future")
    if manifest.device != device:
        out.append("the manifest was prepared for another device")
    if manifest.platform != platform:
        out.append(f"the manifest is for {manifest.platform}, this is {platform}")
    if (manifest.scope == "system") != system:
        out.append("a system manifest is installed with --system, and only then")
    folder = _folder_problem(directory)
    if folder:
        return [*out, folder]
    for role in manifest.roles:
        out += _role_problems(role, platform, manifest.scope)
        if role.label != ROLE_INFO[role.role]["label"] or [(u.kind, u.name) for u in role.units] != unit_names(role.role, platform):
            out.append(f"{role.role}: its label or unit files are not the ones this role installs")
            continue
        rendered = render(replace(role, units=()), platform, manifest.scope, getpass.getuser())
        for unit in role.units:
            data = staged_bytes(directory, unit.name)
            if data is None:
                out.append(f"{unit.name} is missing or not a plain file in the staging directory")
            elif hashlib.sha256(data).hexdigest() != unit.sha256:
                out.append(f"{unit.name} changed after it was prepared")
            elif data != rendered.get(unit.kind):
                out.append(f"{unit.name} does not match the manifest")
            bad = destination_problem(unit.destination, role.role, unit.name, platform, manifest.scope)
            if bad:
                out.append(bad)
    return out
