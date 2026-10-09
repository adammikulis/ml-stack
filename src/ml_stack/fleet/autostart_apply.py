"""`install` and `rollback`: a person puts staged units in place, loads them, and can undo it."""

from __future__ import annotations

import hashlib
import shlex
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ml_stack import authority, home
from ml_stack.files import read_json, write_json, writing
from ml_stack.http import ServerError, request_json
from ml_stack.lock import Busy, only_one
from ml_stack.log import say
from ml_stack.person import require_person
from ml_stack.sentinel.human import mint

from . import autostart_ledger
from .autostart_backends import Backend, backend_for, elevated
from .autostart_manifest import Manifest, parse
from .autostart_prepare import platform_of, read_manifest
from .autostart_verify import Context, problems

__all__ = ["ACTION", "Outcome", "Seams", "Work", "healthy", "install", "rollback"]

ACTION = "install autostart units"
Step = tuple[list[str], bool]
File = tuple[str, Path, Path | None, str]


@dataclass(frozen=True, slots=True)
class Outcome:
    """Whether the work was done, what to tell the person, and the command that undoes it."""

    ok: bool
    lines: tuple[str, ...]
    rollback: str = ""


def healthy(health: Mapping[str, Any]) -> bool:
    """Whether a role's health probe answers within its timeout; a role with no resident process passes."""
    if health.get("kind") != "http":
        return True
    end = time.monotonic() + float(health.get("timeout_s", 30))
    while True:
        try:
            request_json(str(health["url"]), timeout=3)
            return True
        except (ServerError, OSError):
            if time.monotonic() > end:
                return False
            time.sleep(1)


@dataclass(frozen=True, slots=True)
class Work:
    """The commands to run before and after placing files, and the files to place or remove."""

    pre: list[Step]
    files: list[File]
    post: list[Step]


@dataclass(frozen=True, slots=True)
class Seams:
    """What a test replaces: the service manager, the privilege prompt, the person's keyboard, the clock."""

    backend: Backend | None = None
    ask: Callable[[str, str], tuple[bool, str]] | None = None
    probe: Callable[[Mapping[str, Any]], bool] = healthy
    typed: Callable[[str], str] = input
    terminal: tuple[bool, bool] | None = None
    env: Mapping[str, str] | None = None
    now: float = 0.0
    device: str = ""
    platform: str = ""
    show: Callable[[str], None] = say


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _execute(work: Work, system: bool, backend: Backend, seams: Seams) -> str:
    """Run the steps and place the files; the failure text, or an empty string."""
    if system:
        parts = [("{ " + shlex.join(a) + "; true; }") for a, _ in work.pre]
        for act, dest, src, _ in work.files:
            parts += [shlex.join(["mkdir", "-p", str(dest.parent)]), shlex.join(["install", "-m", "0644", str(src), str(dest)])] \
                if act == "write" else [shlex.join(["rm", "-f", str(dest)])]
        parts += [shlex.join(a) for a, _ in work.post]
        ask = seams.ask or elevated
        done, why = ask(" && ".join(parts), "ml-stack needs permission to install its units")
        return "" if done else (why or "permission was not given")
    for argv, _ in work.pre:
        backend.runner(argv)
    for act, dest, src, digest in work.files:
        if act == "remove":
            dest.unlink(missing_ok=True)
            continue
        data = src.read_bytes() if src else b""
        if _sha(data) != digest:
            return f"{src} changed while it was being installed"
        with writing(dest) as temporary:
            temporary.write_bytes(data)
            temporary.chmod(0o644)
    for argv, must in work.post:
        code, out = backend.runner(argv)
        if code and must:
            return out or f"{argv[0]} failed"
    return ""


def _backup(manifest: Manifest, platform: str, backend: Backend) -> Path:
    """Copy every unit an install replaces into its backup folder and write the index."""
    folder = home.state("autostart", "backups", manifest.id)
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    entries = []
    for role in manifest.roles:
        for unit in role.units:
            if platform == "win32":
                code, out = backend.runner(["schtasks", "/Query", "/TN", role.label, "/XML"])
                old = out.encode("utf-16") if code == 0 else None
            else:
                old = Path(unit.destination).read_bytes() if Path(unit.destination).is_file() else None
            if old is not None:
                (folder / unit.name).write_bytes(old)
            entries.append({"role": role.role, "name": unit.name, "destination": unit.destination,
                            "had": old is not None, "sha256": _sha(old) if old is not None else ""})
    write_json(folder / "index.json", {"version": 1, "entries": entries})
    return folder


def _undo(manifest: Manifest, folder: Path, system: bool, backend: Backend, seams: Seams) -> str:
    """Unload the manifest's units and put back what the backup holds."""
    index = read_json(folder / "index.json", {})
    had = {(e["role"], e["name"]): e for e in index.get("entries", [])}
    pre = [(a, False) for r in manifest.roles for a in backend.unload(r)]
    files: list[File] = []
    post: list[Step] = []
    for role in manifest.roles:
        old = [had[(role.role, u.name)] for u in role.units if had.get((role.role, u.name), {}).get("had")]
        if manifest.platform != "win32":
            files += [("write", Path(e["destination"]), folder / e["name"], e["sha256"]) for e in old]
            files += [("remove", Path(u.destination), None, "") for u in role.units
                      if not had.get((role.role, u.name), {}).get("had")]
        if old:
            post += [(a, False) for a in backend.load(role, folder)]
    return _execute(Work(pre, files, post), system, backend, seams)


def _checked(manifest: Manifest, backend: Backend, seams: Seams) -> str:
    """Why the installed units are not running as prepared, or an empty string."""
    for role in manifest.roles:
        if manifest.platform != "win32":
            for unit in role.units:
                path = Path(unit.destination)
                if not path.is_file() or _sha(path.read_bytes()) != unit.sha256:
                    return f"{unit.destination} does not hold the prepared unit"
        if not backend.loaded(role):
            return f"{role.label} is not loaded"
        if not seams.probe(role.health):
            return f"{role.label} did not answer its health probe"
    return ""


def _command(verb: str, system: bool) -> str:
    return shlex.join([sys.executable, "-m", "ml_stack.fleet.autostart", verb, *(["--system"] if system else [])])


def _record(manifest: Manifest, ledger: dict[str, Any], now: float) -> None:
    for role in manifest.roles:
        before = {k: v for k, v in ledger["installed"].get(role.role, {}).items() if k != "previous"}
        ledger["installed"][role.role] = {
            "manifest": manifest.id, "at": now, "scope": manifest.scope, "platform": manifest.platform,
            "role": manifest.to_dict()["roles"][manifest.roles.index(role)], "previous": before}
    ledger["used"] = [*ledger["used"], manifest.id][-200:]
    ledger["last"] = {"id": manifest.id, "manifest": manifest.to_dict()}
    autostart_ledger.write(ledger)


def _summary(manifest: Manifest, show: Callable[[str], None]) -> None:
    show(f"manifest {manifest.id} prepared by {manifest.preparer.get('agent')}"
         f" for {manifest.platform} ({manifest.scope})")
    for role in manifest.roles:
        show(f"  {role.role}: {shlex.join(role.argv)}")
        for unit in role.units:
            show(f"    {unit.destination}  sha256 {unit.sha256[:12]}")


def install(path: Path | str, *, system: bool = False, seams: Seams | None = None) -> Outcome:
    """Install the units staged with ``path``'s manifest; only a person at a terminal may."""
    seam = seams or Seams()
    require_person(ACTION, seam.terminal, seam.env)
    path = Path(path)
    try:
        manifest = parse(read_manifest(path))
    except (OSError, ValueError) as exc:
        return Outcome(False, (f"cannot read the manifest: {exc}",))
    platform = platform_of(seam.platform)
    ledger = autostart_ledger.read()
    found = problems(manifest, path.parent, Context(seam.now or time.time(), seam.device or home.device_id(),
                                                    platform, system))
    if manifest.id in ledger["used"]:
        found.append("this manifest was already installed; prepare a new one")
    if found:
        return Outcome(False, tuple(found))
    _summary(manifest, seam.show)
    mint(ACTION, manifest.id, typed=seam.typed, terminal=seam.terminal, env=seam.env)
    backend = seam.backend or backend_for(platform, manifest.scope)
    try:
        with only_one(home.state("autostart", "apply.lock"), wait=False, note="autostart install"):
            return _apply(manifest, path.parent, system, backend, replace(seam, backend=backend))
    except Busy:
        return Outcome(False, ("another autostart install or rollback is running",))


def _apply(manifest: Manifest, directory: Path, system: bool, backend: Backend, seams: Seams) -> Outcome:
    folder = _backup(manifest, manifest.platform, backend)
    for role in manifest.roles:
        Path(role.workdir).mkdir(mode=0o700, parents=True, exist_ok=True)
    pre = [(a, False) for r in manifest.roles for a in backend.unload(r)]
    files: list[File] = [] if manifest.platform == "win32" else [
        ("write", Path(u.destination), directory / u.name, u.sha256) for r in manifest.roles for u in r.units]
    post: list[Step] = [(a, True) for r in manifest.roles for a in backend.load(r, directory)]
    if manifest.platform == "linux" and manifest.scope == "user":
        post.append((["loginctl", "enable-linger"], False))
    error = _execute(Work(pre, files, post), system, backend, seams)
    if not error:
        error = _checked(manifest, backend, seams)
    if error:
        undone = _undo(manifest, folder, system, backend, seams)
        return Outcome(False, (error, "the previous units were put back" if not undone else f"the previous units could not be put back: {undone}"))
    now = seams.now or time.time()
    _record(manifest, autostart_ledger.read(), now)
    authority.record("autostart.install", manifest=manifest.id, scope=manifest.scope, platform=manifest.platform,
                     roles=[r.role for r in manifest.roles], preparer=manifest.preparer,
                     units=[{"destination": u.destination, "sha256": u.sha256} for r in manifest.roles for u in r.units])
    lines = tuple(f"installed {r.role} ({r.label}) and it answered" for r in manifest.roles)
    return Outcome(True, lines, _command("rollback", system))


def rollback(*, system: bool = False, seams: Seams | None = None) -> Outcome:
    """Restore what the last install replaced and unload its units; only a person at a terminal may."""
    seam = seams or Seams()
    require_person("roll back autostart units", seam.terminal, seam.env)
    ledger = autostart_ledger.read()
    last = ledger["last"]
    if not last:
        return Outcome(False, ("nothing was installed from a manifest",))
    manifest = parse(last["manifest"])
    mint("roll back autostart units", manifest.id, typed=seam.typed, terminal=seam.terminal, env=seam.env)
    backend = seam.backend or backend_for(manifest.platform, manifest.scope)
    try:
        with only_one(home.state("autostart", "apply.lock"), wait=False, note="autostart rollback"):
            error = _undo(manifest, home.state("autostart", "backups", manifest.id), system, backend, seam)
            if error:
                return Outcome(False, (f"could not roll back: {error}",))
            for role in manifest.roles:
                before = ledger["installed"].get(role.role, {}).get("previous")
                ledger["installed"].pop(role.role, None)
                if before:
                    ledger["installed"][role.role] = before
            ledger["last"] = ""
            autostart_ledger.write(ledger)
            authority.record("autostart.rollback", manifest=manifest.id, roles=[r.role for r in manifest.roles])
            return Outcome(True, tuple(f"rolled back {r.role}" for r in manifest.roles))
    except Busy:
        return Outcome(False, ("another autostart install or rollback is running",))

