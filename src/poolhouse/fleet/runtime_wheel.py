"""Build immutable runtime wheels from a git revision."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path

from poolhouse import runtime
from poolhouse.files import writing
from poolhouse.fleet.wheel_provenance import ORIGIN, stamp, wheel_commit
from poolhouse.installed import extras
from poolhouse.lock import only_one
from poolhouse.net import packages, uvinstall
from poolhouse.safenames import unpack


def source_checkout() -> Path | None:
    """Return the tracked source checkout recorded in the installed runtime."""
    marker = Path(__file__).with_name(ORIGIN)
    if not marker.is_file():
        return None
    source = Path(marker.read_text(encoding="utf-8").strip())
    return source if source.is_absolute() and (source / ".git").exists() else None


def cache_wheel(wheel: Path, commit: str, *, prefix: Path | None = None) -> Path:
    """Retain a stamped runtime wheel in the owning installation prefix."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("runtime wheel cache requires a full commit")
    if wheel_commit(wheel) != commit:
        raise ValueError("runtime wheel commit does not match its cache key")
    target = Path(prefix or sys.prefix) / "poolhouse-wheels" / commit / wheel.name
    with writing(target) as temporary:
        shutil.copyfile(wheel, temporary)
    return target


def current_wheel() -> Path | None:
    """Return the cached wheel matching the imported immutable runtime."""
    marker = Path(__file__).with_name("built-from")
    if not marker.is_file():
        return None
    commit = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise OSError("the installed runtime has no full commit provenance")
    found = list((Path(sys.prefix) / "poolhouse-wheels" / commit).glob("poolhouse-*.whl"))
    if len(found) != 1:
        raise OSError("the current runtime wheel is unavailable; update the installed runtime before installing libraries")
    try:
        if wheel_commit(found[0]) != commit:
            raise OSError("the cached wheel does not match the installed runtime")
    except (UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        raise OSError("the cached runtime wheel is invalid") from exc

    return found[0]


@dataclass(frozen=True, slots=True)
class Target:
    """Where and by whom a runtime is built: its prefix, the creation record written first and the host Python."""

    prefix: Path | None = None
    creator: dict | None = None
    host: Path | None = None


def build(checkout: Path, commit: str, stage: Path, *, timeout: float, target: Target | None = None) -> runtime.Runtime:
    """Build an isolated runtime from the committed snapshot of one revision; its source stays in `stage`."""
    target = target or Target()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("a runtime is built from a full commit")
    snapshot = stage / "source.zip"
    _run(["git", "-C", str(checkout), "archive", "--format=zip", "--output", str(snapshot), commit], timeout)
    source, wheels = stage / "source", stage / "wheels"
    unpack(snapshot, source)
    _run([str(target.host or sys.executable), "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(source)], timeout)
    found = list(wheels.glob("poolhouse-*.whl"))
    if len(found) != 1:
        raise ValueError("source revision must build exactly one poolhouse wheel")
    stamp(found[0], commit, checkout)
    return prepare(found[0], commit, timeout=timeout, target=target)


def prepare(wheel: Path, commit: str, *, timeout: float, target: Target | None = None) -> runtime.Runtime:
    """Build and verify a separate owned Python prefix for a stamped wheel; the target's creation record is written into it first."""
    target = target or Target()
    if wheel_commit(wheel) != commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("runtime wheel must match its full source revision")
    with zipfile.ZipFile(wheel) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(records) != 1:
            raise ValueError("runtime wheel must contain one distribution metadata record")
        metadata = BytesParser().parsebytes(archive.read(records[0]))
        if metadata["Name"] != "poolhouse":
            raise ValueError("runtime wheel must provide poolhouse")
        version = metadata["Version"]
    if not version:
        raise ValueError("runtime wheel has no version")
    root = runtime.directory()
    runtime.plain(root)
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime.protect(root.parent)
    root.mkdir(mode=0o700, exist_ok=True)
    runtime.protect(root)
    runtime._owned(root)
    with only_one(root / "install.lock"):
        family = root / commit
        if family.is_symlink():
            raise OSError("runtime revision directory cannot be a symbolic link")
        family.mkdir(mode=0o700, exist_ok=True)
        runtime.protect(family)
        runtime._owned(family)
        if target.prefix is not None and target.prefix.parent != family:
            raise ValueError("a runtime prefix belongs in its own revision directory")
        chosen = runtime.Runtime(target.prefix or family / uuid.uuid4().hex, commit, version, runtime.identity())
        chosen.prefix.mkdir(mode=0o700)
        runtime.protect(chosen.prefix)
        if target.creator is not None:
            (chosen.prefix / "created.json").write_text(json.dumps(target.creator), encoding="utf-8")
        done = False
        try:
            _run([str(target.host or sys.executable), "-m", "venv", str(chosen.prefix)], timeout)
            cached = cache_wheel(wheel, commit, prefix=chosen.prefix)
            spec = f"poolhouse[{extras()}] @ {cached.as_uri()}"
            _install(chosen.python, spec, timeout)
            runtime.verify(chosen)
            done = True
        finally:
            if not done:
                shutil.rmtree(chosen.prefix, ignore_errors=True)
        return chosen


def _install(python: Path, spec: str, timeout: float) -> None:
    """Install the spec with uv's shared cache (cloned files, no bytecode written) when it can; otherwise with pip."""
    done = uvinstall.install(python, spec, timeout=timeout, environment=runtime.installer_environment())
    if done is None:
        _run([str(python), "-m", "pip", "install", spec], timeout)
    elif done.returncode:
        raise ValueError(f"{done.stdout}{done.stderr}".strip() or f"uv exited {done.returncode}")


def _run(argv: list[str], timeout: float) -> str:
    done = (packages.run(argv[0], argv[3:], timeout=timeout, env=runtime.installer_environment()) if argv[1:3] == ["-m", "pip"]
            else subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=runtime.installer_environment()))
    output = f"{done.stdout}{done.stderr}".strip()
    if done.returncode:
        raise ValueError(output or f"{argv[0]} exited {done.returncode}")
    return output
