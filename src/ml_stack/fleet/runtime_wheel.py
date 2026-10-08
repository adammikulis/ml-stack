"""Build immutable runtime wheels from a git revision."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from email.parser import BytesParser
from pathlib import Path

from ml_stack import runtime
from ml_stack.files import writing
from ml_stack.fleet.wheel_provenance import ORIGIN, stamp, wheel_commit
from ml_stack.lock import only_one
from ml_stack.net import packages
from ml_stack.safenames import unpack


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
    target = Path(prefix or sys.prefix) / "ml-stack-wheels" / commit / wheel.name
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
    found = list((Path(sys.prefix) / "ml-stack-wheels" / commit).glob("ml_stack-*.whl"))
    if len(found) != 1:
        raise OSError("the current runtime wheel is unavailable; update the installed runtime before installing libraries")
    try:
        if wheel_commit(found[0]) != commit:
            raise OSError("the cached wheel does not match the installed runtime")
    except (UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        raise OSError("the cached runtime wheel is invalid") from exc

    return found[0]


def build(checkout: Path, commit: str, stage: Path, *, timeout: float) -> runtime.Runtime:
    """Build an isolated runtime from the committed snapshot of one revision; its source stays in `stage`."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("a runtime is built from a full commit")
    snapshot = stage / "source.zip"
    _run(["git", "-C", str(checkout), "archive", "--format=zip", "--output", str(snapshot), commit], timeout)
    source, wheels = stage / "source", stage / "wheels"
    unpack(snapshot, source)
    _run([str(_host_python()), "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(source)], timeout)
    found = list(wheels.glob("ml_stack-*.whl"))
    if len(found) != 1:
        raise ValueError("source revision must build exactly one ml-stack wheel")
    stamp(found[0], commit, checkout)
    return prepare(found[0], commit, timeout=timeout)


def prepare(wheel: Path, commit: str, *, timeout: float) -> runtime.Runtime:
    """Build and verify a separate owned Python prefix for a stamped wheel."""
    if wheel_commit(wheel) != commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("runtime wheel must match its full source revision")
    with zipfile.ZipFile(wheel) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(records) != 1:
            raise ValueError("runtime wheel must contain one distribution metadata record")
        metadata = BytesParser().parsebytes(archive.read(records[0]))
        if metadata["Name"] != "ml-stack":
            raise ValueError("runtime wheel must provide ml-stack")
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
        chosen = runtime.Runtime(family / uuid.uuid4().hex, commit, version, runtime.identity())
        chosen.prefix.mkdir(mode=0o700)
        runtime.protect(chosen.prefix)
        try:
            _run([str(_host_python()), "-m", "venv", str(chosen.prefix)], timeout)
            cached = cache_wheel(wheel, commit, prefix=chosen.prefix)
            from ml_stack.installed import extras
            spec = f"ml-stack[{extras()}] @ {cached.as_uri()}"
            _run([str(chosen.python), "-m", "pip", "install", spec], timeout)
            runtime.verify(chosen)
        except BaseException:
            shutil.rmtree(chosen.prefix, ignore_errors=True)
            raise
        return chosen


def _host_python() -> Path:
    from .environment import Environment
    base = Environment(runtime.directory() / "bootstrap").host_python()
    if base is None:
        raise OSError("install Python 3.13 to prepare an isolated runtime")
    return base


def _run(argv: list[str], timeout: float) -> str:
    done = (packages.run(argv[0], argv[3:], timeout=timeout, env=runtime.installer_environment()) if argv[1:3] == ["-m", "pip"]
            else subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=runtime.installer_environment()))
    output = f"{done.stdout}{done.stderr}".strip()
    if done.returncode:
        raise ValueError(output or f"{argv[0]} exited {done.returncode}")
    return output
