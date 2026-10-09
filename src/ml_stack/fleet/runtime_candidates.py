"""Owned immutable artifacts eligible for local agent runtime repair."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ml_stack.credentials.environment import child_environment
from ml_stack.files import promote
from ml_stack.graph.store import GraphStore
from ml_stack.lock import only_one
from ml_stack.safenames import unpack

HEX = re.compile(r"[0-9a-f]+\Z")


def runtime_environment():
    """Return a secret-free environment for an independent immutable runtime."""
    omitted = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "_MEIPASS2", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}
    env = {key: value for key, value in child_environment().items()
           if key not in omitted and not key.startswith("_PYI_")}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def _hex(value, length):
    if not isinstance(value, str) or len(value) != length or not HEX.fullmatch(value):
        raise ValueError("Candidate identity must use full lowercase hexadecimal values.")
    return value


def _private(path, root):
    path = Path(path).absolute()
    root = Path(root).absolute()
    nearest = next(parent for parent in (root, *root.parents) if parent.exists() or parent.is_symlink())
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("Candidate storage requires plain owned directories.")
            if (parent == nearest or root in parent.parents) and (info.st_uid != os.getuid() or info.st_mode & 0o022):
                raise ValueError("Candidate storage parents must be owned and protected from other writers.")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Candidate storage must be account-private.")
    return path


def digest(path):
    """Return the SHA256 of a plain owned artifact file."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Candidate artifacts must be plain account-owned files.")
        return hashlib.file_digest(source, "sha256").hexdigest()


def _verify(archive, commit):
    if sys.platform != "darwin":
        raise ValueError("Local signed agent runtime repair requires a macOS app artifact.")
    with tempfile.TemporaryDirectory(prefix="runtime-candidate-") as temporary:
        unpack(archive, Path(temporary))
        apps = list(Path(temporary).glob("*.app"))
        if len(apps) != 1:
            raise ValueError("Candidate must contain exactly one signed app bundle.")
        app = apps[0]
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)],
                       check=True, capture_output=True, timeout=30)
        executable = app / "Contents" / "MacOS" / "ml-stack-headless"
        if executable.is_symlink() or not executable.is_file():
            raise ValueError("Candidate has no plain headless runtime.")
        executable.chmod(executable.stat().st_mode | 0o100)
        observed = subprocess.run([str(executable), "--check-agent-runtime"], check=True,
                                  capture_output=True, text=True, timeout=30, env=runtime_environment())
        report = json.loads(observed.stdout)
        if (not isinstance(report, dict) or report.get("commit") != commit
                or report.get("platform") != sys.platform
                or report.get("machine") != platform.machine()
                or report.get("runtime_ready") is not True):
            raise ValueError("Candidate agent readiness or source/platform identity does not match.")
        return app.name


class Candidates:
    def __init__(self, root):
        self.path = _private(Path(root) / "runtime-candidates", root)

    def _store(self):
        database = self.path / "candidates.db"
        self._guard()
        return GraphStore(database)

    def _guard(self):
        _private(self.path, self.path.parent)
        for path in self.path.glob("candidates.*"):
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o022):
                raise ValueError("Candidate registry and auxiliary files must be plain owned files.")

    def register(self, archive, commit, sha256):
        """Copy, verify and record a reviewed immutable app artifact."""
        commit, sha256 = _hex(commit, 40), _hex(sha256, 64)
        self._guard()
        with only_one(self.path / "candidates.lock"):
            if digest(archive) != sha256:
                raise ValueError("Candidate archive digest does not match the reviewed artifact.")
            target = self.path / (sha256 + ".zip")
            if target.is_symlink():
                raise ValueError("Candidate archive cannot be a symbolic link.")
            if not target.exists():
                with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as output:
                    temporary = Path(output.name)
                try:
                    shutil.copyfile(archive, temporary)
                    if digest(temporary) != sha256:
                        raise ValueError("Candidate changed while being copied.")
                    promote(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            if digest(target) != sha256:
                raise ValueError("Stored candidate digest does not match.")
            name = _verify(target, commit)
            row = {"commit": commit, "sha256": sha256, "platform": sys.platform,
                   "machine": platform.machine(), "app": name, "created": time.time()}
            with self._store() as graph:
                existing = next((node for node in graph.nodes("runtime-candidate") if node["id"] == sha256), None)
                if existing and existing["attrs"]["commit"] != commit:
                    raise ValueError("Candidate identity is already bound to another source.")
                graph.upsert_node({"id": sha256, "kind": "runtime-candidate", "label": commit, "attrs": row})
            return row

    def select(self, app_name):
        """Return a reverified matching immutable candidate, or None."""
        self._guard()
        with only_one(self.path / "candidates.lock"), self._store() as graph:
            rows = sorted((node["attrs"] for node in graph.nodes("runtime-candidate")),
                          key=lambda row: row["created"], reverse=True)
            for row in rows:
                if (row["platform"], row["machine"], row["app"]) != (sys.platform, platform.machine(), app_name):
                    continue
                archive = self.path / (_hex(row["sha256"], 64) + ".zip")
                if digest(archive) != row["sha256"]:
                    raise ValueError("Registered candidate artifact has changed.")
                _verify(archive, _hex(row["commit"], 40))
                return row
        return None
