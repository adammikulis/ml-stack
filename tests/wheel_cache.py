"""The built wheel, made once for every test process and run that shares a checkout state.

Building the wheel and installing it into a target directory took 19 to 41 s the first time a
worker reached a test that needs real distribution metadata. The inputs are the sources the
wheel is built from, so the result is kept under the temporary directory by a fingerprint of
those inputs (path, size and modification time): a worker, or a later run, with the same
sources finds the finished wheel and builds nothing; a changed source builds a new one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = ("src/ml_stack", "contracts", "patches")
FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
KEEP_SECONDS = 86400


def _inputs() -> list[Path]:
    found = [ROOT / name for name in FILES]
    for tree in TREES:
        found += sorted(p for p in (ROOT / tree).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return found


def fingerprint() -> str:
    """A digest of every file the wheel is built from: its path, size and modification time."""
    digest = hashlib.sha256()
    for path in _inputs():
        try:
            info = path.stat()
        except OSError:
            continue
        digest.update(f"{path.relative_to(ROOT)}:{info.st_size}:{info.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:20]


def _forget_old(base: Path) -> None:
    for entry in base.iterdir():
        try:
            if time.time() - entry.stat().st_mtime > KEEP_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def _build(into: Path) -> None:
    environment = {**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "--no-cache-dir", "--wheel-dir", str(into), str(ROOT)],
                   env=environment, check=True, capture_output=True, text=True, timeout=180)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    "--target", str(into / "installed"), str(next(into.glob("*.whl")))],
                   env=environment, check=True, capture_output=True, text=True, timeout=180)


def built() -> Path:
    """The directory holding the wheel (``*.whl``) and, under ``installed/``, its contents
    installed with pip. Built on first use for these sources, found afterwards."""
    owner = os.getuid() if hasattr(os, "getuid") else 0
    base = Path(tempfile.gettempdir()) / f"ml-stack-wheel-cache-{owner}"
    base.mkdir(exist_ok=True)
    final = base / fingerprint()
    if not final.is_dir():
        _forget_old(base)
        staging = Path(tempfile.mkdtemp(prefix="building-", dir=base))
        _build(staging)
        try:
            staging.rename(final)
        except OSError:  # another worker finished first
            shutil.rmtree(staging, ignore_errors=True)
    return final
