"""The node binary of a runtime: built from the source tree, recorded with its checksum, verified before every start."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ml_stack.platform import is_windows
from ml_stack.files import read_json, write_json, writing
from ml_stack.runtime_store import MARK

DIRECTORY = "node"
RECORD = "node.json"
BUILD_TIMEOUT = 1800.0
CHUNK = 1 << 20


class NodeBinaryError(OSError):
    """The node binary is missing, was built wrongly or does not match its recorded checksum."""


def name() -> str:
    """The binary's file name: ``.exe`` on Windows."""
    return "poolside-node.exe" if is_windows() else "poolside-node"


def target() -> str:
    """The platform this binary was built for."""
    return f"{sys.platform}-{platform.machine().lower()}"


def sha256(path: Path) -> str:
    """The SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def location(prefix: Path) -> Path:
    """Where a runtime tree keeps its node binary."""
    return prefix / DIRECTORY / name()


def build(source: Path, *, cache: Path | None = None, timeout: float = BUILD_TIMEOUT) -> Path:
    """Build the release `poolside-node` of a source tree and return the binary.

    ``cache`` is a cargo target directory shared between builds so a second build compiles only what changed.
    """
    workspace = source / "app"
    if not (workspace / "poolside-node" / "Cargo.toml").is_file():
        raise NodeBinaryError(f"{source} has no app/poolside-node crate")
    env = {**os.environ, **({"CARGO_TARGET_DIR": str(cache)} if cache else {})}
    done = subprocess.run(["cargo", "build", "--release", "--locked", "-p", "poolside-node"], cwd=workspace,
                          capture_output=True, text=True, timeout=timeout, env=env)
    if done.returncode:
        raise NodeBinaryError(f"cargo build failed: {(done.stderr or done.stdout).strip()[-600:]}")
    made = (cache or workspace / "target") / "release" / name()
    if not made.is_file():
        raise NodeBinaryError(f"cargo wrote no {name()}")
    return made


def describe(binary: Path, *, commit: str = "") -> dict:
    """The record that goes beside a binary: its checksum, size, platform and the commit it was built from."""
    return {"sha256": sha256(binary), "bytes": binary.stat().st_size, "target": target(), "commit": commit}


def install(prefix: Path, binary: Path, *, commit: str = "") -> dict:
    """Copy a binary into a runtime tree, executable, with its record; returns the record."""
    where = location(prefix)
    where.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with writing(where) as temporary:
        shutil.copyfile(binary, temporary)
        temporary.chmod(0o700)
    record = describe(where, commit=commit)
    write_json(where.parent / RECORD, record)
    return record


def record_of(prefix: Path) -> dict:
    """The record of a runtime tree's binary; NodeBinaryError when there is none."""
    try:
        row = json.loads((location(prefix).parent / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NodeBinaryError(f"{prefix} has no node binary record") from exc
    if not isinstance(row, dict) or not isinstance(row.get("sha256"), str):
        raise NodeBinaryError(f"{prefix} has an invalid node binary record")
    return row


def verified(prefix: Path, *, expect: str = "") -> Path:
    """The binary of a runtime tree once its checksum matches its record (and ``expect`` when given)."""
    binary, row = location(prefix), record_of(prefix)
    if not binary.is_file() or binary.is_symlink():
        raise NodeBinaryError(f"{binary} is missing")
    if row.get("target") != target():
        raise NodeBinaryError(f"{binary} was built for {row.get('target')}, not {target()}")
    found = sha256(binary)
    marked = read_json(prefix / MARK, {}).get("node_sha256", found)
    if found != row["sha256"] or found != marked or (expect and found != expect):
        raise NodeBinaryError(f"{binary} does not match its recorded checksum; refusing to start it")
    return binary
