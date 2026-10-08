"""Private relocated role homes and exact fixture state paths."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
from pathlib import Path

from ml_stack.activity.source_snapshot import validate_storage
from ml_stack.sandbox.seatbelt import quote

MAX_CONTENT = 2 * 1024 * 1024 * 1024
MAX_NODES = 16384
STATE_FILES = ("broker.json", "broker-leases.json", "broker-handoff.json")
CASE_VARIANTS = {
    "test_actual_immutable_runtime_receipt_is_verified_outside_the_socket_deadline[holder]": "normal",
    "test_actual_immutable_runtime_receipt_is_verified_outside_the_socket_deadline[broker]": "normal",
    "test_actual_owned_preparation_and_mutual_session_exchange_keep_original_lease": "normal",
    "test_slow_independent_verification_is_cancelled_without_waiting_or_orphaning[slow]": "slow",
    "test_completed_preparation_cannot_extend_a_changed_broker_generation[stale]": "stale",
}
CASE_FILE = "tests/test_holder_protocol.py::"
FENCE_NODE = "tests/test_kernel_role_home.py::test_role_home_atomic_state_and_immutable_fences"
IMPORT_NODE = "tests/test_kernel_role_home.py::test_fixed_normal_role_imports_and_runtime_verification"


def checked_time(deadline: float) -> None:
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("holder role: preparation deadline exceeded")


def file_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns)


def checked_home(home: Path) -> None:
    info = home.lstat()
    if (home.resolve(strict=True) != home or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700
            or len(os.fsencode(home / "holder-channels/999999999.sock")) >= 104):
        raise RuntimeError("holder role: state home is redirected, public or too long")


def checked_inventory(variant: dict, home: Path) -> tuple[Path, list[dict]]:
    checked_home(home)
    runtime, records = variant["runtime"], variant["files"]
    source = Path(runtime["prefix"])
    for part in (runtime["identity"], runtime["commit"], source.name):
        if not isinstance(part, str) or not part or part in {".", ".."} or Path(part).name != part:
            raise RuntimeError("holder role: invalid runtime family component")
    if source.resolve(strict=True) != source or not 1 <= len(records) <= MAX_NODES:
        raise RuntimeError("holder role: source or inventory node bound changed")
    seen, total = set(), 0
    for record in records:
        relative = record["path"]
        if (not isinstance(relative, str) or not relative or relative in seen
                or Path(relative).is_absolute() or ".." in Path(relative).parts
                or str(Path(relative)) != relative or record["type"] not in {"directory", "file", "symlink"}):
            raise RuntimeError("holder role: invalid copied inventory path")
        seen.add(relative)
        if record["type"] == "file":
            if type(record["size"]) is not int or record["size"] < 0:
                raise RuntimeError("holder role: invalid copied file size")
            total += record["size"]
    if "." not in seen or total > MAX_CONTENT:
        raise RuntimeError("holder role: root or content bound changed")
    return source, records


def state_rules(home: Path, *, slow: bool = False) -> list[str]:
    """Grant only exact fixture state targets and finite atomic sibling names."""
    checked_home(home)
    targets = [home / name for name in (*STATE_FILES, *(('slow-verifier.pid',) if slow else ()))]
    literals = " ".join("(literal " + quote(str(path)) + ")" for path in targets)
    temporary = '(regex (string-append "^" (regex-quote ' + quote(str(home / "tmp")) + ') '
    temporary += " ".join('"[a-z0-9_]"' for _ in range(8))
    temporary += ' (regex-quote ".tmp") "$"))'
    return ["(allow file-read* file-write* " + literals + " " + temporary + ")",
            "(allow file-read-data file-read-metadata (literal " + quote(str(home)) + "))"]


def allocate_root(protected: tuple[Path, ...]) -> Path:
    """Create a retained private bank outside protected state and test scratch."""
    base = validate_storage(Path("/private/tmp"), protected)
    root = Path(tempfile.mkdtemp(prefix="mlr-", dir=base))
    info = root.lstat()
    if (root.resolve(strict=True) != root or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o077 or len(os.fsencode(root)) > 32):
        raise RuntimeError("holder role: private allocation is not plain, short and owned")
    return root


def copy_file(source: Path, target: Path, record: dict, deadline: float) -> None:
    """Copy one pinned regular file without shared inode or link traversal."""
    original = source.lstat()
    if (not stat.S_ISREG(original.st_mode) or original.st_uid != os.getuid() or original.st_nlink != 1
            or original.st_mode & 0o222 or original.st_size != record["size"]
            or stat.S_IMODE(original.st_mode) != record["mode"]):
        raise RuntimeError("holder role: source file identity changed")
    reader = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if file_identity(os.fstat(reader)) != file_identity(original):
            raise RuntimeError("holder role: source file changed while opening")
        writer = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            digest, copied = hashlib.sha256(), 0
            while True:
                checked_time(deadline)
                block = os.read(reader, 1024 * 1024)
                if not block:
                    break
                copied += len(block)
                if copied > record["size"]:
                    raise RuntimeError("holder role: source grew while copying")
                digest.update(block)
                pending = memoryview(block)
                while pending:
                    checked_time(deadline)
                    written = os.write(writer, pending)
                    if written <= 0:
                        raise OSError("holder role: copy made no write progress")
                    pending = pending[written:]
            if copied != record["size"] or digest.hexdigest() != record["sha256"]:
                raise RuntimeError("holder role: copied source content changed")
            os.fchmod(writer, record["mode"])
            os.fsync(writer)
            actual = os.fstat(writer)
            if actual.st_nlink != 1 or (actual.st_dev, actual.st_ino) == (original.st_dev, original.st_ino):
                raise RuntimeError("holder role: copy shares its source inode")
            if file_identity(os.fstat(reader)) != file_identity(original) or file_identity(source.lstat()) != file_identity(original):
                raise RuntimeError("holder role: source changed during copy")
        finally:
            os.close(writer)
    finally:
        os.close(reader)


def copy_variant(variant: dict, home: Path, deadline: float) -> dict:
    """Copy one previously verified complete variant into its matching state home."""
    checked_time(deadline)
    source, records = checked_inventory(variant, home)
    runtime = variant["runtime"]
    prefix = home / "runtimes" / runtime["identity"] / runtime["commit"] / source.name
    current = home
    for part in ("runtimes", runtime["identity"], runtime["commit"], source.name):
        current = current / part
        current.mkdir(mode=0o700, exist_ok=False)
    total = 0
    directories = []
    for record in sorted(records, key=lambda value: (len(Path(value["path"]).parts), value["path"])):
        checked_time(deadline)
        relative = record["path"]
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise RuntimeError("holder role: copied inventory path is not relative")
        original = source if relative == "." else source / relative
        target = prefix if relative == "." else prefix / relative
        if record["type"] == "directory":
            if relative != ".":
                target.mkdir(mode=0o700)
            directories.append((target, record["mode"]))
        elif record["type"] == "file":
            total += record["size"]
            if total > MAX_CONTENT:
                raise RuntimeError("holder role: copied content bound exceeded")
            copy_file(original, target, record, deadline)
        elif record["type"] == "symlink":
            if not original.is_symlink() or str(original.readlink()) != record["target"]:
                raise RuntimeError("holder role: source link changed")
            target.symlink_to(record["target"])
            target.chmod(record["mode"], follow_symlinks=False)
        else:
            raise RuntimeError("holder role: unsupported copied node")
    for directory, mode in reversed(directories):
        checked_time(deadline)
        directory.chmod(mode)
    descriptor = {**runtime, "prefix": str(prefix), "python": str(prefix / "bin/python")}
    copied_records = json.loads(json.dumps(records))
    for record in copied_records:
        if record["type"] != "file":
            target = prefix if record["path"] == "." else prefix / record["path"]
            record["size"] = target.lstat().st_size
    return {"runtime": descriptor, "files": copied_records}


def selected_cases(command: list[str]) -> dict[str, str]:
    selected = {}
    for value in command:
        if value in {FENCE_NODE, IMPORT_NODE}:
            selected[value] = "normal"
            continue
        if not value.startswith(CASE_FILE):
            continue
        case = value[len(CASE_FILE):]
        base = case.split("[", 1)[0]
        candidates = {name: variant for name, variant in CASE_VARIANTS.items()
                      if name == case or (case == base and name.split("[", 1)[0] == base)}
        if "[" in case and base in {name.split("[", 1)[0] for name in CASE_VARIANTS} and not candidates:
            raise RuntimeError("holder role: unreviewed fixture parameter")
        selected.update({CASE_FILE + name: variant for name, variant in candidates.items()})
    return dict(sorted(selected.items()))


def copy_cases(manifest: dict, command: list[str], protected: tuple[Path, ...], deadline: float) -> tuple[Path, dict]:
    """Allocate distinct retained homes for the exact selected fixture cases."""
    cases = selected_cases(command)
    if not cases:
        raise RuntimeError("holder role: no reviewed immutable fixture case selected")
    checked_time(deadline)
    root, entries = allocate_root(protected), {}
    for index, (case, variant) in enumerate(cases.items()):
        checked_time(deadline)
        home = root / str(index)
        home.mkdir(mode=0o700)
        channels = home / "holder-channels"
        channels.mkdir(mode=0o700)
        copied = copy_variant(manifest["variants"][variant], home, deadline)
        entries[case] = {"home": str(home), "variant": variant, "prepared": copied}
    original_files = {name: str(Path(value["runtime"]["prefix"]) / "pyvenv.cfg")
                      for name, value in manifest["variants"].items()}
    return root, {"version": 1, "source_commit": manifest["source_commit"], "cases": entries,
                  "original_variant_files": original_files}
