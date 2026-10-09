"""The tree a test shard runs: packed on the sending side, checked and unpacked on the receiving one."""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

MOST_PACKED = 24 << 20
MOST_UNPACKED = 256 << 20
MOST_MEMBERS = 20000
EXECUTABLE = 0o111
SHIPPED = re.compile(r"^(src|tests|scripts|contracts|patches|packaging|docs|app)/|^[^/]+$")
"""The top-level places a shard may carry; hidden directories such as .git are never among them."""


class TreeError(ValueError):
    """A tree that cannot be packed, or that a receiving device refuses to unpack."""


def tracked_paths(root: Path) -> list[str]:
    """Tracked and untracked-but-not-ignored files under ``root`` that exist now, sorted."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root,
                         capture_output=True, check=True).stdout
    names = sorted({name.decode() for name in out.split(b"\0") if name})
    return [name for name in names if (root / name).is_file() and not (root / name).is_symlink()
            and SHIPPED.match(name) and not name.startswith(".")]


def pack(root: Path) -> tuple[bytes, str]:
    """The gzip tar of the tree at ``root`` and the sha256 of those bytes."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=6) as archive:
        for name in tracked_paths(root):
            info = archive.gettarinfo(str(root / name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if info.mode & EXECUTABLE else 0o644
            with (root / name).open("rb") as handle:
                archive.addfile(info, handle)
    data = buffer.getvalue()
    if len(data) > MOST_PACKED:
        raise TreeError(f"the packed tree is {len(data)} bytes; at most {MOST_PACKED} travel in one request")
    return data, hashlib.sha256(data).hexdigest()


def clean_name(name: str) -> str:
    """``name`` as a relative POSIX path that stays inside the tree, or TreeError."""
    path = PurePosixPath(name)
    bad = (not name or "\0" in name or "\\" in name or path.is_absolute() or ".." in path.parts
           or ":" in path.parts[0] or name != path.as_posix() or name.startswith("."))
    if bad or not SHIPPED.match(name):
        raise TreeError(f"the tree names a path a shard does not carry: {name[:80]!r}")
    return name


def members(data: bytes, wanted: str) -> list[tarfile.TarInfo]:
    """The members of the packed tree once its digest is ``wanted`` and each one is a plain file."""
    if len(data) > MOST_PACKED:
        raise TreeError("the packed tree is larger than a shard may carry")
    if hashlib.sha256(data).hexdigest() != wanted:
        raise TreeError("the tree does not match its digest")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            found = archive.getmembers()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise TreeError(f"the tree is not a readable archive: {exc}") from None
    if len(found) > MOST_MEMBERS or sum(member.size for member in found) > MOST_UNPACKED:
        raise TreeError("the tree holds more than a shard may carry")
    for member in found:
        if not member.isreg():
            raise TreeError(f"the tree holds something other than a plain file: {member.name[:80]!r}")
        clean_name(member.name)
    return found


def unpack(data: bytes, wanted: str, destination: Path) -> list[str]:
    """Write the verified tree under ``destination``; the names written."""
    found = members(data, wanted)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in found:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise TreeError(f"the tree escapes its folder: {member.name[:80]!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise TreeError(f"cannot read {member.name[:80]!r}")
            target.write_bytes(source.read())
            target.chmod(0o755 if member.mode & EXECUTABLE else 0o644)
    return [member.name for member in found]
