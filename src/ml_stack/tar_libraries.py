"""Contained shared-library aliases in release tar archives."""

from __future__ import annotations

import posixpath
import re
import tarfile
from itertools import islice
from pathlib import Path, PurePosixPath, PureWindowsPath

from ml_stack.safenames import MOST_ENTRIES, MOST_UNPACKED, Unsafe, safe_join

LIBRARY = re.compile(r".+\.so(?:\.\d+)*$")


def members(tf: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, tarfile.TarInfo]]:
    """Archive entries paired with the regular files their library aliases name."""
    listed = list(islice(tf, MOST_ENTRIES + 1))
    if len(listed) > MOST_ENTRIES:
        raise Unsafe("the archive has too many entries")
    index: dict[str, tarfile.TarInfo] = {}
    for item in listed:
        safe_join(Path("/archive"), item.name.rstrip("/"))
        name = str(PurePosixPath(item.name))
        if name in index:
            raise Unsafe(f"{name!r} occurs more than once")
        if not (item.isfile() or item.isdir() or item.issym()):
            raise Unsafe(f"{name!r} is a hard link or device")
        index[name] = item
    for name in index:
        for parent in PurePosixPath(name).parents:
            held = index.get(str(parent))
            if held is not None and not held.isdir():
                raise Unsafe(f"{name!r} passes through a file or link")
    resolved: dict[str, tarfile.TarInfo] = {}
    pairs = [(item, _regular(item, index, resolved)) for item in listed]
    if sum(source.size for item, source in pairs if not item.isdir()) > MOST_UNPACKED:
        raise Unsafe("materialized libraries exceed the unpacked size limit")
    return pairs


def _regular(item: tarfile.TarInfo, index: dict[str, tarfile.TarInfo],
             resolved: dict[str, tarfile.TarInfo]) -> tarfile.TarInfo:
    seen: set[str] = set()
    while item.issym():
        if item.name in resolved:
            item = resolved[item.name]
            break
        if item.name in seen:
            raise Unsafe(f"{item.name!r} is a cyclic library link")
        seen.add(item.name)
        if not LIBRARY.fullmatch(PurePosixPath(item.name).name):
            raise Unsafe(f"{item.name!r} is not a shared-library alias")
        destination = item.linkname
        if destination.startswith(("/", "\\")) or PureWindowsPath(destination).drive:
            raise Unsafe(f"{destination!r} is an absolute link")
        reached = posixpath.normpath(str(PurePosixPath(item.name).parent / destination))
        safe_join(Path("/archive"), reached)
        if reached not in index:
            raise Unsafe(f"{item.name!r} is a dangling library link")
        item = index[reached]
        if not LIBRARY.fullmatch(PurePosixPath(item.name).name) or not (item.isfile() or item.issym()):
            raise Unsafe(f"{item.name!r} is not a regular shared library")
    for name in seen:
        resolved[name] = item
    return item


def unpack(archive: Path, into: Path) -> None:
    """Extract files and materialize validated library aliases as regular files."""
    with tarfile.open(archive) as tf:
        planned = members(tf)
        for item, source in planned:
            target = safe_join(into, item.name.rstrip("/"))
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tf.extractfile(source)
            if stream is None:
                raise Unsafe(f"{source.name!r} cannot be read")
            with stream, target.open("wb") as out:
                while block := stream.read(1 << 20):
                    out.write(block)
            if source.mode & 0o111:
                target.chmod(0o755)
