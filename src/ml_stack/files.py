"""JSON files that another process may be reading while they are rewritten.

A pipeline writes its state as it goes and something else — a page being served, a second
command, a person with ``cat`` — reads it at any moment. A plain ``open(..., "w")`` shows
that reader an empty file and then half of one. Everything here writes beside the file and
renames over it, so a reader sees the old contents or the new, never the middle.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Set
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["UNVERSIONED", "CrossDevice", "promote", "prune_orphans", "read_json", "records_in",
           "sha256_file", "version_of", "versioned", "write_json", "write_text", "writing"]

#: A record with no version key.
UNVERSIONED = 0


class CrossDevice(OSError):
    """The source and the target are on different filesystems."""


def promote(source: Path | str, target: Path | str) -> Path:
    """Put a finished ``source`` in ``target``'s place in one step. Returns ``target``.

    ``source`` may be a file, a directory or a symlink, and must be on the same filesystem
    as ``target``; when it is not, `CrossDevice` says so rather than the bare ``EXDEV``.
    """
    source, target = Path(source), Path(target)
    try:
        source.replace(target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        raise CrossDevice(
            errno.EXDEV,
            f"{source} and {target} are on different filesystems, so moving one onto the "
            f"other would copy it") from None
    return target


@contextmanager
def writing(path: Path | str, *, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a temporary path beside ``path`` that takes ``path``'s place on a clean exit.

    A reader of ``path`` sees the old contents or the new ones. Leaving the block by an
    exception takes the temporary away and leaves ``path`` as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, made = tempfile.mkstemp(dir=path.parent, suffix=suffix)
    os.close(fd)
    tmp = Path(made)
    try:
        yield tmp
        promote(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` so a concurrent reader sees the old file or the new one."""
    with writing(path) as tmp:
        tmp.write_text(text, encoding=encoding)


def write_json(path: Path, obj: Any, *, indent: int | None = 2,
               default: Callable[[Any], Any] | None = None) -> None:
    """Write ``obj`` as JSON so a concurrent reader sees the old file or the new one.

    ``default`` is `json.dump`'s: what to call on a value JSON has no shape for. A failure
    part-way (a value that is not JSON, a full disk) leaves the old file as it was and no
    temporary behind.
    """
    with writing(path) as tmp, tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=default)


def versioned(record: Mapping[str, Any], version: int) -> dict[str, Any]:
    """``record`` carrying the version key that says which shape it has."""
    return {"version": int(version), **dict(record)}


def version_of(record: Any) -> int:
    """The version ``record`` declares, or ``UNVERSIONED`` when it carries no version key."""
    if not isinstance(record, Mapping):
        return UNVERSIONED
    try:
        return int(record.get("version", UNVERSIONED) or UNVERSIONED)
    except (TypeError, ValueError):
        return UNVERSIONED


def read_json(path: Path, default: Any) -> Any:
    """What ``path`` holds, or ``default`` when it is missing or is not JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def prune_orphans(directory: Path, live: Set[str], suffix: str = ".json") -> list[str]:
    """Delete every ``<id><suffix>`` in ``directory`` whose id is not in ``live``.

    For a directory of one file per record of a log — an extraction per message, a
    thumbnail per page — that would otherwise keep records the log has since dropped.
    Returns the ids that went, sorted. A directory that does not exist has no orphans.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    gone: list[str] = []
    for f in sorted(directory.glob(f"*{suffix}")):
        if f.stem not in live:
            f.unlink()
            gone.append(f.stem)
    return gone


def records_in(path: str | Path, key: str) -> list[dict[str, Any]]:
    """The list of mappings a JSON file holds under ``key``, or the file itself when it is
    a list. Raises ``ValueError`` naming the file when there are none."""
    held = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    found = held.get(key) if isinstance(held, Mapping) else held
    if not isinstance(found, list) or not found:
        raise ValueError(f"{path}: no {key}")
    return [dict(one) for one in found if isinstance(one, Mapping)]


def sha256_file(path: Path | str, *, chunk: int = 1 << 20) -> str:
    """The hex sha256 of a file, read ``chunk`` bytes at a time."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
