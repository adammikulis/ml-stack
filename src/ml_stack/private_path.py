"""Whether a path may hold a private credential: a plain file or directory this user alone can read."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from ml_stack.windows_private import problem as windows_problem

__all__ = ["problem", "redirected", "windows_mount"]


def redirected(info: os.stat_result) -> bool:
    """Whether ``info`` describes a symlink or a Windows reparse point."""
    return (stat.S_ISLNK(info.st_mode) or (os.name == "nt"
            and bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)))


def problem(path: Path) -> str:
    """Why ``path`` may not hold a credential (or be the directory of them), or an empty string."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "missing"
    if redirected(info):
        return "is a symlink or Windows reparse point"
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        return "is not a plain file"
    if windows_mount(path):
        return ("is on a Windows-mounted filesystem; keep workspace tokens under the WSL "
                "home directory")
    if os.name == "nt":
        return windows_problem(path)
    if info.st_uid != os.getuid():
        return "belongs to another user"
    if info.st_mode & 0o077:
        return f"mode {info.st_mode & 0o777:o} lets others read it; chmod {'700' if stat.S_ISDIR(info.st_mode) else '600'}"
    return ""


def windows_mount(path: Path) -> bool:
    """Whether a WSL path is backed by a Windows filesystem."""
    if sys.platform == "win32":
        return False
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    if "microsoft" not in release:
        return False
    target = path.resolve()
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    best = (0, False)
    for line in mounts:
        fields = line.split()
        if len(fields) < 3 or fields[2] not in ("9p", "drvfs"):
            continue
        mountpoint = Path(fields[1].replace("\\040", " "))
        try:
            target.relative_to(mountpoint)
        except ValueError:
            continue
        if len(mountpoint.parts) > best[0]:
            best = (len(mountpoint.parts), True)
    return best[1]
