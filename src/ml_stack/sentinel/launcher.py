"""The one-click way to the review screen: a file to double-click, and the window the
notification's Review button opens.

The file is a fixed template. It carries the absolute path of the installed
``ml-stack-security`` and nothing else: no held text, no secret, no environment from whoever
wrote it. It does not change directory and runs the command in a login shell so ``PATH`` is
the person's own. It is never overwritten without ``force``.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import stat
import sys
from pathlib import Path

from ml_stack import desktop
from ml_stack.sentinel.store import sentinel_dir

__all__ = ["LauncherError", "install_launcher", "installed_command", "open_review"]

MAC_NAME = "Review quarantine.command"
LINUX_NAME = "Review quarantine.desktop"
_UNSAFE_EXEC = set('"`$\\\n\r%')


class LauncherError(Exception):
    """The launcher could not be written, with the reason in plain words."""


def installed_command() -> str:
    """The absolute path of this install's ``ml-stack-security``."""
    beside = Path(sys.executable).parent / "ml-stack-security"
    found = str(beside) if beside.is_file() else shutil.which("ml-stack-security")
    if not found:
        raise LauncherError("ml-stack-security is not installed on PATH; install ml-stack first")
    return str(Path(found).absolute())


def _mac_text(exe: str) -> str:
    inner = shlex.quote(exe) + " review"
    return ("#!/bin/sh\n"
            "# Opens ml-stack's quarantine review. Written by "
            "`ml-stack-security review --install-launcher`.\n"
            f"exec \"${{SHELL:-/bin/sh}}\" -l -c {shlex.quote(inner)}\n")


def _linux_text(exe: str) -> str:
    if _UNSAFE_EXEC & set(exe):
        raise LauncherError("the program's path has a character a .desktop file cannot hold")
    return ("[Desktop Entry]\nType=Application\nName=Review quarantine\n"
            "Comment=Review what ml-stack's sentinel is holding\n"
            f"Exec=\"{exe}\" review\nTerminal=true\nCategories=System;\n")


def _write(path: Path, text: str, force: bool) -> None:
    if force and (path.is_symlink() or path.exists()):
        path.unlink()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    except FileExistsError:
        raise LauncherError(f"{path} already exists; pass --force to replace it") from None
    except OSError as exc:
        raise LauncherError(f"cannot write {path}: {exc.strerror}") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    path.chmod(stat.S_IRWXU)


def install_launcher(directory: Path, *, force: bool = False, system: str | None = None,
                     exe: str | None = None) -> Path:
    """Write the launcher into ``directory`` and return its path."""
    system = system or platform.system()
    if system not in ("Darwin", "Linux"):
        raise LauncherError("a launcher file is written on macOS and Linux only")
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise LauncherError(f"{directory} is not a folder")
    exe = exe or installed_command()
    mac = system == "Darwin"
    path = directory / (MAC_NAME if mac else LINUX_NAME)
    _write(path, _mac_text(exe) if mac else _linux_text(exe), force)
    return path


def open_review() -> bool:
    """Open a terminal window running the review screen. Nothing is passed to it."""
    try:
        way = desktop.which_way()
        if way == "macos":
            folder = sentinel_dir() / "launcher"
            folder.mkdir(parents=True, exist_ok=True)
            path = install_launcher(folder, force=True, system="Darwin")
            return desktop.open_terminal([str(path)], way=way)
        return desktop.open_terminal([installed_command(), "review"], way=way)
    except (LauncherError, OSError):
        return False
