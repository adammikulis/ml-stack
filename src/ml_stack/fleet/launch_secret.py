"""The launch secret: the value this daemon's own window and the owner's terminal present to
ask for a one-shot sign-in ticket."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from ml_stack.files import writing
from ml_stack.private_path import problem
from ml_stack.sentinel import human
from ml_stack.windows_private import restrict

from .session import same

__all__ = ["DIRECTORY", "FILE", "HEADER", "LaunchError", "LaunchSecret", "read"]

DIRECTORY = "launch"
FILE = "secret.json"
HEADER = "X-ML-Stack-Launch"
VERSION = 1
BYTES = 32


class LaunchError(RuntimeError):
    """The launch secret cannot be written or trusted."""


def _private_directory(root: Path) -> Path:
    path = root / DIRECTORY
    human.protect(path)
    why = problem(path)
    if why not in {"", "missing"} and not why.startswith("mode "):
        raise LaunchError(f"{path}: {why}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        restrict(path)
    else:
        path.chmod(0o700)
    return path


class LaunchSecret:
    """A random value generated when the daemon starts, kept in memory and in an owner-only file."""

    def __init__(self, root: Path | str, port: int) -> None:
        self.value = secrets.token_urlsafe(BYTES)
        self.path = _private_directory(Path(root)) / FILE
        with writing(self.path) as tmp:
            if os.name == "nt":
                restrict(tmp)
            else:
                tmp.chmod(0o600)
            tmp.write_text(json.dumps({"version": VERSION, "secret": self.value, "port": port}),
                           encoding="utf-8")

    def matches(self, presented: str) -> bool:
        """Whether ``presented`` is this daemon's secret."""
        return bool(presented) and same(presented, self.value)


def read(root: Path | str) -> dict[str, object]:
    """The record the daemon under ``root`` wrote; `LaunchError` when it is missing or unsafe."""
    directory = Path(root) / DIRECTORY
    path = directory / FILE
    for target in (directory, path):
        why = problem(target)
        if why:
            raise LaunchError(f"no running ml-stack daemon is recorded here ({target}: {why})")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LaunchError(f"{path} cannot be read: {error}") from error
    if (not isinstance(record, dict) or record.get("version") != VERSION
            or not isinstance(record.get("secret"), str) or not isinstance(record.get("port"), int)):
        raise LaunchError(f"{path} is not a launch record")
    return record
