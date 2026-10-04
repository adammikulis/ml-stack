"""The API key each leased model server was started with, and the lookup that finds it by URL."""

from __future__ import annotations

import os
import secrets
import time
import urllib.parse
from pathlib import Path

from ml_stack import home
from ml_stack.files import read_json, write_json

__all__ = ["bind", "for_url", "issue"]

LOOPBACK = ("127.0.0.1", "localhost", "::1")
STARTING_S = 600


def _file() -> Path:
    return home.state("server-keys.json")


def _entries() -> dict[str, dict]:
    found = read_json(_file(), {})
    return found if isinstance(found, dict) else {}


def _alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _live(entry: object) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
        return False
    if entry.get("pid") == 0:
        return time.time() - float(entry.get("at", 0)) < STARTING_S
    return _alive(entry.get("pid"))


def _save(entries: dict[str, dict]) -> None:
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, entries)
    path.chmod(0o600)


def issue(port: int) -> str:
    """A new key for the server about to start on ``port``, recorded for `for_url`."""
    key = secrets.token_urlsafe(32)
    entries = {p: e for p, e in _entries().items() if _live(e)}
    entries[str(int(port))] = {"key": key, "pid": 0, "at": time.time()}
    _save(entries)
    return key


def bind(port: int, pid: int) -> None:
    """Tie the key issued for ``port`` to the server process ``pid``."""
    entries = _entries()
    if str(int(port)) in entries:
        entries[str(int(port))]["pid"] = int(pid)
        _save(entries)


def for_url(url: str) -> str:
    """The key of the leased server at a loopback ``url``, or ``""`` when none is running there."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname not in LOOPBACK:
        return ""
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return ""
    entry = _entries().get(str(port))
    return entry["key"] if isinstance(entry, dict) and _live(entry) else ""
