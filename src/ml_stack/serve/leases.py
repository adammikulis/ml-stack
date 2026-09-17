"""The file recording which model servers this machine is running, and reading it back."""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.serve.process import pid_exists

__all__ = ["already_up", "lease_file", "merge_state", "orphaned", "reap_one",
           "recorded_servers"]


def lease_file() -> Path:
    """The file recording which model servers this machine is running."""
    return home.moved("servers.json")


def merge_state(on_disk: dict, mine: dict, owner_pid: int) -> dict:
    """This process's servers, merged over every other record whose owner or server is alive."""
    merged = {
        key: entry
        for key, entry in on_disk.items()
        if isinstance(entry, dict)
        and entry.get("owner_pid") != owner_pid
        and (pid_exists(entry.get("owner_pid")) or pid_exists(entry.get("pid")))
    }
    merged.update(mine)
    return merged


def reap_one(held: Any, *, grace_s: float) -> None:
    """Wait on a child this process started, so its pid leaves the table."""
    if held is None:
        return
    with contextlib.suppress(subprocess.TimeoutExpired, ChildProcessError, OSError):
        held.wait(timeout=grace_s)


def orphaned(entry: dict) -> bool:
    """Whether a record's server is running on after the process that leased it has gone."""
    owner, pid = entry.get("owner_pid"), entry.get("pid")
    return (isinstance(owner, int) and isinstance(pid, int) and owner != pid
            and not pid_exists(owner) and pid_exists(pid))


def already_up(model: str, port: int, *, state_file: Path | None = None) -> dict | None:
    """The recorded server on ``port`` if it serves ``model`` (by file name) and its
    process is alive -- whatever its settings. A conversation that would lease one slot on a
    port already holding the same weights with other settings uses what is up rather than
    reloading them: the reload is the cost, the settings are not (Adam, 2026-09-03)."""
    entry = recorded_servers(state_file).get(int(port))
    if not entry:
        return None
    if Path(str(entry.get("model") or "")).name != Path(str(model)).name:
        return None
    return entry if pid_exists(int(entry.get("pid") or 0)) else None


def recorded_servers(state_file: Path | None = None) -> dict[int, dict]:
    """Every server in the lease file, keyed by port."""
    try:
        parsed = json.loads((state_file or lease_file()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}

    out: dict[int, dict] = {}
    for key, entry in parsed.items():
        if not isinstance(entry, dict):
            continue
        try:
            out[int(entry.get("port", key))] = entry
        except (TypeError, ValueError):
            continue
    return out
