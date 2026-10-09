"""Saving and restoring a running llama-server's slot caches, with a guard file beside each dump.

A dump is only valid for the model, per-slot context, slot count and llama-server build that
wrote it. `save_slot` writes ``<dump>.guard.json`` recording those four; `restore_slot`
reads it back and refuses when the server in front of it differs or the guard is missing.
The dump itself is written by llama-server into its ``--slot-save-path``, which is the
``directory`` these functions put the guard in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from poolhouse.http import ServerError, request_json
from poolhouse.serve.backend import default_slot_save_path

__all__ = ["GUARD_FIELDS", "SlotDump", "SlotGuardRefused", "current_guard", "dump_name",
           "guard_path", "restore_all", "restore_slot", "save_all", "save_slot"]

GUARD_FIELDS = ("model", "model_bytes", "ctx_size", "parallel", "llama_server_version")


class SlotGuardRefused(RuntimeError):
    """A dump was not restored because its guard is missing or the server differs from it."""


@dataclass(frozen=True, slots=True)
class SlotDump:
    """One slot's cache as saved or restored: the slot, the file, and the tokens it held."""

    slot: int
    filename: str
    tokens: int


def guard_path(directory: str | Path, filename: str) -> Path:
    """The guard file that sits beside the dump ``filename``."""
    return Path(directory) / f"{filename}.guard.json"


def dump_name(model: str, slot: int, name: str = "slot") -> str:
    """The dump filename for ``name`` on ``slot`` of ``model``; ``slot<N>`` is the unnamed one."""
    stem = PurePosixPath(model.replace("\\", "/")).stem or "model"
    return f"{stem}.slot{slot}.bin" if name == f"slot{slot}" else f"{stem}.slot{slot}.{name}.bin"


def current_guard(base_url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """The model, per-slot context, slot count and build the server at ``base_url`` reports."""
    props = request_json(f"{base_url.rstrip('/')}/props", method="GET", timeout=timeout)
    if not isinstance(props, dict):
        raise ServerError(f"{base_url}/props answered with {type(props).__name__}, not an object")
    settings = props.get("default_generation_settings") or {}
    path = str(props.get("model_path") or settings.get("model") or "")
    try:
        size: int | None = Path(path).stat().st_size
    except OSError:
        size = None
    return {
        "model": PurePosixPath(path.replace("\\", "/")).name,
        "model_bytes": size,
        "ctx_size": settings.get("n_ctx"),
        "parallel": props.get("total_slots"),
        "llama_server_version": props.get("build_info"),
    }


def save_slot(base_url: str, slot: int, filename: str, *,
              directory: str | Path | None = None, timeout: float = 120.0) -> SlotDump:
    """Save ``slot``'s cache as ``filename`` and write the guard file beside it."""
    root = Path(directory) if directory else default_slot_save_path()
    guard = {**current_guard(base_url), "slot_id": slot, "filename": filename}
    reply = request_json(f"{base_url.rstrip('/')}/slots/{slot}?action=save",
                         payload={"filename": filename}, timeout=timeout)
    root.mkdir(parents=True, exist_ok=True)
    guard_path(root, filename).write_text(json.dumps(guard, indent=2), encoding="utf-8")
    return SlotDump(slot, filename, _tokens(reply, "n_saved", "n_written_tokens"))


def restore_slot(base_url: str, slot: int, filename: str, *,
                 directory: str | Path | None = None, timeout: float = 120.0) -> SlotDump:
    """Restore ``filename`` into ``slot``, after checking its guard against the server."""
    root = Path(directory) if directory else default_slot_save_path()
    where = guard_path(root, filename)
    if not where.is_file():
        raise SlotGuardRefused(
            f"{filename} has no guard file at {where}; a cache that cannot be matched to "
            f"the server that wrote it is not restored")
    saved = json.loads(where.read_text(encoding="utf-8"))
    now = current_guard(base_url)
    differ = [f"{key}: saved {saved.get(key)!r}, server {now.get(key)!r}"
              for key in GUARD_FIELDS if saved.get(key) != now.get(key)]
    if differ:
        raise SlotGuardRefused(
            f"{filename} was saved by a different server layout ({'; '.join(differ)})")
    reply = request_json(f"{base_url.rstrip('/')}/slots/{slot}?action=restore",
                         payload={"filename": filename}, timeout=timeout)
    return SlotDump(slot, filename, _tokens(reply, "n_restored", "n_read_tokens"))


def save_all(base_url: str, slots: Mapping[str, int], *, model: str = "",
             directory: str | Path | None = None, timeout: float = 120.0) -> list[SlotDump]:
    """Save every slot in ``slots`` (``{name: slot id}``) under its dump name."""
    label = model or current_guard(base_url)["model"]
    return [save_slot(base_url, sid, dump_name(label, sid, name),
                      directory=directory, timeout=timeout) for name, sid in slots.items()]


def restore_all(base_url: str, slots: Mapping[str, int], *, model: str = "",
                directory: str | Path | None = None,
                timeout: float = 120.0) -> list[SlotDump]:
    """Restore every slot in ``slots`` that has a dump; a slot with none is passed over."""
    root = Path(directory) if directory else default_slot_save_path()
    label = model or current_guard(base_url)["model"]
    restored = []
    for name, sid in slots.items():
        filename = dump_name(label, sid, name)
        if not (root / filename).exists() and not guard_path(root, filename).exists():
            continue
        restored.append(restore_slot(base_url, sid, filename, directory=root, timeout=timeout))
    return restored


def _tokens(reply: Any, *keys: str) -> int:
    if not isinstance(reply, dict):
        return 0
    for key in keys:
        if isinstance(reply.get(key), int):
            return int(reply[key])
    return 0
