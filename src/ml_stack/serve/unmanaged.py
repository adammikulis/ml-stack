"""Inference servers running on this machine that the lease registry does not hold.

They cannot be prevented, so they are found, counted against the memory budget and shown as
unmanaged. They are never leased from. ``ML_STACK_ADOPT_UNMANAGED`` (or the ``adopt_unmanaged``
limit) set to ``auto`` or ``ask`` lets the broker take one into the registry when the process
listening on its port passes `examine`: a loopback listener, owned by this user, running a
recognised server binary, whose ``/props`` and ``/health`` look like llama-server's. An adopted
server is queued and counted like any other and is never stopped by ml-stack.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack import limits
from ml_stack.client.health import is_healthy
from ml_stack.http import ServerError, request_json
from ml_stack.serve import admission
from ml_stack.serve.ports import DEFAULT_HOST, SERVER_BINARIES
from ml_stack.serve.process import every_server, listener

__all__ = ["ENV", "MODES", "Examined", "adopt_entry", "examine", "listener", "mode",
           "unmanaged_servers"]

ENV = "ML_STACK_ADOPT_UNMANAGED"
MODES = ("off", "ask", "auto")
LOOPBACK = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def mode() -> str:
    """``off``, ``ask`` or ``auto``: the environment first, then the limits file."""
    for named in (os.environ.get(ENV, ""), limits.read().adopt_unmanaged):
        if str(named).strip().lower() in MODES:
            return str(named).strip().lower()
    return "off"


def unmanaged_servers(records: Mapping[int, Mapping[str, Any]]) -> list[dict]:
    """Every llama-server process whose port the lease registry has no record of."""
    return [one for one in every_server() if not one.get("defunct")
            and int(one["port"]) not in records]


@dataclass(frozen=True)
class Examined:
    """Whether the listener on a port may be adopted, and what was found out about it."""

    ok: bool
    why: str = ""
    pid: int = 0
    model: str = ""
    rss: int = 0
    device: str = "gpu"


def _mine(found: Mapping[str, Any]) -> bool:
    if found.get("uid") is not None and hasattr(os, "getuid"):
        return int(found["uid"]) == os.getuid()
    import getpass

    return str(found.get("user") or "").rsplit("\\", 1)[-1] == getpass.getuser()


def examine(port: int, *, find: Callable[[int], dict[str, Any] | None] | None = None) -> Examined:
    """Whether the listener on ``port`` is a llama-server of this user's on loopback. The
    process is identified before anything is sent to it, and what is sent is two GETs
    with no credentials."""
    found = (find or listener)(port)
    if found is None:
        return Examined(False, f"no process this user can inspect listens on port {port}")
    pid = int(found.get("pid") or 0)
    if str(found.get("ip")) not in LOOPBACK:
        return Examined(False, f"pid {pid} listens on {found.get('ip')}, not loopback", pid)
    if not _mine(found):
        return Examined(False, f"pid {pid} is owned by {found.get('user') or 'another user'}",
                        pid)
    exe = Path(str(found.get("exe") or "")).name
    if exe not in SERVER_BINARIES:
        return Examined(False, f"pid {pid} runs {exe or 'an unknown program'}, which is not "
                               f"a recognised server binary", pid)
    base_url = f"http://{DEFAULT_HOST}:{port}"
    if not is_healthy(base_url, timeout=2.0):
        return Examined(False, f"pid {pid} does not answer /health", pid)
    try:
        props = request_json(f"{base_url}/props", timeout=5.0)
    except ServerError as exc:
        return Examined(False, f"pid {pid} does not answer /props: {exc}", pid)
    settings = props.get("default_generation_settings") if isinstance(props, dict) else None
    if not (isinstance(settings, dict) and isinstance(props.get("total_slots"), int)):
        return Examined(False, f"pid {pid}: /props is not a llama-server's", pid)
    return Examined(True, "", pid, str(props.get("model_path") or settings.get("model") or ""),
                    int(found.get("rss") or 0), admission.device_of(found))


def adopt_entry(port: int, seen: Examined) -> dict[str, Any]:
    """The registry entry for an adopted server: held by its own pid, so it is dropped when
    the process ends, and marked ``unmanaged`` so nothing ml-stack does stops it."""
    return {"port": port, "pid": seen.pid, "backend": "llama.cpp", "model": seen.model,
            "owner_pid": seen.pid, "base_url": f"http://{DEFAULT_HOST}:{port}",
            "device": seen.device, "est_bytes": seen.rss, "unmanaged": True}
