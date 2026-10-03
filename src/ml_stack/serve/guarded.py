"""What the Broker asks sentinel before a model server starts: a quarantined model or server is
refused, a pinned model must still hash to its pin, and an unpinned model is pinned on first use."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import sentinel
from ml_stack.sentinel import Sentinel
from ml_stack.sentinel.adapters import serve_hooks
from ml_stack.sentinel.events import Event, Severity
from ml_stack.sentinel.watch import Scanner
from ml_stack.serve.backend import ServerFailed
from ml_stack.serve.leases import recorded_servers

__all__ = ["SentinelRefused", "arm", "blocked", "register", "verify"]


class SentinelRefused(ServerFailed):
    """Sentinel holds the model, or the model no longer matches its pin."""


_HOOKED: weakref.WeakKeyDictionary[Sentinel, set[str]] = weakref.WeakKeyDictionary()


def _file(model: object) -> Path | None:
    """The model file ``model`` names, or None when it is not a file on this machine."""
    try:
        path = Path(str(model)).expanduser()
        return path if path.is_file() else None
    except (OSError, ValueError):
        return None


def blocked(model: object) -> str:
    """Why sentinel keeps ``model`` from being served, or an empty string. Reads the store
    only; nothing is hashed."""
    node = sentinel.default()
    if node.mode == sentinel.Mode.OFF:
        return ""
    path = Path(str(model)).expanduser()
    if node.store.blocked("model", str(path)):
        return (f"{path.name} is quarantined by sentinel; a person releases it with "
                f"`ml-stack-security release`")
    return ""


def register(node: Sentinel, state_file: Path, stop: Callable[[int], object]) -> None:
    """Make a quarantined model or server stop the servers ml-stack started for it. Once per
    sentinel and lease file."""
    done = _HOOKED.setdefault(node, set())
    if str(state_file) in done:
        return
    done.add(str(state_file))
    serve_hooks(node, lambda: recorded_servers(state_file), stop)


def arm(manager: Any) -> Scanner | None:
    """What a process that runs the Broker arms: the decoys, the stop hooks for ``manager``'s
    servers and the periodic scan. Returns the scan loop, which the caller stops."""
    node = sentinel.armed()
    register(node, manager.state_file, manager.reclaim)
    return sentinel.arm_scan(node)


def verify(model: object, *, state_file: Path, stop: Callable[[int], Any]) -> None:
    """Refuse to start a server for ``model`` when sentinel holds it or its bytes differ from
    the pin. A model with no pin is pinned now, with source ``first-use``."""
    node = sentinel.armed()
    register(node, state_file, stop)
    if node.mode == sentinel.Mode.OFF:
        return
    if why := blocked(model):
        raise SentinelRefused(why)
    path = _file(model)
    if path is None:
        return
    if str(path) not in node.manifest.pins():
        node.manifest.pin(path, "model", source="first-use")
        node.bus.emit(Event("model.pinned_first_use", Severity.NOTICE, "serve", f"model:{path}",
                            {"name": path.name}, node.clock()))
        return
    if not node.verify_before_load(path, cached=True):
        raise SentinelRefused(f"{path.name} does not match its pin and is quarantined by "
                              f"sentinel; the file was moved aside")
