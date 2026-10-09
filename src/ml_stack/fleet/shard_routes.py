"""The daemon's `/workspace/v1/test-shards` routes: what a paired device can ask of this one."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .settings import Settings
from .shard_host import Declined, ShardHost

PATH = re.compile(r"/workspace/v1/test-shards(?:/([0-9a-f]{32}))?")


def consent(settings: Any, path: Path | None) -> bool:
    """Whether this device's person has turned test shards on (read from disk when a path is known)."""
    if path is not None and path.is_file():
        return bool(Settings.load(path).test_shards)
    return bool(getattr(settings, "test_shards", False))


def default_host(daemon: Any) -> ShardHost:
    """The host a daemon answers from when it was not handed one: shards follow the saved setting."""
    ui = daemon.ui
    path = getattr(ui, "settings_path", None)
    host = ShardHost(Path(daemon.files_root) / "test-shards",
                     lambda: consent(getattr(ui, "settings", None), path))
    daemon.shards = host
    return host


def answer(handler: Any, host: ShardHost, body: bytes | None) -> bool:
    """Answer a test-shard request; False when the path is not one."""
    match = PATH.fullmatch(urlparse(handler.path).path)
    if not match:
        return False
    opening = handler._sealing()
    if opening is None or not opening[2]:
        handler._send(403, {"error": "test shards are asked for with sealed fleet requests"})
        return True
    device = handler._workspace_device
    try:
        if handler.command == "GET":
            reply = host.state(device, match[1]) if match[1] else host.describe(device)
        elif handler.command == "POST" and not match[1]:
            handler._send(201, host.submit(device, body or b""))
            return True
        else:
            handler._send(405, {"error": "that method is not allowed here"})
            return True
    except Declined as refusal:
        handler._send(refusal.status, {"error": str(refusal)})
        return True
    handler._send(200, reply)
    return True


def serve(handler: Any, daemon: Any, body: bytes | None = None) -> bool:
    """Answer ``handler``'s request when it is a test-shard one, from the daemon's host; False otherwise."""
    if not PATH.fullmatch(urlparse(handler.path).path):
        return False
    return answer(handler, daemon.shards or default_host(daemon), body)
