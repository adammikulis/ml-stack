"""The receiving side of test shards: consent, who may send, and the shards in flight."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path

from . import shard_result, shard_run, shard_spec, shard_tree

MOST_ACTIVE = 2
MOST_KEPT = 50
ID = re.compile(r"^[0-9a-f]{32}$")


class Declined(Exception):
    """A request the host will not take; ``status`` and the message go back to the sender."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ShardHost:
    """Takes `test-shard` requests from paired devices its person has allowed, and runs them."""

    def __init__(self, folder: Path, enabled: Callable[[], bool],
                 command: Callable[[Path, list[str], Path], list[str]] = shard_run.shard_command) -> None:
        self.folder, self.enabled, self.command = Path(folder), enabled, command
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()
        self._running: dict[str, threading.Thread] = {}

    def capability(self) -> dict:
        """What this device tells a paired sender: whether it takes shards, and where it runs them."""
        return {"accepts": bool(self.enabled()), "platform": shard_result.stamp(),
                "active": len(self._running), "most_active": MOST_ACTIVE}

    def describe(self, device: object) -> dict:
        """`capability` for any paired device, whether or not shards are on."""
        if device is None:
            raise Declined(403, "test shards come from a paired device only")
        return self.capability()

    def allow(self, device: object) -> None:
        """Decline unless a shard is switched on here and ``device`` is a paired device of this owner."""
        if not self.enabled():
            raise Declined(403, "test shards are off on this device; its person turns them on")
        if device is None:
            raise Declined(403, "test shards come from a paired device only")
        if not getattr(device, "mine", False):
            raise Declined(403, "test shards come from a device its owner marked as their own")

    def submit(self, device: object, body: bytes) -> dict:
        """Check and start the shard in ``body``; its public record, or `Declined`."""
        self.allow(device)
        try:
            header, data = shard_spec.split(body)
            shard = shard_spec.check_header(header)
            shard_spec.check_present(shard["files"], [m.name for m in
                                                      shard_tree.members(data, shard["tree_sha256"])])
        except (shard_spec.Refused, shard_tree.TreeError) as exc:
            raise Declined(400, str(exc)) from None
        with self._lock:
            self._claim(shard["id"])
            thread = threading.Thread(target=self._finish, args=(shard, data), daemon=True)
            self._running[shard["id"]] = thread
        thread.start()
        return {"id": shard["id"], "state": "running"}

    def _claim(self, shard_id: str) -> None:
        if len(self._running) >= MOST_ACTIVE:
            raise Declined(429, f"{MOST_ACTIVE} shards are already running here")
        try:
            (self.folder / f"{shard_id}.claim").touch(exist_ok=False, mode=0o600)
        except FileExistsError:
            raise Declined(409, "that shard id was already used") from None

    def _finish(self, shard: dict, data: bytes) -> None:
        try:
            shard_run.run_shard(shard, data, self.folder, self.command)
        finally:
            with self._lock:
                self._running.pop(shard["id"], None)
            self._prune()

    def _prune(self) -> None:
        done = sorted(self.folder.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for path in done[:-MOST_KEPT]:
            path.unlink(missing_ok=True)
            path.with_suffix(".claim").unlink(missing_ok=True)

    def state(self, device: object, shard_id: str) -> dict:
        """The finished result of a shard, or ``{"state": "running"}``; `Declined` for an unknown one."""
        self.allow(device)
        if not ID.match(shard_id) or not (self.folder / f"{shard_id}.claim").exists():
            raise Declined(404, "no such shard")
        path = self.folder / f"{shard_id}.json"
        return json.loads(path.read_text()) if path.is_file() else {"id": shard_id, "state": "running"}
