"""The sending side of test shards: the pool's devices, and a tree and test files sent to one through the node.

Everything goes through the local node (`shard_call`): it connects to the device over TLS pinned to
the certificate the pool record holds, and the device decides from that certificate who asked. The
registered session behind the token is only the label that appears in the audit trail.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import node_health, node_launch
from ml_stack.board import session as board_session
from ml_stack.board.client import NodeError
from ml_stack.fleet import shard_tree

CHUNK = 256 << 10
POLL_S = 2.0


class ShardError(RuntimeError):
    """A shard that could not be sent or did not finish; the message says why."""


def pool_devices(state: Path | None = None) -> list[dict]:
    """The other active devices of this node's pool: name, fingerprint, address, whether connected."""
    try:
        pool = node_health.call(state or node_launch.default_state(), "pool_status")
    except (OSError, ValueError) as exc:
        raise ShardError(f"no node answers on this device ({exc}); start it with `python -m ml_stack.device_setup --yes`") from None
    return [{"name": m.get("name") or m["fingerprint"][:12], "fingerprint": m["fingerprint"], "addr": m.get("addr") or "",
             "connected": bool(m.get("connected"))}
            for m in pool.get("members", []) if m.get("status") == "active" and not m.get("self")]


def choose(spec: str, devices: list[dict]) -> list[dict]:
    """The devices ``spec`` names: ``all``, a name, or a fingerprint; ShardError when none or two match."""
    if spec == "all":
        if not devices:
            raise ShardError("this pool has no other active device")
        return devices
    found = [d for d in devices if spec in (d["name"], d["fingerprint"])]
    if len(found) != 1:
        names = ", ".join(d["name"] for d in devices) or "none"
        raise ShardError(f"{'no' if not found else 'two'} active pool device is called {spec!r} (devices: {names})")
    return found


class Shards:
    """One registered session's calls to the shard ops of the devices of its pool."""

    def __init__(self, session: board_session.Session | None = None) -> None:
        self.session = session or board_session.connect()

    def call(self, device: str, op: str, **args: Any) -> dict:
        """One shard op on ``device`` (its fingerprint); the device's answer."""
        try:
            return self.session.client.call("shard_call", "", self.session.token, device=device, op=op, args=args)
        except NodeError as exc:
            raise ShardError(str(exc)) from None

    def capability(self, device: str) -> dict:
        """What ``device`` says about shards: ``accepts`` (and ``reason`` when not), platform, python, free slots."""
        return self.call(device, "shard_caps")

    def send(self, device: str, root: Path, tier: str, files: list[str], timeout_s: int) -> str:
        """Pack the tree at ``root``, upload it to ``device`` in order and start the run; the shard id."""
        try:
            tree, digest = shard_tree.pack(root)
        except (shard_tree.TreeError, OSError) as exc:
            raise ShardError(str(exc)) from None
        shard_id = secrets.token_hex(16)
        for offset in range(0, len(tree), CHUNK):
            self.call(device, "shard_put", id=shard_id, offset=offset, data=tree[offset:offset + CHUNK].hex())
        self.call(device, "shard_start", id=shard_id, tree_sha256=digest, size=len(tree), tier=tier, files=files,
                  timeout_s=timeout_s)
        return shard_id

    def wait(self, device: str, shard_id: str, *, timeout_s: float, poll_s: float = POLL_S,
             tick: Callable[[], None] = lambda: None) -> dict:
        """Poll until the shard has ended; its result, or ShardError when it failed or was cancelled."""
        deadline = time.monotonic() + timeout_s
        while True:
            got = self.call(device, "shard_status", id=shard_id)
            if got["state"] == "done":
                return got["result"]
            if got["state"] in ("failed", "cancelled"):
                raise ShardError(f"shard {shard_id} {got['state']}: {got.get('error', '')}")
            if time.monotonic() > deadline:
                self.cancel(device, shard_id)
                raise ShardError(f"shard {shard_id} still running after {timeout_s:.0f}s; cancelled")
            tick()
            time.sleep(poll_s)

    def cancel(self, device: str, shard_id: str) -> dict:
        """Stop a shard on ``device`` and everything it started."""
        return self.call(device, "shard_cancel", id=shard_id)
