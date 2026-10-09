"""The sending side of test shards: find a paired device, send it a tree and test files, wait for the result."""

from __future__ import annotations

import base64
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from ml_stack import home, http, macauth
from ml_stack.fleet import tls
from ml_stack.fleet.onboard.requests import Devices
from ml_stack.hub.peerbook import PeerBook

from . import shard_spec, shard_tree
from .remote import Peer, PeerError

ROUTE = "/workspace/v1/test-shards"
DAEMON_PORT = 8770
POLL_S = 2.0


class NoDevice(LookupError):
    """No unique, active, paired device by that name."""


def paired_row(name: str, book: PeerBook, devices: Devices) -> dict:
    """The one peer-book row of the active paired device called ``name``."""
    active = {device.fingerprint for device in devices.all() if device.status == "active"}
    rows = [row for row in book.rows() if row.get("name") == name and row.get("source") == "pairing"
            and row.get("fingerprint") in active and row.get("certificate") and row.get("device_secret")]
    if len(rows) != 1:
        raise NoDevice(f"no unique active paired device called {name!r}")
    return rows[0]


def device_peer(name: str, *, port: int = DAEMON_PORT, host: str = "") -> Peer:
    """A pinned, signed connection to the paired device ``name``'s daemon."""
    directory = home.state("onboard")
    row = paired_row(name, PeerBook(directory / "peers.json"), Devices(directory / "devices.json"))
    where = host or urlsplit(row["url"]).hostname or ""
    if not where:
        raise NoDevice(f"the peer book has no address for {name!r}")
    http.pin(f"{where}:{port}", tls.pinned_context(row["certificate"]))
    secret = macauth.derive(base64.urlsafe_b64decode(row["device_secret"]))
    return Peer(f"https://{where}:{port}", secret, timeout=300)


def capability(peer: Peer) -> dict:
    """What the device says about test shards: ``accepts``, its platform and how many it is running."""
    return peer._json("GET", ROUTE)


def send(peer: Peer, root: Path, files: list[str], timeout_s: int) -> str:
    """Pack the tree at ``root`` and send it with ``files``; the shard id."""
    tree, digest = shard_tree.pack(root)
    shard_id = secrets.token_hex(16)
    header = {"id": shard_id, "tree_sha256": digest, "files": files, "timeout_s": timeout_s}
    peer._request("POST", ROUTE, data=shard_spec.frame(header, tree),
                  headers={"Content-Type": "application/octet-stream"})
    return shard_id


def wait(peer: Peer, shard_id: str, *, timeout_s: float, poll_s: float = POLL_S,
         tick: Callable[[], None] = lambda: None) -> dict:
    """Poll until the shard is finished; its result record."""
    deadline = time.monotonic() + timeout_s
    while True:
        got = peer._json("GET", f"{ROUTE}/{shard_id}")
        if got.get("state") != "running":
            return got
        if time.monotonic() > deadline:
            raise PeerError(f"shard {shard_id} still running after {timeout_s:.0f}s")
        tick()
        time.sleep(poll_s)
