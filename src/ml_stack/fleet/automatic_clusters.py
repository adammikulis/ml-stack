"""Automatic Development cluster discovery, admission and convergence."""

import hashlib
import secrets
import threading
from collections.abc import Callable
from pathlib import Path

from ml_stack.fleet import cluster_modes, discovery
from ml_stack.fleet.onboard import joining


def offers(port: int | None = None) -> list[tuple[str, dict]]:
    """Return bounded TLS Development cluster offers from local discovery."""
    return [(host, row) for host, row in joining._ask_join("", timeout_s=1.5, port=port, most=64)
            if row.get("mode") == "dev" and row.get("method") == "automatic"
            and row.get("tls") is True and isinstance(row.get("fingerprint"), str)
            and len(row["fingerprint"]) == 64
            and all(c in "0123456789abcdef" for c in row["fingerprint"]) and isinstance(row.get("cluster_id"), str)
            and len(row["cluster_id"]) == 64
            and all(c in "0123456789abcdef" for c in row["cluster_id"])]


def receive(host: str, offer: dict) -> discovery.Membership:
    """Join one advertised Development cluster over its pinned TLS endpoint."""
    call = joining._Call(joining.Joiner(host, offer["port"], True), 1.0)
    call.seen = offer["fingerprint"]
    nonce = secrets.token_hex(16)
    status, result = call.post("automatic", {"group": offer["group"], "nonce": nonce})
    if status != 200:
        raise discovery.DiscoveryError(str(result.get("error", "automatic join refused")))
    key = result.get("key")
    if (result.get("mode") != "dev" or result.get("group") != offer["group"]
            or result.get("nonce") != nonce or not isinstance(key, str) or len(key) != 43
            or hashlib.sha256(key.encode()).hexdigest() != offer["cluster_id"]):
        raise discovery.DiscoveryError("automatic cluster response does not match its advertised identity")
    try:
        return discovery.Membership(result["group"], key.encode(), mode="dev")
    except ValueError as error:
        raise discovery.DiscoveryError("automatic cluster response contains an invalid key") from error


def ensure(path: Path | str | None = None, *, mode: str | None = None,
           port: int | None = None) -> discovery.Membership:
    """Retain Production membership or converge Development devices onto one cluster."""
    existing = discovery.memberships(path)
    selected = cluster_modes.validate(mode or (existing[0].mode if existing else "dev"))
    if selected == "prod":
        if not existing or existing[0].mode != "prod":
            raise discovery.DiscoveryError("Production mode needs an explicitly admitted Production cluster")
        return existing[0]
    if existing and existing[0].mode != "dev":
        raise discovery.DiscoveryError("select an explicitly admitted Development cluster before changing modes")
    try:
        candidates = offers(port)
    except (OSError, discovery.DiscoveryError):
        candidates = []
    current = existing[0] if existing else None
    own = hashlib.sha256(current.key).hexdigest() if current else ""
    candidates = sorted(candidates, key=lambda item: (item[1]["cluster_id"], item[0]))
    for host, offered in candidates[:8]:
        if current and own <= offered["cluster_id"]:
            break
        try:
            member = receive(host, offered)
        except (OSError, discovery.DiscoveryError):
            continue
        return discovery.adopt(member, path)
    return current or discovery.mint_cluster("development", path, mode="dev")


def converge(stop: threading.Event, changed: Callable[[], None],
             path: Path | str | None = None, *, interval_s: float = 15.0) -> None:
    """Refresh Development membership until the daemon stops."""
    while not stop.wait(interval_s):
        before = discovery.memberships(path)
        if not before or before[0].mode != "dev":
            continue
        try:
            ensure(path, mode="dev")
            changed()
        except (OSError, discovery.DiscoveryError):
            continue
