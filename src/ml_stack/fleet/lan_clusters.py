"""Named LAN hints from the existing salt handshake."""

from __future__ import annotations

import base64
import json
import secrets
import time
from typing import Any

from . import discovery


def _hint(raw: bytes, nonce: str) -> str | None:
    if len(raw) > 2048:
        return None
    try:
        msg = json.loads(raw)
        if not isinstance(msg, dict) or msg.get("v") != discovery.PROTOCOL:
            return None
        if msg.get("kind") != "salt" or msg.get("nonce") != nonce:
            return None
        salt = base64.urlsafe_b64decode(msg["salt"] + "==")
        if not 16 <= len(salt) <= 64:
            return None
        return discovery.require_name(msg.get("group"))
    except (ValueError, TypeError, KeyError, discovery.DiscoveryError):
        return None


def nearby(*, timeout_s: float = 1.5, port: int | None = None) -> list[dict[str, Any]]:
    """Unverified cluster-name hints; joining separately authenticates the response."""
    nonce = secrets.token_hex(16)
    hello = json.dumps({"v": discovery.PROTOCOL, "kind": "hello", "nonce": nonce}).encode()
    groups: set[str] = set()
    sources: dict[str, int] = {}
    target_port = port if port is not None else discovery.default_port()
    with discovery._socket(broadcast=True, bind=("", 0)) as sock:
        discovery._say(sock, hello, discovery.default_group(), target_port)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and len(groups) < 32:
            sock.settimeout(max(0.01, deadline - time.monotonic()))
            try:
                raw, addr = sock.recvfrom(2049)
            except TimeoutError:
                break
            if addr[0] not in sources and len(sources) >= 64:
                continue
            sources[addr[0]] = sources.get(addr[0], 0) + 1
            if sources[addr[0]] > 8 or len(sources) > 64:
                continue
            group = _hint(raw, nonce)
            if group:
                groups.add(group)
    return [{"group": name} for name in sorted(groups, key=str.casefold)]


def verified_salt(passphrase: str, group: str) -> discovery.Salting:
    """Authenticate an existing cluster before writing a membership."""
    found = discovery.find_salt(passphrase, group=group)
    if found is None:
        raise discovery.DiscoveryError("cluster is no longer available; refresh nearby clusters")
    return discovery.Salting(salt=found[0])
