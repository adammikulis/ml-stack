"""Named LAN hints from the maintained PAKE join discovery."""

from __future__ import annotations

import json
import secrets
import time
from typing import Any

from . import discovery


def _hint(raw: bytes, nonce: str) -> dict[str, str] | None:
    if len(raw) > 2048:
        return None
    try:
        msg = json.loads(raw)
        if not isinstance(msg, dict) or msg.get("v") != discovery.PROTOCOL:
            return None
        if msg.get("kind") != "join" or msg.get("nonce") != nonce:
            return None
        name = discovery.require_name(msg.get("group"))
        if msg.get("method") not in ("passphrase", "recovery"):
            return None
        port = msg.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            return None
        return {"group": name, **({"method": "recovery"} if msg["method"] == "recovery" else {})}
    except (ValueError, TypeError, KeyError, discovery.DiscoveryError):
        return None


def nearby(*, timeout_s: float = 1.5, port: int | None = None) -> list[dict[str, Any]]:
    """Unverified cluster-name hints; joining separately authenticates the response."""
    nonce = secrets.token_hex(16)
    hello = json.dumps({"v": discovery.PROTOCOL, "kind": "join?", "group": "", "nonce": nonce}).encode()
    groups: dict[tuple[str, str], dict[str, str]] = {}
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
                groups[(group["group"], group.get("method", "passphrase"))] = group
    return sorted(groups.values(), key=lambda row: row["group"].casefold())
