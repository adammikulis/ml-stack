"""Single-use cluster invitations and challenge proofs."""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import threading
import time
import urllib.parse
from typing import Any


def encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode(raw: str) -> bytes:
    if len(raw) > 8192:
        raise ValueError("invitation is too long")
    return base64.b64decode(raw + "=" * (-len(raw) % 4), altchars=b"-_", validate=True)


def proof(secret: bytes, fields: dict[str, Any], fingerprint: str) -> str:
    message = "ml-stack-invite/v1\n" + "\n".join(str(fields.get(k, "")) for k in
        ("id", "challenge", "kind")) + "\n" + fingerprint + "\n" + "\n".join(
        str(fields.get(k, "")) for k in ("platform", "device_name", "public_key"))
    return hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()


class Invitations:
    def __init__(self, members: Any, origin: Any):
        self.members = members
        self.origin = origin
        self.clock = time.time
        self.lock = threading.Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, list[float]] = {}

    def permit(self, source: str) -> None:
        with self.lock:
            now = self.clock()
            self.requests = {key: [t for t in times if t > now - 60]
                             for key, times in self.requests.items() if any(t > now - 60 for t in times)}
            if len(self.requests) >= 128 and source not in self.requests:
                raise ValueError("invitation exchange busy")
            times = self.requests.setdefault(source, [])
            if len(times) >= 20 or sum(map(len, self.requests.values())) >= 256:
                raise ValueError("invitation exchange rate limit")
            times.append(now)

    def _prune(self) -> None:
        self.rows = {k: v for k, v in self.rows.items() if v["expires"] > self.clock()}

    def mint(self, group: str, kind: str = "computer") -> dict[str, Any]:
        if not isinstance(group, str) or not 1 <= len(group) <= 128:
            raise ValueError("a bounded cluster name is required")
        if kind != "computer":
            raise ValueError("only owned computer invitations are available")
        member = next((m for m in self.members() if m.group == group), None)
        if member is None:
            raise ValueError("choose a cluster this computer belongs to")
        endpoint, fingerprint = self.origin()
        url = urllib.parse.urlsplit(endpoint)
        address = ipaddress.ip_address(url.hostname or "")
        if (url.scheme != "https" or not address.is_private or address.is_loopback
                or address.is_unspecified or address.is_multicast or url.username
                or url.password or url.path or url.query or url.fragment
                or not 1 <= (443 if url.port is None else url.port) <= 65535
                or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("a reachable LAN TLS listener is required")
        with self.lock:
            self._prune()
            if len(self.rows) >= 32:
                raise ValueError("revoke an invitation before creating another")
            ident, secret = secrets.token_hex(16), secrets.token_bytes(32)
            expires = int(self.clock()) + 600
            payload = {"v": 1, "endpoint": endpoint, "fingerprint": fingerprint, "id": ident,
                           "secret": encode(secret), "expires": expires, "kind": kind}
            self.rows[ident] = {"secret": secret, "expires": expires, "group": group,
                "binding": hashlib.sha256(member.key).digest(), "fingerprint": fingerprint,
                "challenges": {}, "attempts": 0, "issued": 0}
            invite = "ml-stack://enroll?data=" + encode(json.dumps(payload, separators=(",", ":")).encode())
            return {"id": ident, "invite": invite, "address": endpoint, "expires": expires}

    def revoke(self, ident: str) -> None:
        with self.lock:
            self.rows.pop(ident, None)

    def exchange(self, action: str, fields: dict[str, Any]) -> dict[str, Any]:
        for name, limit in (("id", 32), ("challenge", 64), ("proof", 64),
                            ("kind", 16), ("platform", 16), ("device_name", 128), ("public_key", 0)):
            value = fields.get(name, "")
            if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
                raise ValueError("invalid invitation field")
        if len(fields.get("id", "")) != 32:
            raise ValueError("invalid invitation id")
        if fields.get("kind") != "computer" or fields.get("platform") != "computer":
            raise ValueError("this invitation enrolls an owned computer")
        if not isinstance(fields.get("device_name"), str) or not 1 <= len(fields["device_name"]) <= 128:
            raise ValueError("a bounded device name is required")
        if fields.get("public_key", "") != "":
            raise ValueError("unsupported public key")
        with self.lock:
            self._prune()
            row = self.rows.get(fields.get("id"))
            if row is None:
                raise ValueError("invitation expired, revoked or already used")
            member = next((m for m in self.members() if m.group == row["group"]), None)
            if member is None or not hmac.compare_digest(hashlib.sha256(member.key).digest(), row["binding"]):
                raise ValueError("cluster membership changed")
            if self.origin()[1] != row["fingerprint"]:
                raise ValueError("listener certificate changed")
            challenges = row["challenges"]
            if action == "challenge":
                row["issued"] += 1
                if row["issued"] > 10:
                    raise ValueError("invitation challenge limit reached")
                challenges.clear()
                challenge = secrets.token_hex(32)
                challenges[challenge] = (self.clock() + 60, dict(fields))
                return {"challenge": challenge, "expires": min(row["expires"], int(self.clock()) + 60)}
            if action != "redeem":
                raise ValueError("unknown invitation action")
            claimed = challenges.pop(fields.get("challenge"), None)
            row["attempts"] += 1
            if row["attempts"] >= 3:
                self.rows.pop(fields["id"], None)
            if claimed is None or claimed[0] <= self.clock() or any(
                fields.get(k, "") != claimed[1].get(k, "") for k in
                ("id", "kind", "platform", "device_name", "public_key")):
                raise ValueError("challenge expired or changed")
            expected = proof(row["secret"], fields, row["fingerprint"])
            if not hmac.compare_digest(expected, str(fields.get("proof", ""))):
                raise ValueError("invitation proof did not match")
            self.rows.pop(fields["id"], None)
            grant = json.dumps({"kind": "computer", "group": member.group, "key": member.key.decode()},
                               separators=(",", ":")).encode()
            signature = hmac.new(row["secret"], ("ml-stack-invite-grant/v1\n" + fields["challenge"] + "\n").encode() + grant,
                                 hashlib.sha256).hexdigest()
            return {"grant_data": encode(grant), "proof": signature}
