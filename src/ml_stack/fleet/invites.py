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

from .membership import fingerprint_of
from .onboard.lan import in_tailnet


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
    def __init__(self, members: Any, origin: Any, enrol: Any = None):
        self.members = members
        self.origin = origin
        self.enrol = enrol
        """``enrol(group, certificate, name)`` lists a computer that redeemed an invitation as a device of the cluster."""
        self.clock = time.time
        self.lock = threading.Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, list[float]] = {}
        self.devices: dict[str, dict[str, Any]] = {}

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
        self.devices = {k: v for k, v in self.devices.items() if v["expires"] > self.clock()}

    def mint(self, group: str, kind: str = "computer") -> dict[str, Any]:
        if not isinstance(group, str) or not 1 <= len(group) <= 128:
            raise ValueError("a bounded cluster name is required")
        if kind not in {"computer", "android"}:
            raise ValueError("choose an owned computer or Android device")
        member = next((m for m in self.members() if m.group == group), None)
        if member is None:
            raise ValueError("choose a cluster this computer belongs to")
        endpoint, fingerprint = self.origin()
        url = urllib.parse.urlsplit(endpoint)
        address = ipaddress.ip_address(url.hostname or "")
        if (url.scheme != "https" or not (address.is_private or in_tailnet(str(address))) or address.is_loopback
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
            self.rows[ident] = {"secret": secret, "expires": expires, "group": group, "kind": kind,
                "binding": hashlib.sha256(member.key).digest(), "fingerprint": fingerprint,
                "endpoint": endpoint,
                "challenges": {}, "attempts": 0, "issued": 0}
            invite = "ml-stack://enroll?data=" + encode(json.dumps(payload, separators=(",", ":")).encode())
            return {"id": ident, "invite": invite, "address": endpoint, "expires": expires}

    def revoke(self, ident: str) -> None:
        with self.lock:
            self.rows.pop(ident, None)
            self.devices.pop(ident, None)

    def active_devices(self) -> list[dict[str, Any]]:
        with self.lock:
            self._prune()
            return [{"device_id": ident, "device_name": row["device_name"],
                     "group": row["group"], "expires": row["expires"]}
                    for ident, row in self.devices.items()]

    def authorize(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) != 43:
            raise ValueError("Android session required")
        digest = hashlib.sha256(token.encode()).digest()
        with self.lock:
            self._prune()
            row = next((v for v in self.devices.values() if hmac.compare_digest(v["token"], digest)), None)
            if row is None:
                raise ValueError("Android session expired or revoked")
            member = next((m for m in self.members() if m.group == row["group"]), None)
            if (member is None or not hmac.compare_digest(hashlib.sha256(member.key).digest(), row["binding"])
                    or self.origin() != (row["endpoint"], row["fingerprint"])):
                raise ValueError("Android session authority changed")
            return dict(row)

    def exchange(self, action: str, fields: dict[str, Any]) -> dict[str, Any]:
        for name, limit in (("id", 32), ("challenge", 64), ("proof", 64),
                            ("kind", 16), ("platform", 16), ("device_name", 128), ("public_key", 2048)):
            value = fields.get(name, "")
            if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
                raise ValueError("invalid invitation field")
        if len(fields.get("id", "")) != 32:
            raise ValueError("invalid invitation id")
        if fields.get("kind") not in {"computer", "android"} or fields.get("platform") != fields.get("kind"):
            raise ValueError("unsupported device platform")
        if not isinstance(fields.get("device_name"), str) or not 1 <= len(fields["device_name"]) <= 128:
            raise ValueError("a bounded device name is required")
        if (fields.get("public_key", "") != "") == (fields["kind"] == "android") or (
                fields["kind"] == "computer" and not fingerprint_of(fields["public_key"])):
            raise ValueError("a computer joins with its device certificate; a phone has none")
        with self.lock:
            self._prune()
            row = self.rows.get(fields.get("id"))
            if row is None:
                raise ValueError("invitation expired, revoked or already used")
            if row["kind"] != fields["kind"]:
                raise ValueError("invitation device kind changed")
            if row["kind"] == "android" and len(fields["device_name"]) > 80:
                raise ValueError("Android device name is too long")
            member = next((m for m in self.members() if m.group == row["group"]), None)
            if member is None or not hmac.compare_digest(hashlib.sha256(member.key).digest(), row["binding"]):
                raise ValueError("cluster membership changed")
            if self.origin() != (row["endpoint"], row["fingerprint"]):
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
            payload = self._grant(row, fields, member)
            self.rows.pop(fields["id"], None)
            if row["kind"] == "computer" and self.enrol is not None:
                self.enrol(member.group, fields["public_key"], fields["device_name"])
            grant = json.dumps(payload, separators=(",", ":")).encode()
            signature = hmac.new(row["secret"], ("ml-stack-invite-grant/v1\n" + fields["challenge"] + "\n").encode() + grant,
                                 hashlib.sha256).hexdigest()
            return {"grant_data": encode(grant), "proof": signature}

    def _grant(self, row: dict, fields: dict, member: Any) -> dict[str, Any]:
        if row["kind"] != "android":
            return {"kind": "computer", "group": member.group, "key": member.key.decode(), "mode": member.mode}
        if len(self.devices) >= 128:
            raise ValueError("revoke an Android device before enrolling another")
        ident, token = secrets.token_hex(16), encode(secrets.token_bytes(32))
        expires = int(self.clock()) + 3600
        endpoint, fingerprint = self.origin()
        self.devices[ident] = {"token": hashlib.sha256(token.encode()).digest(),
            "binding": row["binding"], "group": member.group, "expires": expires,
            "endpoint": endpoint, "fingerprint": fingerprint, "device_name": fields["device_name"]}
        return {"kind": "android", "device_id": ident, "token": token, "expires": expires,
            "capabilities": ["fleet.status", "chat"], "endpoint": endpoint,
            "fingerprint": fingerprint, "group": member.group}
