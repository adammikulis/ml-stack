"""Which devices are in a cluster, by certificate: the record every link off this machine is checked against.

A device is its certificate (`tls.identity`); its fingerprint is the SHA-256 of the DER. A request from
another machine is answered only when the certificate it showed in the TLS handshake is listed here as
active, and a revoked or unknown one is refused at its next handshake and its next request, even on a
connection already open. Revoking is one record that spreads by `merge`, and a revocation is never undone.
The join flows call `enrol` once they have authenticated the newcomer; `merge` takes rows only from a
peer that has itself authenticated as an active member.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import re
import ssl
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from poolhouse import home, lock
from poolhouse.files import write_json
from poolhouse.platform import private_file

__all__ = ["ACTIVE", "REVOKED", "Device", "Revoked", "Roster", "fingerprint_of", "roster"]

ACTIVE, REVOKED = "active", "revoked"
VERSION = 1
FINGERPRINT = re.compile(r"[0-9a-f]{64}")
MOST_ROWS = 4096


class Revoked(PermissionError):
    """A certificate that was put out of the cluster; it cannot be let back in."""


def fingerprint_of(cert: str) -> str:
    """The fingerprint of a certificate given as base64 DER, or ``""`` when it is not one."""
    try:
        der = base64.b64decode(cert, validate=True)
        ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))
    except (ValueError, binascii.Error, TypeError, ssl.SSLError):
        return ""
    return hashlib.sha256(der).hexdigest()


@dataclass(frozen=True, slots=True)
class Device:
    """One device's standing in one cluster."""

    fingerprint: str
    cert: str
    """The certificate, base64 DER; empty on a revocation recorded from a fingerprint alone."""
    name: str
    status: str
    at: float
    by: str
    """The fingerprint of the device that recorded this, or why (``pairing``, ``self``)."""

    @property
    def active(self) -> bool:
        return self.status == ACTIVE

    def public(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def read(cls, row: Any) -> Device | None:
        """A row from the file or a peer, or None when it is not what it claims: a fingerprint
        that is not the hash of the certificate beside it is a tampered pin and is dropped."""
        try:
            fingerprint, cert, status = str(row["fingerprint"]), str(row.get("cert", "")), str(row["status"])
            at, name, by = float(row.get("at", 0)), str(row.get("name", ""))[:80], str(row.get("by", ""))[:80]
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        if status not in (ACTIVE, REVOKED) or not FINGERPRINT.fullmatch(fingerprint):
            return None
        if cert and fingerprint_of(cert) != fingerprint:
            return None
        if status == ACTIVE and not cert:
            return None
        return cls(fingerprint, cert, name, status, at, by)


def _join(held: Device | None, new: Device) -> Device:
    """The standing two records of one device settle on: revoked wins, and stays."""
    if held is None or (new.status == REVOKED and held.status != REVOKED):
        return new
    if held.status == REVOKED and new.status == REVOKED and not held.cert and new.cert:
        return new
    return held


class Roster:
    """The devices of one cluster, kept in a file named for the cluster's key (so every part of
    this machine that holds the key finds the same record, whatever file the key came from)."""

    def __init__(self, key: bytes) -> None:
        self.file = home.state("fleet", "members", hashlib.sha256(key).hexdigest()[:32] + ".json")
        self._seen: tuple[Any, ...] = ()
        self._rows: dict[str, Device] = {}

    # -- reading -----------------------------------------------------------
    def stamp(self) -> tuple[int, int, int]:
        """What changes when the record does: lets a reader skip a file it already has."""
        try:
            info = self.file.stat()
        except OSError:
            return (0, 0, 0)
        return (info.st_mtime_ns, info.st_size, info.st_ino)

    def _load(self) -> dict[str, Device]:
        stamp = self.stamp()
        if stamp == self._seen:
            return self._rows
        rows: dict[str, Device] = {}
        with contextlib.suppress(OSError, ValueError):
            raw = json.loads(self.file.read_text())
            for row in (raw.get("devices", []) if isinstance(raw, dict) else [])[:MOST_ROWS]:
                if (device := Device.read(row)) is not None:
                    rows[device.fingerprint] = _join(rows.get(device.fingerprint), device)
        self._seen, self._rows = stamp, rows
        return rows

    def devices(self) -> list[Device]:
        return sorted(self._load().values(), key=lambda d: (d.name, d.fingerprint))

    def get(self, fingerprint: str) -> Device | None:
        return self._load().get(fingerprint)

    def is_active(self, fingerprint: str) -> bool:
        one = self._load().get(fingerprint)
        return one is not None and one.active

    def certs_pem(self) -> str:
        """Every active device's certificate as PEM: what a TLS server trusts a client to be."""
        return "".join(ssl.DER_cert_to_PEM_cert(base64.b64decode(d.cert))
                       for d in self.devices() if d.active)

    # -- writing -----------------------------------------------------------
    def _write(self, change: Any) -> int:
        """Apply ``change(rows)`` to the file under its lock; how many records it altered."""
        with lock.only_one(self.file.with_name(self.file.name + ".lock"), announce=lambda *_: None):
            self._seen = ()
            rows = dict(self._load())
            altered = change(rows)
            if altered:
                write_json(self.file, {"v": VERSION, "devices": [
                    d.public() for d in sorted(rows.values(), key=lambda d: d.fingerprint)]})
                private_file(self.file)
            self._seen = ()
        return altered

    def enrol(self, cert: str, name: str, by: str) -> Device:
        """Let a device in. A certificate that was revoked here stays out."""
        fingerprint = fingerprint_of(cert)
        if not FINGERPRINT.fullmatch(fingerprint):
            raise ValueError("a device is enrolled with its certificate, base64 DER")
        new = Device(fingerprint, cert, name[:80], ACTIVE, time.time(), by)

        def change(rows: dict[str, Device]) -> int:
            held = rows.get(fingerprint)
            if held is not None and not held.active:
                raise Revoked(f"{name or fingerprint[:12]} was put out of the cluster; it needs a new certificate")
            if held is None:
                rows[fingerprint] = new
                return 1
            return 0

        self._write(change)
        return self.get(fingerprint) or new

    def revoke(self, fingerprint: str, by: str) -> Device:
        """Put a device out: it is refused from its next handshake and its next request."""
        if not FINGERPRINT.fullmatch(fingerprint):
            raise ValueError("a device is revoked by its fingerprint, 64 hex digits")

        def change(rows: dict[str, Device]) -> int:
            held = rows.get(fingerprint)
            if held is not None and not held.active:
                return 0
            rows[fingerprint] = Device(fingerprint, held.cert if held else "", held.name if held else "",
                                       REVOKED, time.time(), by)
            return 1

        self._write(change)
        return self.get(fingerprint)  # type: ignore[return-value]

    def export(self) -> list[dict[str, Any]]:
        return [d.public() for d in self.devices()]

    def merge(self, rows: Iterable[Any], by: str) -> int:
        """Take what a peer that is an active member knows: new devices are added and every
        revocation is applied. Returns how many records changed."""
        heard = [d for row in list(rows)[:MOST_ROWS] if (d := Device.read(row)) is not None]

        def change(held: dict[str, Device]) -> int:
            altered = 0
            for device in heard:
                kept = held.get(device.fingerprint)
                cert = device.cert or (kept.cert if kept else "")
                new = _join(kept, Device(device.fingerprint, cert, device.name or (kept.name if kept else ""),
                                         device.status, device.at, device.by or by))
                if new is not kept:
                    held[device.fingerprint], altered = new, altered + 1
            return altered

        return self._write(change)

    def forget(self) -> None:
        """Drop the record when this machine leaves the cluster."""
        with contextlib.suppress(OSError):
            self.file.unlink()


def roster(key: bytes) -> Roster:
    """The record of the cluster whose key is ``key``."""
    return Roster(key)
